import os
import uuid
import re
import shutil
import json
import time
import asyncio
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from pipeline.transcribe import decode_audio, transcribe_and_diarize
from pipeline.spellcheck import spell_check_enabled, spell_check_segments
from pipeline.llm_cleanup import CleanupSettings, run_llm_cleanup
from pipeline.self_identification import suggest_speaker_names
from pipeline.renderer import render_markdown

logger = logging.getLogger("api.main")

# uvicorn installs handlers only for its own loggers; without a root handler
# at INFO, the pipeline's per-stage VRAM instrumentation lines (AC3 evidence)
# never reach `docker logs`. Configure the root before any pipeline import.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger().setLevel(logging.INFO)

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/autodiarist_jobs"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Queue-hardening limits (card t_daebf369). Read at request/boot time (not
# import time) so a deployment can tune them via env without code changes.
DEFAULT_MAX_QUEUE_DEPTH = 50
DEFAULT_RETRY_AFTER_SECONDS = 30
DEFAULT_QUEUE_POLL_SECONDS = 0.2
# MAX_JOBS_PER_SESSION is DEPRECATED as of V1.5 (card t_82bfbc15): a session
# may hold any number of active jobs; the env var is still accepted so
# existing .env files load unchanged, but validate_config() warns that it is
# ignored and admission is queue-depth only.
# Which header names the session identity; default X-Session. "Authorization"
# stays honored when explicitly configured via SESSION_HEADER.
SESSION_HEADER_DEFAULT = "X-Session"
# Server-issued session id cookie (V1.5, card t_920fb7dc). The value is also
# returned in the POST body so API clients without a cookie jar can reuse it.
# Name is overridable via SESSION_COOKIE_NAME for proxies that reserve it.
SESSION_COOKIE_NAME_DEFAULT = "autodiarist_session"
# Fair-share scheduling (V1.5, card t_4b40acfd): the worker claims the next
# job round-robin across sessions (each session keeps FIFO order among its own
# jobs), and a session may hold at most MAX_PROCESSING_PER_SESSION in-flight
# jobs — enforced by the scheduler itself, not by the single-worker accident.
# The round-robin pointer persists to JOBS_DIR/<SCHEDULER_STATE_NAME> so
# fairness survives a restart.
DEFAULT_MAX_PROCESSING_PER_SESSION = 1
# Global concurrency (V1.5.1, card t_dbad2235): the queue used to be serial
# BY CONSTRUCTION — one worker task awaiting each job to completion before
# claiming the next. MAX_PROCESSING_GLOBAL spawns N concurrent workers and
# enforces N in-flight jobs AT CLAIM TIME (the atomic _claim_next_queued
# check is what prevents double-claim; the per-session cap is untouched).
# Default MUST remain 1 so an existing deployment is byte-for-byte unchanged
# until it opts in; per-host values come from MEASUREMENT (measured on the
# T400: N=3 sustains 3/3 engines at 3207/4096 MiB — see CONFIG.md), never
# from a guess.
DEFAULT_MAX_PROCESSING_GLOBAL = 1
SCHEDULER_STATE_NAME = "_scheduler.json"
# Session-scoped job listing (V1.5, card t_f97d12a8): default page size and
# hard max for GET /v1/jobs. Deliberately constants, not env knobs (scope 8:
# "keep it knob-free"). ?limit= outside [1, MAX_LIST_LIMIT] clamps, never 4xx.
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
# Queue-aware wait estimate (V1.5, card t_b3ae9366). ETA comes from ONE
# linear model — duration_seconds x PROCESSING_SECONDS_PER_AUDIO_SECOND —
# summed over the fair execution order. It is an ESTIMATE, never a
# guarantee: it ignores diarization variance and chunk overhead, and
# observed wall time can differ (short files land worst — fixed startup
# cost dominates; the long-file asymptote is what the default is tuned to).
#
# The default rate is HARDWARE-SPECIFIC and documented at its source: the
# V1.5 T400 (4 GB Turing) integration run measured a 39-minute meeting
# (2340 s of audio) at 423-434 s wall on CUDA — 434/2340 ~= 0.185 s of GPU
# time per s of audio. Re-derive it for any other GPU (see CONFIG.md).
# Set the env var to 0 to disable ETA emission entirely.
DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND = 0.185
# Substituted into the prefix sum for a queued job AHEAD whose admission
# probe failed (duration unknown) — one unprobeable file must not poison
# every estimate behind it. The polled job itself gets NO ETA when its own
# duration is unknown: the API never fabricates a duration it does not
# have, and never re-probes at request time. Configurable via
# AUDIO_DURATION_FALLBACK_SECONDS.
DEFAULT_AUDIO_DURATION_SECONDS = 300.0
# Running-phase ETA (card t_d64bdf29). Once a job is PROCESSING its
# completion estimate is derived from its OWN observed rate — wall time
# since the attempt started against the published monotonic progress
# fraction — NEVER from the fixed rate constant above (that constant only
# gates emission here; it remains the queue side's model). The job gets
# null until it has genuinely crossed a meaningful band: the default
# 0.30 is the transcribing band's lower edge, so decode + startup cost
# is fully behind the elapsed-time observation and the derived rate
# reflects steady-state work. Tunable via RUNNING_ETA_MIN_PROGRESS.
DEFAULT_RUNNING_ETA_MIN_PROGRESS = 0.30
# Freshness allowance for the last valid running-phase estimate while
# progress is frozen (a stalled stage, or a band that publishes no
# fine-grained work — spellcheck / llm_cleanup). Beyond this window the
# estimate goes null rather than persisting a value reality has not
# reconfirmed. Tunable via RUNNING_ETA_REANCHOR_SECONDS.
DEFAULT_RUNNING_ETA_REANCHOR_SECONDS = 120.0
# HTTP token charset (RFC 7230) — header/cookie names must match it.
_HEADER_TOKEN_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


def _validated_env_name(var: str, default: str) -> str:
    """Read a header/cookie NAME knob, falling back to the documented default."""
    raw = (os.environ.get(var) or "").strip()
    if not raw:
        return default
    if not _HEADER_TOKEN_RE.fullmatch(raw):
        logger.warning("config: %s=%r is not a valid HTTP token; using %r", var, raw, default)
        return default
    return raw


def _session_header_name() -> str:
    return _validated_env_name("SESSION_HEADER", SESSION_HEADER_DEFAULT)


def _session_cookie_name() -> str:
    return _validated_env_name("SESSION_COOKIE_NAME", SESSION_COOKIE_NAME_DEFAULT)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _max_processing_per_session() -> int:
    """Cap on concurrently-PROCESSING jobs per session (card t_4b40acfd).

    Read at every claim so a deployment can tune it via env. Values < 1 or
    unparseable input warn and clamp to the default: a zero cap would freeze
    the queue entirely, exactly the MAX_QUEUE_DEPTH=0 footgun.
    """
    raw = (os.environ.get("MAX_PROCESSING_PER_SESSION") or "").strip()
    if not raw:
        return DEFAULT_MAX_PROCESSING_PER_SESSION
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "config: MAX_PROCESSING_PER_SESSION=%r is not an integer; using %d",
            raw, DEFAULT_MAX_PROCESSING_PER_SESSION,
        )
        return DEFAULT_MAX_PROCESSING_PER_SESSION
    if value < 1:
        logger.warning(
            "config: MAX_PROCESSING_PER_SESSION=%d would freeze the queue; clamped to %d",
            value, DEFAULT_MAX_PROCESSING_PER_SESSION,
        )
        return DEFAULT_MAX_PROCESSING_PER_SESSION
    return value


def _max_processing_global() -> int:
    """Cap on concurrently-PROCESSING jobs across the WHOLE service
    (V1.5.1, card t_dbad2235).

    Read at every claim so the service never starts a job it cannot run, and
    once at startup to size the worker pool. Lowering the env at runtime
    tightens the cap immediately; RAISING it above the live worker count only
    takes effect after a restart (the pool is fixed at boot). Values < 1 or
    unparseable input warn and clamp to the default, exactly like
    _max_processing_per_session(): a zero worker count would freeze the queue
    entirely. Default 1 preserves today's serial behaviour for every existing
    deployment until an operator opts in.
    """
    raw = (os.environ.get("MAX_PROCESSING_GLOBAL") or "").strip()
    if not raw:
        return DEFAULT_MAX_PROCESSING_GLOBAL
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "config: MAX_PROCESSING_GLOBAL=%r is not an integer; using %d",
            raw, DEFAULT_MAX_PROCESSING_GLOBAL,
        )
        return DEFAULT_MAX_PROCESSING_GLOBAL
    if value < 1:
        logger.warning(
            "config: MAX_PROCESSING_GLOBAL=%d would freeze the queue; clamped to %d",
            value, DEFAULT_MAX_PROCESSING_GLOBAL,
        )
        return DEFAULT_MAX_PROCESSING_GLOBAL
    return value


def _processing_rate() -> float:
    """Seconds of GPU processing per second of audio (card t_b3ae9366).

    Read at every estimate so a deployment can retune via env. EXACT 0 is
    the documented switch that disables ETA emission. Negative or
    non-numeric warn and fall back to the default — mirroring the
    _max_processing_per_session style. The default is the measured T400
    rate (0.185 = 434 s wall for 2340 s of audio); on other hardware it
    is off by roughly the CUDA-vs-CPU factor (see CONFIG.md).
    """
    raw = (os.environ.get("PROCESSING_SECONDS_PER_AUDIO_SECOND") or "").strip()
    if not raw:
        return DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "config: PROCESSING_SECONDS_PER_AUDIO_SECOND=%r is not a number; using %.3f",
            raw, DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND,
        )
        return DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND
    if value < 0:
        logger.warning(
            "config: PROCESSING_SECONDS_PER_AUDIO_SECOND=%r is negative; using %.3f "
            "(0 is the documented way to disable ETA emission)",
            value, DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND,
        )
        return DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND
    return value


def _running_eta_min_progress() -> float:
    """Fraction a running job must cross before it gets an estimate
    (card t_d64bdf29 AC4). Out-of-range or non-numeric values warn and
    fall back to the default, mirroring _processing_rate's discipline."""
    raw = (os.environ.get("RUNNING_ETA_MIN_PROGRESS") or "").strip()
    if not raw:
        return DEFAULT_RUNNING_ETA_MIN_PROGRESS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "config: RUNNING_ETA_MIN_PROGRESS=%r is not a number; using %.2f",
            raw, DEFAULT_RUNNING_ETA_MIN_PROGRESS)
        return DEFAULT_RUNNING_ETA_MIN_PROGRESS
    if not 0.0 < value < 1.0:
        logger.warning(
            "config: RUNNING_ETA_MIN_PROGRESS=%r is outside (0, 1); using %.2f",
            raw, DEFAULT_RUNNING_ETA_MIN_PROGRESS)
        return DEFAULT_RUNNING_ETA_MIN_PROGRESS
    return value


def _running_eta_reanchor_seconds() -> float:
    """Freshness allowance for a frozen running-phase estimate (card
    t_d64bdf29): wall seconds progress may stand still before the last
    valid value is dropped (null) instead of held stale."""
    raw = (os.environ.get("RUNNING_ETA_REANCHOR_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_RUNNING_ETA_REANCHOR_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "config: RUNNING_ETA_REANCHOR_SECONDS=%r is not a number; using %.1f",
            raw, DEFAULT_RUNNING_ETA_REANCHOR_SECONDS)
        return DEFAULT_RUNNING_ETA_REANCHOR_SECONDS
    if value <= 0:
        logger.warning(
            "config: RUNNING_ETA_REANCHOR_SECONDS=%r must be positive; using %.1f",
            raw, DEFAULT_RUNNING_ETA_REANCHOR_SECONDS)
        return DEFAULT_RUNNING_ETA_REANCHOR_SECONDS
    return value


# --- Job retention + disk-pressure eviction (card t_5e6200fc) -----------------
# RETENTION_DAYS is no longer a dead knob: the age sweep deletes FINISHED job
# dirs (done/failed/cancelled — AC "cancelled/failed age out too") whose
# status.json mtime is older than RETENTION_DAYS, at boot and hourly after.
# RETENTION_DAYS=0 disables age-based expiry. The live pair (queued,
# processing) is never age-swept: queued jobs must survive the boot-requeue
# window (requeue runs before the pass) and a silently-expired in-flight job
# is exactly the failure mode the original README caveat warned about.
# Disk pressure: when free space on the JOBS_DIR filesystem drops below
# DISK_MIN_FREE_MIB, evict job dirs oldest-mtime-first until the watermark is
# recovered; a 'processing' job is never evicted (AC4) and every removal logs
# token + reason. Eviction scope is JOBS_DIR only — the mounted CrispASR
# cache volume is out of scope for this card.
DEFAULT_RETENTION_DAYS = 30
RETENTION_SWEEP_INTERVAL_SECONDS = 3600  # hourly age sweep after the boot pass
DEFAULT_DISK_MIN_FREE_MIB = 8192
DEFAULT_DISK_CHECK_INTERVAL_SECONDS = 300

_LIVE_STATUSES = frozenset({"queued", "processing"})
_MIB = 1024 * 1024


def _retention_days() -> int:
    return max(0, _env_int("RETENTION_DAYS", DEFAULT_RETENTION_DAYS))


def _disk_min_free_mib() -> int:
    return max(0, _env_int("DISK_MIN_FREE_MIB", DEFAULT_DISK_MIN_FREE_MIB))


def _disk_check_interval_seconds() -> int:
    return max(1, _env_int("DISK_CHECK_INTERVAL_SECONDS", DEFAULT_DISK_CHECK_INTERVAL_SECONDS))


def _job_status(job_dir: Path):
    """Job status string, or None when status.json is unreadable/corrupt."""
    try:
        return json.loads((job_dir / "status.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def _job_mtime(job_dir: Path) -> float:
    """Expiry clock: status.json mtime when present, else the dir mtime
    (covers dirs torn down before the first status write landed)."""
    try:
        return (job_dir / "status.json").stat().st_mtime
    except OSError:
        return job_dir.stat().st_mtime


def _iter_job_dirs(jobs_dir: Path) -> list[Path]:
    try:
        return [p for p in jobs_dir.iterdir() if p.is_dir()]
    except OSError:
        return []


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _free_bytes(jobs_dir: Path) -> int:
    return shutil.disk_usage(jobs_dir).free


def sweep_expired_jobs(jobs_dir: Path, retention_days: int, now: float | None = None) -> list[str]:
    """Delete finished job dirs older than retention_days (AC1).

    retention_days <= 0 disables age-based expiry entirely (AC2). Every
    removal is logged with the token and reason=age-expiry. The live pair
    and dirs with no readable status.json are left alone; disk-pressure
    eviction is their only reclaim path."""
    if retention_days <= 0:
        return []
    cutoff = (time.time() if now is None else now) - retention_days * 86400.0
    removed: list[str] = []
    for job_dir in _iter_job_dirs(jobs_dir):
        mtime = _job_mtime(job_dir)
        if mtime >= cutoff:
            continue
        status = _job_status(job_dir)
        if status is None or status in _LIVE_STATUSES:
            continue
        shutil.rmtree(job_dir, ignore_errors=True)
        removed.append(job_dir.name)
        logger.info(
            "retention sweep: removed job token=%s status=%s age_days=%.1f reason=age-expiry",
            job_dir.name,
            status,
            (time.time() - mtime) / 86400.0,
        )
    return removed


def enforce_disk_pressure(jobs_dir: Path, min_free_mib: int, free_bytes_fn=None) -> list[str]:
    """Evict job dirs oldest-mtime-first until free space >= the watermark (AC3).

    min_free_mib <= 0 disables eviction before any filesystem probe. A job in
    'processing' state is NEVER evicted (AC4); queued jobs are evictable —
    upload admission reuses seq slots after a purge, so a dropped queued
    dir is not a dangling queue entry. Dirs with no readable status.json are
    skipped (garbage like .upload-* temp dirs is reclaimed by its own
    writers; deleting half-written state under a live fd is worse than
    staying under water for a cycle). When nothing evictable remains, the
    pass logs a warning and exits — it must not spin against the probe."""
    if min_free_mib <= 0:
        return []
    probe = free_bytes_fn or _free_bytes
    watermark = min_free_mib * _MIB
    evicted: list[str] = []
    while True:
        free = probe(jobs_dir)
        if free >= watermark:
            break
        candidates = []
        for job_dir in _iter_job_dirs(jobs_dir):
            status = _job_status(job_dir)
            if status is None or status == "processing":
                continue
            candidates.append((_job_mtime(job_dir), job_dir, status))
        if not candidates:
            logger.warning(
                "disk pressure: free=%d bytes below watermark=%d MiB on %s; "
                "no evictable job dirs remain (processing jobs are never evicted)",
                free,
                min_free_mib,
                jobs_dir,
            )
            break
        candidates.sort(key=lambda item: item[0])
        _, victim, status = candidates[0]
        freed = _dir_size_bytes(victim)
        shutil.rmtree(victim, ignore_errors=True)
        evicted.append(victim.name)
        logger.warning(
            "retention eviction: removed job token=%s status=%s freed_bytes=%d "
            "reason=disk-pressure watermark_mib=%d",
            victim.name,
            status,
            freed,
            min_free_mib,
        )
    return evicted


def _age_sweep_pass() -> list[str]:
    return sweep_expired_jobs(JOBS_DIR, _retention_days())


def _disk_pressure_pass() -> list[str]:
    return enforce_disk_pressure(JOBS_DIR, _disk_min_free_mib())


def run_retention_boot_pass() -> None:
    """One synchronous age+disk pass at boot, before the interval loops take
    over: sweep work that came due while the service was down runs first."""
    removed = _age_sweep_pass()
    evicted = _disk_pressure_pass()
    if removed or evicted:
        logger.info(
            "retention boot pass: %d age-expired, %d disk-evicted", len(removed), len(evicted)
        )


async def _retention_loop(sleep_func=None, interval_seconds=None) -> None:
    """Hourly age sweep (AC5). The boot pass is run separately by the
    lifespan, so this loop sleeps first. sleep_func/interval are injectable
    for deterministic tests."""
    sleep_func = sleep_func or asyncio.sleep
    if interval_seconds is None:
        interval_seconds = RETENTION_SWEEP_INTERVAL_SECONDS
    while True:
        await sleep_func(interval_seconds)
        _age_sweep_pass()


async def _disk_pressure_loop(sleep_func=None, interval_seconds=None) -> None:
    """Disk-pressure check every DISK_CHECK_INTERVAL_SECONDS (AC5)."""
    sleep_func = sleep_func or asyncio.sleep
    if interval_seconds is None:
        interval_seconds = _disk_check_interval_seconds()
    while True:
        await sleep_func(interval_seconds)
        _disk_pressure_pass()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Pipeline stages & progress model (card t_14ac6b49).
#
# Every status.json carries a machine-readable ``stage`` — one of
# STAGE_ORDER below — plus a ``progress`` float in [0, 1]. Each stage owns
# the progress band [its value, the next stage's value): publishing a stage
# starts at its band's lower edge, and fine-grained within-band fractions
# (engine pass/chunk counts) interpolate inside it. Progress is enforced
# monotonic non-decreasing in update_state(), equals exactly 1.0 at ``done``,
# and a failed job keeps the last monotonic value (never 1.0) while ``stage``
# marks where it broke. Values are ordinal markers of real work boundaries,
# not time estimates.
# ---------------------------------------------------------------------------

STAGE_PROGRESS = {
    "queued": 0.00,
    "decoding": 0.05,
    "transcribing": 0.30,
    "diarizing": 0.60,
    "spellcheck": 0.75,
    # llm_cleanup is published only on opt-in runs (an injected ``llm``
    # seam, card t_48c2efae); HTTP/queue runs skip the stage entirely.
    "llm_cleanup": 0.80,
    "self_identification": 0.85,
    "rendering": 0.92,
    "done": 1.00,
}
STAGE_ORDER = list(STAGE_PROGRESS)

# Engine progress callbacks use terse pass names; map them to public stages.
_ENGINE_STAGES = {"asr": "transcribing", "diar": "diarizing"}


def _stage_band(stage: str) -> tuple[float, float]:
    idx = STAGE_ORDER.index(stage)
    start = STAGE_PROGRESS[stage]
    end = STAGE_PROGRESS[STAGE_ORDER[idx + 1]] if idx + 1 < len(STAGE_ORDER) else 1.0
    return start, end


def _stage_progress(stage: str, fraction: float = 0.0) -> float:
    """Interpolate progress inside ``stage``'s band at ``fraction`` done."""
    start, end = _stage_band(stage)
    return round(start + max(0.0, min(1.0, fraction)) * (end - start), 4)


# ---------------------------------------------------------------------------
# Persistent FIFO job queue.
#
# The queue lives entirely on disk: each job dir holds a status.json carrying
# {status, seq, session, queued_at, ...}. `seq` is a monotonic admission
# sequence assigned under _queue_lock and defines FIFO pickup order, which is
# exactly the order clients see as queue_position. A restart (fresh process or
# fresh lifespan) rebuilds the queue by scanning JOBS_DIR, so queued jobs
# survive and admission limits are revalidated from persisted state.
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()

TOKEN_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

def validate_token(token: str):
    if not TOKEN_REGEX.match(token):
        raise HTTPException(status_code=422, detail="Invalid token format")


def _state_path(token: str) -> Path:
    return JOBS_DIR / token / "status.json"


def _read_state(token: str) -> dict | None:
    p = _state_path(token)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        logger.warning("unreadable status.json for %s", token)
        return None


def _journal_append(job_dir: Path, st_data: dict) -> None:
    """Append-only lifecycle journal.

    Transient stage publishes are overwritten on status.json within
    milliseconds, so "stage transitions appear in order" must remain
    auditable after the fact (card t_14ac6b49; QA/post-mortem evidence).
    Journaling never fails a job.
    """
    try:
        with open(job_dir / "status_journal.jsonl", "a", encoding="utf-8") as jfh:
            jfh.write(json.dumps({
                "at": round(time.time(), 4),
                "status": st_data.get("status"),
                "stage": st_data.get("stage"),
                "progress": st_data.get("progress"),
                "message": st_data.get("message"),
            }) + "\n")
    except OSError:
        logger.warning("journal append failed for %s", st_data.get("token", "?"))


def _write_state(state: dict) -> None:
    """Atomic state write; callers hold _queue_lock on transition paths."""
    token = state["token"]
    job_dir = JOBS_DIR / token
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, job_dir / "status.json")
    _journal_append(job_dir, state)


def _iter_states() -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    states = []
    for entry in JOBS_DIR.iterdir():
        if not entry.is_dir():
            continue
        st = _read_state(entry.name)
        if st and "status" in st:
            states.append(st)
    return states


def _queued_states(states: list[dict] | None = None) -> list[dict]:
    states = _iter_states() if states is None else states
    return sorted(
        (s for s in states if s.get("status") == "queued"),
        key=lambda s: (s.get("seq", float("inf")), s.get("token", "")),
    )


# --- Fair-share scheduling (card t_4b40acfd) ----------------------------------
# Per-session FIFO sub-queues served round-robin. The pointer (which session
# was served when) persists under JOBS_DIR so a restart continues the same
# round instead of letting a big submitter re-jump the line.

def _scheduler_state_path() -> Path:
    return JOBS_DIR / SCHEDULER_STATE_NAME


def _load_scheduler_state() -> dict:
    """Read the round-robin pointer; a fresh process derives fairness from
    this file alone (claim_next never trusts an in-memory copy).

    Missing file or malformed JSON heals to a fresh state with a warning:
    the worst case is one round of tie-break-by-arrival, never a crash."""
    fresh = {"serve_counter": 0, "last_served": {}}
    path = _scheduler_state_path()
    if not path.exists():
        return fresh
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "scheduler state: unreadable %s (%s) — resetting round-robin pointer",
            path, exc,
        )
        return fresh
    if not isinstance(raw, dict):
        logger.warning("scheduler state: malformed %s — resetting round-robin pointer", path)
        return fresh
    try:
        counter = max(0, int(raw.get("serve_counter", 0)))
        last = {str(k): int(v) for k, v in (raw.get("last_served") or {}).items()}
    except (TypeError, ValueError):
        logger.warning("scheduler state: malformed %s — resetting round-robin pointer", path)
        return fresh
    return {"serve_counter": counter, "last_served": last}


def _save_scheduler_state(sched: dict) -> None:
    path = _scheduler_state_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sched), encoding="utf-8")
    tmp.replace(path)


def _session_key(state: dict) -> str:
    """Grouping key for the per-session sub-queue: the dedicated hashed
    session_ident (V1.5) with graceful fallbacks for legacy states."""
    return str(state.get("session_ident") or state.get("session") or "_anonymous")


def _queued_groups(states: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for st in _queued_states(states):
        groups.setdefault(_session_key(st), []).append(st)
    return groups


def _rr_pick(groups: dict[str, list[dict]], served: dict[str, int]) -> str:
    """Round-robin turn: the session served longest ago wins (never-served
    sessions count as served at 0, so new arrivals cannot starve behind an
    established submitter); ties break by earliest FIFO head seq, then the
    session key for full determinism."""
    return min(groups, key=lambda k: (served.get(k, 0), groups[k][0].get("seq", float("inf")), k))


def _fair_queued_order(states: list[dict], sched: dict) -> list[str]:
    """Pure simulation of the fair pickup order over the queued jobs — the
    sequence _claim_next_queued() would produce if nothing new arrived and no
    cap blocked a session (see README: queue_position is a rank in THIS
    order). Does not mutate inputs or the scheduler file."""
    groups = _queued_groups(states)
    served = dict(sched["last_served"])
    counter = sched["serve_counter"]
    order: list[str] = []
    while groups:
        key = _rr_pick(groups, served)
        st = groups[key].pop(0)
        if not groups[key]:
            del groups[key]
        order.append(str(st.get("token", "")))
        counter += 1
        served[key] = counter
    return order


# --- Queue-aware wait estimate (card t_b3ae9366) ------------------------------


def _known_duration_seconds(state: dict) -> float | None:
    """The job's probed audio duration, or None when it is unknown/zero.

    Only ever read from job state — the API never re-probes at request
    time (AC5): the value is the ffprobe container-header result recorded
    at admission, refreshed once by the decode-time probe when the job
    actually runs.
    """
    try:
        dur = float(state.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        return None
    return dur if dur > 0 else None


def _estimate_seconds_for(state: dict, rate: float) -> float:
    """Linear model cost of one job: duration_seconds x rate.

    A job whose duration is unknown substitutes the documented fallback
    constant so one unprobeable file cannot poison the estimates of every
    job queued behind it; the owning client still gets no ETA for its own
    unknown-duration job (see _wait_estimates)."""
    dur = _known_duration_seconds(state)
    if dur is None:
        dur = max(0.0, _env_float("AUDIO_DURATION_FALLBACK_SECONDS",
                                  DEFAULT_AUDIO_DURATION_SECONDS))
    return dur * rate


def _processing_remaining_seconds(processing: dict | None, rate: float,
                                  now: float) -> float:
    """The currently-PROCESSING job's honest remaining share (never > est).

    With started_at (every real claim sets one; reclaimed attempts preserve
    it): remaining = estimate - wall time so far, clamped to [0, estimate]
    — a straggler that overshoots its linear estimate contributes 0, it
    never subtracts from the prefix (AC5 negative clamp). The value falls
    monotonically as `now` advances, so a polled estimate only moves when
    queue members genuinely leave (AC4). Without started_at (seeded or
    pre-V1.5 state), the monotonic progress fraction is the fallback."""
    if processing is None:
        return 0.0
    est = _estimate_seconds_for(processing, rate)
    started = processing.get("started_at")
    try:
        if started is not None:
            remaining = est - (now - float(started))
        else:
            prog = float(processing.get("progress") or 0.0)
            remaining = est * (1.0 - min(1.0, max(0.0, prog)))
    except (TypeError, ValueError):
        remaining = est
    return max(0.0, min(est, remaining))


# --- Running-phase ETA (card t_d64bdf29) --------------------------------------
#
# A PROCESSING job's completion estimate comes from its OWN observed rate:
#     fraction = published monotonic progress (stage bands on real work
#                boundaries — decode finish, engine chunk counts, ...)
#     rate     = fraction / wall seconds since the attempt started
#     remaining = (1 - fraction) / rate
# The fixed duration-rate model NEVER feeds this number (AC1); it only
# contributes the startup-cost emission gate below. Two honesty devices
# keep the payload trustworthy:
#   * emission gates (AC4): null until the job crosses
#     RUNNING_ETA_MIN_PROGRESS and, when its duration is known, has lived
#     longer than the linear startup-cost floor — a fresh job in warm-up
#     has no meaningful rate observation;
#   * the publication ratchet (_running_eta_ratchet): the published value
#     never inflates, re-anchors only down on real progress, and goes
#     null — never stale — once progress freezes past
#     RUNNING_ETA_REANCHOR_SECONDS or a held value passes its own
#     promised completion time (fresh or null, per the card).


def _progress_percent(state: dict) -> int:
    """Integer 0..100 view of the banded float `progress` (card t_d64bdf29
    AC6): clients render a bar without inventing their own rounding, and
    the value stays consistent with the stage bands by construction (it is
    the same published number, rounded) and monotonic (progress itself is
    clamped non-decreasing). Unparseable/missing progress -> 0."""
    try:
        value = float(state.get("progress") or 0.0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, round(value * 100)))


def _running_estimate(state: dict, now: float) -> tuple[float, float] | None:
    """Pure observed-rate math: (remaining_seconds, completion_epoch) or
    None when no honest estimate exists yet. Never touches the ratchet."""
    if state.get("status") != "processing":
        return None
    started = state.get("attempt_started_at") or state.get("started_at")
    if started is None:
        return None
    try:
        elapsed = now - float(started)
        frac = float(state.get("progress") or 0.0)
    except (TypeError, ValueError):
        return None
    if elapsed <= 0 or not 0.0 < frac < 1.0:
        return None
    if frac <= _running_eta_min_progress():
        return None
    dur = _known_duration_seconds(state)
    if dur is not None:
        # Startup-cost floor: while elapsed is shorter than the linear
        # model's estimate for everything BEFORE steady transcription
        # (the queued->transcribing band), the observation window is
        # dominated by fixed cost and the derived rate would be a lie.
        # Gate only — the emitted value below never uses this model.
        if elapsed <= dur * _processing_rate() * STAGE_PROGRESS["transcribing"]:
            return None
    remaining = elapsed * (1.0 - frac) / frac
    return remaining, now + remaining


_running_eta_floor: dict[str, dict] = {}


def _running_eta_ratchet(state: dict, now: float | None = None,
                         ) -> tuple[float, float] | None:
    """Publication memory for the running-phase estimate (AC2 + freshness).

    Contract across consecutive polls of one attempt: a published value
    never moves UP; when progress advances the estimate may tighten (ratchet
    min with the recomputation); when progress freezes — a stalled stage or
    a band with no fine-grained publishes (spellcheck, llm_cleanup) — the
    value is held only until RUNNING_ETA_REANCHOR_SECONDS of no-advance, or
    until its own promised completion passes, then goes NULL. It only
    returns on a real progress advance. The in-memory map is pruned at the
    two ends of a job's life (claim pops the previous attempt's entry — an
    estimate from across a restart must not cap the new attempt; terminal
    status updates pop), so it cannot grow with the jobs directory.

    Callers hold no lock; every mutation is under _queue_lock (the same
    lock every state read/write takes, so poll and listing agree)."""
    token = state.get("token")
    if now is None:
        now = time.time()
    est = _running_estimate(state, now)
    if token is None:
        return est
    with _queue_lock:
        try:
            prog = float(state.get("progress") or 0.0)
        except (TypeError, ValueError):
            prog = 0.0
        entry = _running_eta_floor.get(token)
        if entry is None:
            lpa = state.get("last_progress_at")
            try:
                advanced_at = float(lpa) if lpa is not None else now
            except (TypeError, ValueError):
                advanced_at = now
            floor, expired = est, False
            if est is not None and (now - advanced_at) > \
                    _running_eta_reanchor_seconds():
                # First observation of a job whose progress has ALREADY
                # been frozen past the allowance (seeded, stalled, or
                # published before this process saw it): the freshly
                # derived value is not yet trustworthy either — null until
                # a real advance re-anchors it.
                floor, expired = None, True
        else:
            advanced = prog > entry["prog"]
            advanced_at = now if advanced else entry["advanced_at"]
            expired = entry["expired"] and not advanced
            if expired:
                floor = None
            elif (now - advanced_at) > _running_eta_reanchor_seconds():
                floor, expired = None, True  # frozen past the allowance
            elif (entry["floor"] is not None
                  and now > entry["floor"][1]):
                floor, expired = None, True  # held value missed its own ETA
            elif est is None:
                floor = None
            elif entry["floor"] is None:
                floor = est
            else:
                floor = (est[0], est[1]) if est[0] <= entry["floor"][0] \
                    else entry["floor"]  # ratchet: never inflate
        if floor is not None:
            _running_eta_floor[token] = {
                "prog": prog, "advanced_at": advanced_at,
                "floor": floor, "at": now, "expired": expired}
        elif expired:
            # Remember WHY it went null: the anchor stays down until
            # progress genuinely advances again (fresh or null, never stale).
            _running_eta_floor[token] = {
                "prog": prog, "advanced_at": advanced_at,
                "floor": None, "at": now, "expired": True}
        else:
            _running_eta_floor.pop(token, None)
        return floor


def _wait_estimates(states: list[dict], sched: dict,
                    now: float | None = None) -> dict[str, tuple[float, float]]:
    """token -> (estimated_wait_seconds, estimated_completion_epoch) for
    every estimable queued job, in the FAIR execution order (AC3/AC4).

    Prefix-sum walk. estimated_wait_seconds answers the question a waiting
    client actually asks — "how long until MY result exists" — so it is
    the currently-processing job's REMAINING estimate + every queued job
    ahead of the position + the job's OWN estimate. Consequences:
      * positions are distinct and non-overlapping by construction (each
        adds its own positive cost); completion times chain exactly:
        c(k+1) = c(k) + own(k+1);
      * there is no shared/blanket wait — the only common term is the
        processing job's remaining estimate, which shrinks with elapsed
        time and vanishes when it leaves the processing state;
      * the whole map is recomputed from live state on every poll, so
        admissions, cancellations and completions are reflected instantly
        (AC5). It is an ESTIMATE of a linear model, not a guarantee.
    A queued job with unknown own duration gets NO entry (honest null,
    never a fabricated duration). PROCESSING_SECONDS_PER_AUDIO_SECOND=0
    disables emission entirely (empty map).
    """
    rate = _processing_rate()
    if rate <= 0:
        return {}
    if now is None:
        now = time.time()
    by_token = {str(s.get("token")): s for s in states}
    processing = next(
        (s for s in states if s.get("status") == "processing"), None)
    prefix = _processing_remaining_seconds(processing, rate, now)
    out: dict[str, tuple[float, float]] = {}
    for token in _fair_queued_order(states, sched):
        st = by_token.get(token)
        if st is None:
            continue
        own = _estimate_seconds_for(st, rate)
        wait = max(0.0, prefix + own)
        if _known_duration_seconds(st) is not None:
            out[token] = (wait, now + wait)
        prefix += own
    return out


def _iso_utc(ts: float) -> str:
    """Wall-clock stamp in the documented YYYY-MM-DDTHH:MM:SSZ form.
    fromtimestamp+strftime is the only spelling that keeps the literal Z
    (astimezone().isoformat() emits '+00:00')."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _select_next_queued(
    states: list[dict], sched: dict, cap: int,
    global_cap: int | None = None,
) -> tuple[dict | None, dict]:
    """Pick the next claimable job: round-robin across sessions, FIFO inside
    a session, sessions at the processing cap skipped. Returns (job, new
    scheduler state); (None, sched) when nothing may be claimed.

    ``global_cap`` (V1.5.1, card t_dbad2235) is the service-wide in-flight
    ceiling: when the TOTAL number of processing jobs has reached it, nothing
    is claimable regardless of sessions — enforced here so the claim path can
    never start a job the service cannot run."""
    inflight: dict[str, int] = {}
    for st in states:
        if st.get("status") == "processing":
            key = _session_key(st)
            inflight[key] = inflight.get(key, 0) + 1
    if global_cap is not None and sum(inflight.values()) >= global_cap:
        return None, sched
    groups = _queued_groups(states)
    eligible = {k: g for k, g in groups.items() if inflight.get(k, 0) < cap}
    if not eligible:
        return None, sched
    key = _rr_pick(eligible, sched["last_served"])
    chosen = eligible[key][0]
    counter = sched["serve_counter"] + 1
    return chosen, {"serve_counter": counter,
                    "last_served": {**sched["last_served"], key: counter}}


def _hash_ident(raw: str) -> str:
    """Domain-separated SHA-256 digest (first 32 hex chars).

    Raw session tokens, cookies and IPs are hashed exactly as in V1.0 so
    no secret or PII ever lands in job state on disk.
    """
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _new_session_id() -> str:
    """Fresh high-entropy session id (uuid4 hex, ~122 bits of entropy)."""
    return uuid.uuid4().hex


def _resolve_session(request: Request) -> tuple[str, str, str]:
    """(identity hash, session id, source) for the V1.5 precedence chain.

    Highest first: explicit SESSION_HEADER header (default "X-Session"),
    server-issued session cookie, client IP, "anonymous". The session id is
    the presented header/cookie value, or a freshly issued one when the
    client had neither (the caller returns it via Set-Cookie + response body,
    and it is hashed under the dedicated `session_ident` job-state key).
    Stateless: the server stores no session table; only hashes are persisted.
    """
    header = _session_header_name()
    value = request.headers.get(header)
    if value and value.strip():
        return _hash_ident(f"header:{header}:{value.strip()}"), value.strip(), "header"
    cookie = request.cookies.get(_session_cookie_name())
    if cookie and cookie.strip():
        return _hash_ident(f"cookie:{cookie.strip()}"), cookie.strip(), "cookie"
    session = _new_session_id()
    if request.client and request.client.host:
        return _hash_ident(f"ip:{request.client.host}"), session, "ip"
    return _hash_ident("anonymous"), session, "anonymous"


def _probe_duration(path: str) -> float:
    """WAV duration used for both chunk-window planning and job metadata."""
    try:
        from pipeline.transcribe import _probe_duration as ffprobe_duration

        return ffprobe_duration(path)
    except Exception:
        return 0.0


def _engine_progress(update_state):
    """Map the engine's pass/chunk counts onto public stage bands.

    Each report (stage name, pass done, pass total) is a real work boundary:
    ASR passes interpolate inside the ``transcribing`` band, diarized
    assembly inside ``diarizing``. No timers — a chunked engine that reports
    per-chunk boundaries refines within the band for free.
    """

    def on_progress(stage: str, done: int, total: int) -> None:
        public = _ENGINE_STAGES.get(
            stage, stage if stage in STAGE_PROGRESS else "transcribing")
        done = min(int(done), max(1, int(total)))
        update_state("processing", _stage_progress(public, done / max(1, total)),
                     f"{public.capitalize()} pass {done}/{total}", stage=public)

    return on_progress


def process_job_background(token: str, audio_path: str, filename: str,
                           llm=None, cleanup_settings: "CleanupSettings | None" = None):
    """Run one job end to end.

    ``llm`` is the opt-in LLM cleanup seam (card t_48c2efae): a caller-
    injected ``prompt -> response`` callable. The HTTP/queue path never
    passes it and reads no env toggle that would hardwire a network call —
    the cleanup pass is off unless an embedding caller injects a client,
    mirroring the ``SELF_ID_LLM``/``extra_evidence`` seam discipline.
    """
    job_dir = JOBS_DIR / token
    state_file = job_dir / "status.json"

    def update_state(st, prog=None, msg="", err="", stage=None, dur=None):
        with _queue_lock:
            if state_file.exists():
                try:
                    st_data: dict = json.loads(state_file.read_text())
                except Exception:
                    st_data = {"token": token}
                # A cancelled/absent job must not be resurrected by a straggler.
                if st_data.get("status") in ("cancelled",):
                    return
                st_data["status"] = st
                prev = float(st_data.get("progress") or 0.0)
                if stage is not None:
                    st_data["stage"] = stage
                if dur:
                    # AC1 refresh (card t_b3ae9366): the decode-time probe of
                    # the real 16k wav replaces the admission-time estimate.
                    st_data["duration_seconds"] = float(dur)
                if st == "failed":
                    # Stage stays where the work broke; progress keeps its
                    # monotonic position but 1.0 is reserved for done, so a
                    # failure never looks success-ish.
                    failing = stage or st_data.get("stage") or "unknown"
                    st_data["stage"] = failing
                    floor = STAGE_PROGRESS.get(failing, prev)
                    target = max(prev, floor, prev if prog is None else prog)
                    st_data["progress"] = round(min(target, 0.9999), 4)
                    if not msg:
                        msg = f"Failed during {failing}: {err}"
                elif prog is not None:
                    # Monotonic non-decreasing across the whole job (card
                    # t_14ac6b49 AC2): a stage publish never rewinds.
                    st_data["progress"] = round(max(prev, prog), 4)
                if st == "processing":
                    # AC6 enabler (card t_d64bdf29): the running-phase ETA
                    # must distinguish a genuinely stalled stage from a
                    # band that publishes no fine-grained progress, so the
                    # time of the last REAL advance is journaled — same
                    # clamped-monotonic semantics as `progress` itself.
                    if float(st_data.get("progress") or 0.0) > prev:
                        st_data["last_progress_at"] = time.time()
                else:
                    # Terminal/failed: the job's running-phase estimate
                    # memory is dead — prune at the source so the ratchet
                    # map cannot grow with the jobs directory.
                    _running_eta_floor.pop(token, None)
                if msg:
                    st_data["message"] = msg
                if err:
                    st_data["error"] = err
                tmp = job_dir / "status.json.tmp"
                tmp.write_text(json.dumps(st_data))
                os.replace(tmp, state_file)
                _journal_append(job_dir, st_data)

    try:
        mock = os.environ.get("MOCK_PIPELINE") == "1"
        update_state("processing", STAGE_PROGRESS["decoding"], "Decoding audio...",
                     stage="decoding")
        if mock:
            # Mock mode never reads the input bytes; skip ffmpeg so synthetic
            # test payloads (and GPU-free CI runs) short-circuit cleanly.
            wav_path = audio_path
            duration = 0.0
        else:
            wav_path = str(job_dir / "audio_16k.wav")
            decode_audio(audio_path, wav_path)
            duration = _probe_duration(wav_path)

        update_state("processing", _stage_progress("transcribing"), "Transcribing speech...",
                     stage="transcribing",
                     dur=duration if not mock else None)
        segments = transcribe_and_diarize(
            wav_path,
            mock=mock,
            duration=duration or None,
            progress=_engine_progress(update_state),
        )

        if not segments:
            update_state("failed", err="Empty transcript result.",
                         msg="Failed during transcribing: empty transcript result.",
                         stage="transcribing")
            return

        # The engine call returning is the diarization completion boundary
        # (its progress callback fired per pass/chunk inside the bands).
        update_state("processing", _stage_progress("diarizing", 1.0),
                     "Diarized assembly complete", stage="diarizing")

        suggestions: dict = {}
        # Card t_72405565: the vocabulary pass is ON by default and shares the
        # module's opt-out semantics (SPELL_CHECK in {0,false,no,off} disables
        # it). The old hard gate on os.environ == "1" meant the shipped
        # service never ran the pass at all.
        if spell_check_enabled() and not mock:
            update_state("processing", STAGE_PROGRESS["spellcheck"],
                         "Spell-checking transcript...", stage="spellcheck")
            segments = spell_check_segments(segments).segments

        cleanup_evidence: list = []
        cleanup_summary = ""
        if llm is not None:
            update_state("processing", STAGE_PROGRESS["llm_cleanup"],
                         "Running opt-in LLM cleanup pass...",
                         stage="llm_cleanup")
            cleanup = run_llm_cleanup(
                segments, llm=llm,
                settings=cleanup_settings or CleanupSettings.from_env(),
            )
            segments = cleanup.segments
            cleanup_evidence = cleanup.name_evidence
            cleanup_summary = (
                f"chunks accepted: {cleanup.accepted} | rejected: {cleanup.rejected} "
                f"(failed: {cleanup.failed}) | permitted line edits: "
                f"{cleanup.changed_lines} | name evidence: "
                f"{len(cleanup.name_evidence)}"
            )
            (job_dir / "llm_cleanup_report.md").write_text(
                cleanup.report_markdown, encoding="utf-8"
            )
            logger.info("llm_cleanup: %s", cleanup_summary)

        update_state("processing", STAGE_PROGRESS["self_identification"],
                     "Suggesting speaker identities...", stage="self_identification")
        suggestions = suggest_speaker_names(segments, extra_evidence=cleanup_evidence)

        update_state("processing", STAGE_PROGRESS["rendering"],
                     "Rendering markdown report...", stage="rendering")
        meta = {
            "filename": filename,
            "source": filename,
            "duration_seconds": duration,
            "suggestions": suggestions,
            "cleanup_summary": cleanup_summary,
        }
        md_text = render_markdown(segments, meta)

        (job_dir / "markdown_report.md").write_text(md_text, encoding="utf-8")

        result_data = {
            "token": token,
            "status": "done",
            "filename": filename,
            "duration_seconds": duration,
            "segments": segments
        }
        (job_dir / "output_result.json").write_text(json.dumps(result_data))
        update_state("done", 1.0, "Transcription complete.", stage="done")

    except Exception as exc:
        logger.exception("Background job failed for token %s", token)
        update_state("failed", err=str(exc))


def _claim_next_queued() -> dict | None:
    """Atomically pick the fair-share next job and mark it processing.

    V1.5 (card t_4b40acfd): selection is round-robin across sessions (FIFO
    within a session) and honours MAX_PROCESSING_PER_SESSION — a session that
    already holds the cap in flight is skipped even when its jobs sit at the
    global FIFO head. The pointer is persisted to disk per claim, so the
    service does not lose its place on restart.
    """
    cap = _max_processing_per_session()
    with _queue_lock:
        # MAX_PROCESSING_GLOBAL is read INSIDE the lock next to the selection
        # so the effective cap cannot change between the in-flight count and
        # the claim. Raising it at runtime is still bounded by the fixed
        # startup worker pool; lowering it takes effect on the next claim.
        global_cap = _max_processing_global()
        job, sched = _select_next_queued(
            _iter_states(), _load_scheduler_state(), cap, global_cap)
        if job is None:
            return None
        _save_scheduler_state(sched)
        job["status"] = "processing"
        job["stage"] = "queued"
        # Clamp, never reset: a job requeued by the lifespan reclaim keeps the
        # dead attempt's progress, and AC2 forbids any observed regression
        # across the whole job (restart included). Fresh jobs still move 0.0 ->
        # 0.05; reclaimed jobs stay at their preserved value and the band
        # publishes catch up via update_state()'s max(prev, prog) clamp.
        job["progress"] = max(float(job.get("progress") or 0.0), 0.05)
        # Fair-scheduled pickup: the admission-order rank is no longer
        # meaningful once the job leaves the queue (AC6, card t_4b40acfd).
        job["queue_position"] = None
        job["message"] = "Claimed by worker"
        job["started_at"] = time.time()
        # Card t_d64bdf29: started_at is the job's FIRST-ever claim (kept
        # stable across reclaims); attempt_started_at marks this attempt,
        # the honest denominator for the running-phase observed rate. A new
        # attempt also drops the dead attempt's estimate memory — an
        # estimate computed from pre-restart work must not cap the new
        # attempt's value (the card's restart-staleness trap).
        job["attempt_started_at"] = job["started_at"]
        _running_eta_floor.pop(job["token"], None)
        _write_state(job)
        logger.info(
            "queue: claimed job token=%s session=%s serve_counter=%d cap=%d "
            "global_cap=%d",
            job.get("token"), _session_key(job), sched["serve_counter"], cap,
            global_cap,
        )
        return job


class _WakeHub:
    """Fan-out wake for N queue workers (V1.5.1, card t_dbad2235).

    The single shared `asyncio.Event` was safe with exactly ONE waiter: the
    worker that found nothing called `clear()` immediately before waiting, so
    a set() could never be consumed by anyone else. With N workers that
    breaks — one worker's `clear()` can erase the notification another worker
    has not reached its `wait()` on yet, stranding it until the poll timeout.
    The hub hands every worker its own event; `set()` fans out to all of them
    and each worker clears only its own, so no worker can eat another's
    wake-up. A set() landing while a worker is mid-job stays pending: the
    worker's next wait returns instantly, clears, and re-claims.
    """

    def __init__(self) -> None:
        self._events: set[asyncio.Event] = set()

    def register(self) -> asyncio.Event:
        ev = asyncio.Event()
        self._events.add(ev)
        return ev

    def unregister(self, ev: asyncio.Event) -> None:
        self._events.discard(ev)

    def set(self) -> None:
        # Called from POST admission (same loop); wakes every waiting worker.
        for ev in self._events:
            ev.set()

    async def wait(self, ev: asyncio.Event, timeout: float) -> None:
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            ev.clear()


async def _queue_worker_loop(worker_index: int = 0) -> None:
    """One fair-share queue worker: claim, process, repeat.

    N of these run concurrently behind the claim-time MAX_PROCESSING_GLOBAL
    cap (V1.5.1, card t_dbad2235); each worker still processes ONE job at a
    time and never claims while busy, so total concurrency is bounded by the
    pool size AND by the cap checked inside the atomic claim. The pipeline is
    CPU/GPU-bound synchronous code, so it runs in the dedicated per-job
    executor (sized to the worker count at boot — falls back to the loop
    default when no pool was created) and the event loop stays responsive.
    A job that raises is logged and skipped, never fatal to this worker.
    """
    loop = asyncio.get_running_loop()
    executor = getattr(app.state, "job_executor", None)
    wake = getattr(app.state, "wake", None)
    hub = wake if isinstance(wake, _WakeHub) else None
    gate = hub.register() if hub is not None else None
    try:
        while True:
            job = _claim_next_queued()
            if job is not None:
                try:
                    await loop.run_in_executor(
                        executor,
                        process_job_background,
                        job["token"],
                        job.get("audio_path", str(JOBS_DIR / job["token"] / "audio_source.wav")),
                        job.get("filename", "unknown"),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "worker %d crashed processing %s", worker_index,
                        job["token"])
                continue
            poll = _env_float("QUEUE_POLL_SECONDS", DEFAULT_QUEUE_POLL_SECONDS)
            if hub is not None and gate is not None:
                # No job claimable: wait for the next admission fan-out, with
                # the short poll as the safety net that makes any lost notify
                # harmless (it cannot happen through the hub, by design).
                await hub.wait(gate, poll)
            elif isinstance(wake, asyncio.Event):
                # Legacy single-event path (app.state.wake set outside the
                # lifespan); safe only with one waiter, kept for compatibility.
                wake.clear()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=poll)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(poll)
    finally:
        if hub is not None and gate is not None:
            hub.unregister(gate)


def validate_config() -> str:
    """Boot-time config validation + report (V1.5 AC8, card t_920fb7dc).

    Surfaces every session/queue/retention knob the service reads —
    SESSION_HEADER and SESSION_COOKIE_NAME (HTTP-token validated, invalid
    values warn and fall back to the documented defaults), MAX_QUEUE_DEPTH
    and RETENTION_DAYS — so misconfiguration is visible in `docker logs`
    instead of silently degrading behavior. Invalid integers keep the
    established _env_int fallback; a non-positive depth is called out and
    pinned to the default. MAX_JOBS_PER_SESSION is accepted but IGNORED
    (deprecated with the per-session rejection path, card t_82bfbc15): if
    it is set, startup logs a warning saying so — no job is ever rejected
    on its basis.
    """
    header = _session_header_name()
    cookie = _session_cookie_name()
    depth = _env_int("MAX_QUEUE_DEPTH", DEFAULT_MAX_QUEUE_DEPTH)
    days = _retention_days()
    if "MAX_JOBS_PER_SESSION" in os.environ:
        logger.warning(
            "config: MAX_JOBS_PER_SESSION=%r is DEPRECATED and IGNORED — "
            "sessions may hold multiple active jobs; the only admission "
            "rejection is queue depth (503). Remove it from your env.",
            os.environ["MAX_JOBS_PER_SESSION"],
        )
    if depth < 1:
        logger.warning(
            "config: MAX_QUEUE_DEPTH=%d would reject every submission; using %d",
            depth, DEFAULT_MAX_QUEUE_DEPTH,
        )
        depth = DEFAULT_MAX_QUEUE_DEPTH
    # ETA model constant (card t_b3ae9366): invalid values warn here via
    # _processing_rate() and fall back to the default; exact 0 is valid and
    # switches ETA emission off. The value is hardware-specific — the
    # default is the measured T400 rate, re-derive it for other GPUs.
    rate = _processing_rate()
    summary = (
        f"config: session header={header} cookie={cookie} "
        f"max_queue_depth={depth} max_processing_per_session="
        f"{_max_processing_per_session()} "
        f"max_processing_global={_max_processing_global()} "
        f"retention_days={days} "
        f"eta_rate={rate} eta={'on' if rate > 0 else 'OFF'}"
    )
    logger.info(summary)
    return summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    # Boot revalidation: jobs left 'processing' by a crash/restart return to
    # the queue with their seq intact, preserving FIFO order across restarts;
    # admission limits are revalidated by every POST from this persisted state.
    reclaimed = 0
    with _queue_lock:
        for st in _iter_states():
            if st.get("status") == "processing":
                st["status"] = "queued"
                st["stage"] = "queued"
                st["message"] = "Requeued after service restart"
                _write_state(st)
                reclaimed += 1
    if reclaimed:
        logger.info("requeued %d orphaned processing job(s) on boot", reclaimed)

    app.state.wake = _WakeHub()
    worker_enabled = os.environ.get("QUEUE_WORKER_ENABLED", "1") == "1"
    worker_tasks: list[asyncio.Task] = []
    app.state.job_executor = None
    if worker_enabled:
        # N concurrent workers behind the claim-time MAX_PROCESSING_GLOBAL cap
        # (V1.5.1, card t_dbad2235; the pool size is read once at boot — the
        # cap itself is re-read at every claim). The per-job executor is sized
        # to the worker count on purpose: asyncio's default ThreadPoolExecutor
        # is min(32, cpu_count + 4) wide, so on a small host (the T400 has 4
        # cores -> default width 8) a larger cap could otherwise be silently
        # under-delivered by thread starvation.
        n_workers = _max_processing_global()
        app.state.job_executor = ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="asr-job",
        )
        worker_tasks = [
            asyncio.create_task(_queue_worker_loop(i),
                                name=f"asr-queue-worker-{i}")
            for i in range(n_workers)
        ]
        logger.info(
            "concurrency: MAX_PROCESSING_GLOBAL=%d — %d worker(s) spawned, "
            "per-job executor width %d",
            n_workers, len(worker_tasks), n_workers,
        )

    # Retention + disk-pressure wiring (card t_5e6200fc): the boot pass runs
    # synchronously here — deliberately AFTER the boot requeue above, so
    # reclaimed jobs are 'queued' and exempt from the age sweep — then the
    # hourly age sweep and the DISK_CHECK_INTERVAL_SECONDS disk check take
    # over as background tasks (AC5). One INFO line records the whole policy.
    run_retention_boot_pass()
    retention_task = asyncio.create_task(_retention_loop(), name="retention-sweep")
    disk_task = asyncio.create_task(_disk_pressure_loop(), name="disk-pressure-check")
    logger.info(
        "retention: days=%d (0 disables age sweep), sweep interval=%ds, "
        "disk watermark=%d MiB, disk check=%ds",
        _retention_days(),
        RETENTION_SWEEP_INTERVAL_SECONDS,
        _disk_min_free_mib(),
        _disk_check_interval_seconds(),
    )
    try:
        yield
    finally:
        for worker_task in worker_tasks:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        if app.state.job_executor is not None:
            # Do not block shutdown on a straggler thread; the queue state on
            # disk is already 'processing' and the boot requeue reclaims it.
            app.state.job_executor.shutdown(wait=False)
        for loop_task in (retention_task, disk_task):
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task


app = FastAPI(title="AutoDiarist", version="1.0.0", lifespan=lifespan)


class JobResponse(BaseModel):
    token: str
    status: str
    session: str = ""
    # 1-based rank at admission in the FAIR-SCHEDULED execution order
    # (V1.5, cards t_82bfbc15 + t_4b40acfd): the client learns its queue
    # position from the 202 itself, without a first poll.
    queue_position: int | None = None
    # Queue-aware wait ESTIMATE at admission (V1.5, card t_b3ae9366):
    # seconds from now until the result should exist, and the matching
    # UTC wall-clock time. ESTIMATE only — linear model, never a
    # guarantee. None when the duration probe failed or the rate
    # constant disables ETA emission (see README / CONFIG.md).
    estimated_wait_seconds: float | None = None
    estimated_completion_at: str | None = None


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    token: str
    status: str


@app.post("/v1/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request, response: Response, file: UploadFile = File(...)):
    token = str(uuid.uuid4())
    job_dir = JOBS_DIR / token

    # 1. Extension validation BEFORE touching disk.
    name = file.filename or "upload"
    ext = name.split(".")[-1].lower()
    if ext not in ("wav", "mp3", "m4a", "aac", "ogg", "flac"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Supported: wav, mp3, m4a, aac, ogg, flac")

    # 2. Create job dir and stream upload.
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / "audio_source.wav"  # keeping original extension name
    
    size = 0
    max_size = int(os.environ.get("MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES))
    try:
        with open(audio_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {max_size} bytes")
                buffer.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    # 2b. Duration probe at admission (V1.5, card t_b3ae9366): a cheap
    #     ffprobe container-header read — NO full decode — off the event
    #     loop, so the very first response can carry an honest wait
    #     estimate. A failed/unparseable probe stores duration_seconds
    #     None (unknown): the job then shows NO ETA, and polls never
    #     re-probe (the value is refreshed exactly once by the decode-time
    #     probe when the job actually runs).
    probed = await run_in_threadpool(_probe_duration, str(audio_path))
    duration_seconds = round(float(probed), 3) if probed and probed > 0 else None

    # 3. Admission control under the queue lock: queue depth is the single
    #    system-level backpressure gate (AC1, card t_daebf369). Since V1.5
    #    (card t_82bfbc15) the per-session quota rejection is gone: every
    #    submission that fits under the depth is accepted with 202 and
    #    queued behind the others. Rejected uploads leave no trace on disk.
    session, session_id, session_source = _resolve_session(request)
    depth_limit = _env_int("MAX_QUEUE_DEPTH", DEFAULT_MAX_QUEUE_DEPTH)
    retry_after = str(_env_int("RETRY_AFTER_SECONDS", DEFAULT_RETRY_AFTER_SECONDS))
    with _queue_lock:
        states = _iter_states()
        queued = [s for s in states if s.get("status") == "queued"]
        if len(queued) >= depth_limit:
            shutil.rmtree(job_dir, ignore_errors=True)
            # Detail names both the queue depth and the Retry-After value
            # (card t_82bfbc15 scope 5) so a client can back off from the
            # response body alone.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Queue is full ({len(queued)}/{depth_limit} jobs queued). "
                    f"Retry after {retry_after}s."
                ),
                headers={"Retry-After": retry_after},
            )
        seq = max((s.get("seq", 0) for s in states), default=0) + 1
        admitted_at = time.time()
        state = {
            "token": token,
            "status": "queued",
            "session": session,
            # Dedicated session key (V1.5): hash of the session id the client
            # presents (or that was just issued to it) so the scheduler and
            # job listing can group per person. Raw id never hits disk.
            "session_ident": _hash_ident(f"session:{session_id}"),
            "session_source": session_source,
            # created_at (card t_f97d12a8): stable job-creation timestamp for
            # the session listing's newest-first order. queued_at stays the
            # queue-facing clock (boot requeue stamps it on restart); a fresh
            # admission sets both. Legacy jobs lacking created_at fall back
            # to queued_at in the listing.
            "created_at": admitted_at,
            "queued_at": admitted_at,
            "seq": seq,
            "progress": 0.0,
            "message": "Queued",
            "stage": "queued",
            "filename": name,
            "audio_path": str(audio_path),
            # AC1: duration probed at admission (None = unknown). The ETA
            # model reads ONLY this value from state — never the audio.
            "duration_seconds": duration_seconds,
        }
        _write_state(state)
        # Fair-schedule rank at admission (1-based, card t_4b40acfd): the
        # position in the round-robin execution order the worker will
        # actually follow — NOT raw arrival order. A queued job that
        # arrived first can therefore report a position behind a later
        # arrival from a different session (semantic change from FIFO;
        # within one session the order stays FIFO by seq). Same rank GET
        # /v1/jobs/{token} recomputes for a queued job.
        sched = _load_scheduler_state()
        simulated = states + [state]  # processing job included: it is the
        # prefix the queue-aware ETA must account for (card t_b3ae9366).
        queue_position = _fair_queued_order(simulated, sched).index(token) + 1
        eta = _wait_estimates(simulated, sched).get(token)

    # Wake the worker if it's idle-waiting.
    wake = getattr(app.state, "wake", None)
    if wake is not None:
        wake.set()

    # Session identity issuance (V1.5): the cookie lets browsers stay in one
    # bucket without any client setup; the body field lets raw-HTTP clients
    # reuse the id without a cookie jar. Stateless server: the value comes
    # from the client (header/cookie) or is freshly minted here.
    response.set_cookie(
        key=_session_cookie_name(),
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return JobResponse(
        token=token,
        status="queued",
        session=session_id,
        queue_position=queue_position,
        # Queue-aware ETA at admission (card t_b3ae9366) — honest estimate,
        # recomputed by the scheduler order simulation, never a guarantee.
        estimated_wait_seconds=round(eta[0], 1) if eta else None,
        estimated_completion_at=_iso_utc(eta[1]) if eta else None,
    )


def _listing_created_at(state: dict) -> float:
    """Ordering clock for the session listing: created_at, falling back to
    queued_at for jobs admitted before the field existed (t_f97d12a8)."""
    return state.get("created_at") or state.get("queued_at") or 0.0


def _session_job_entries(
    states: list[dict], session_ident: str, limit: int, queued_order: list[str],
    estimates: dict[str, tuple[float, float]] | None = None,
    running_estimates: dict[str, tuple[float, float] | None] | None = None,
) -> tuple[list[dict], int]:
    """Pure session view (card t_f97d12a8): the caller's jobs only,
    newest-first (created_at desc, seq desc tiebreak), truncated to `limit`.

    Every entry carries token/status/progress/stage/source/created_at;
    queued entries additionally carry queue_position (the fair-scheduled
    execution rank from `queued_order`, the same simulation GET /v1/jobs/
    {token} and POST 202 use) and, while ETA emission is enabled, the
    queue-aware wait estimate estimated_wait_seconds /
    estimated_completion_at — the SAME live values the poll endpoint
    reports (card t_b3ae9366), null when no honest estimate exists for
    that job; failed entries carry the error string.
    Cancelled jobs stay in the listing — they are terminal states users
    want to see. Returns (entries, total_before_truncation)."""
    mine = [s for s in states if s.get("session_ident") == session_ident]
    mine.sort(key=lambda s: (_listing_created_at(s), s.get("seq") or 0), reverse=True)
    rank = {token: i + 1 for i, token in enumerate(queued_order)}
    entries = []
    for s in mine[:limit]:
        entry = {
            "token": s.get("token"),
            "status": s.get("status"),
            "progress": s.get("progress", 0.0),
            # AC6 (card t_d64bdf29): integer percent view, same value the
            # poll reports, so one field answers every listing renderer.
            "progress_percent": _progress_percent(s),
            "stage": s.get("stage") or s.get("status"),
            "source": s.get("filename", "unknown"),
            "created_at": _listing_created_at(s),
        }
        if s.get("status") == "queued":
            entry["queue_position"] = rank.get(s["token"])
            if estimates is not None:
                # Computed from the same live queue snapshot as the rank —
                # never persisted, never re-probed (AC5).
                eta = estimates.get(s["token"])
                entry["estimated_wait_seconds"] = round(eta[0], 1) if eta else None
                entry["estimated_completion_at"] = _iso_utc(eta[1]) if eta else None
        if s.get("status") == "processing" and running_estimates is not None:
            # Running-phase ETA (card t_d64bdf29): the SAME live ratchet
            # value the poll endpoint reports for this token, aliased onto
            # estimated_wait_seconds so one field answers "when will this
            # finish" for queued AND processing entries. Null until an
            # honest estimate exists.
            est = running_estimates.get(s["token"])
            rem = round(est[0], 1) if est else None
            entry["estimated_remaining_seconds"] = rem
            entry["estimated_wait_seconds"] = rem
            entry["estimated_completion_at"] = _iso_utc(est[1]) if est else None
        if s.get("status") == "failed":
            entry["error"] = s.get("error") or s.get("message") or ""
        entries.append(entry)
    return entries, len(mine)


@app.get("/v1/jobs")
async def list_jobs(request: Request, response: Response, limit: int = DEFAULT_LIST_LIMIT):
    """Session-scoped job listing (V1.5, card t_f97d12a8).

    GET /v1/jobs/{token} answers "what is THIS job doing" for anyone holding
    the token; this endpoint answers "what did I submit" — it is filtered by
    the same hashed session identity admission stamps on every job, so a
    client can find its token, watch status/progress, and cancel a queued
    job without the server ever storing a plaintext session id. Other
    sessions' jobs cannot appear (identity hash equality is the only filter)
    and there is deliberately no admin/cross-session mode.

    Order: newest-first (created_at desc, seq desc tiebreak). ?limit= takes
    precedence over the default page size, clamped into [1, MAX_LIST_LIMIT]
    — out-of-range values clamp, they never 4xx. Non-integer limit is the
    only rejected input (FastAPI 422).
    """
    _, session_id, _ = _resolve_session(request)
    page = max(1, min(limit, MAX_LIST_LIMIT))
    with _queue_lock:
        states = _iter_states()
        # Service-wide in-flight count (V1.5.1, card t_dbad2235): a test
        # harness or operator watching real concurrency reads it here instead
        # of inferring it from poll deltas. Computed from the SAME snapshot the
        # claim path uses, under the same lock.
        processing_count = sum(1 for st in states
                               if st.get("status") == "processing")
        sched = _load_scheduler_state()
        queued_order = _fair_queued_order(states, sched)
        # One live queue snapshot drives rank AND ETA (card t_b3ae9366):
        # both surfaces agree because both come from this pass. Disabled
        # emission (rate=0) omits the fields entirely, like the poll.
        estimates = _wait_estimates(states, sched) if _processing_rate() > 0 else None
    # Running-phase ETA for this snapshot's processing entries (card
    # t_d64bdf29): the SAME ratchet the poll endpoint drives, so both
    # surfaces agree. Called OUTSIDE the queue lock — the ratchet takes
    # it itself. None when the feature is disabled, like `estimates`.
    running = None
    if _processing_rate() > 0:
        running = {st["token"]: _running_eta_ratchet(st) for st in states
                   if st.get("status") == "processing" and st.get("token")}
    entries, total = _session_job_entries(
        states, _hash_ident(f"session:{session_id}"), page, queued_order,
        estimates, running,
    )
    # Echo the identity exactly like POST does (card t_920fb7dc): a client
    # that only had an IP-derived identity gets the minted id back via
    # cookie (+ default header when the client doesn't already manage one)
    # so its next listing matches its submissions.
    response.set_cookie(
        key=_session_cookie_name(),
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    if not request.headers.get(_session_header_name(), "").strip():
        response.headers[_session_header_name()] = session_id
    return {"jobs": entries, "count": len(entries), "total": total,
            "limit": page, "processing_count": processing_count}


@app.get("/v1/jobs/{token}")
async def get_job_status(token: str):
    validate_token(token)
    state = _read_state(token)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if "stage" not in state:
        # Pre-stage job dirs (persisted before this API version): mirror the
        # coarse status so the field is never absent from a response.
        state["stage"] = state.get("status", "unknown")
    if state.get("status") == "queued":
        # queue_position is the 1-based rank in the FAIR-SCHEDULED execution
        # order the worker actually follows (card t_4b40acfd): per-session
        # FIFO sub-queues served round-robin — NOT raw admission order, so a
        # job can report a rank behind a later arrival from a different
        # session. Recomputed from the current queue + persisted pointer, so
        # it only moves downward as work ahead completes or leaves the queue
        # (a cancelled job drops out of the simulation; positions behind it
        # compress). Assumes no new arrivals — same caveat FIFO had.
        # estimated_wait_seconds / estimated_completion_at (card t_b3ae9366)
        # are computed in that SAME fair order on EVERY poll — never read
        # back from disk, never re-probed — so the estimate tracks queue
        # movement (admissions/cancellations/completions) instantly. They
        # are an ESTIMATE from a linear duration model, not a guarantee, and
        # are None for a job whose own duration probe failed at admission.
        with _queue_lock:
            states = _iter_states()
            sched = _load_scheduler_state()
            order = _fair_queued_order(states, sched)
            estimates = _wait_estimates(states, sched)
        state["queue_position"] = order.index(token) + 1 if token in order else None
        if _processing_rate() > 0:
            eta = estimates.get(token)
            state["estimated_wait_seconds"] = round(eta[0], 1) if eta else None
            state["estimated_completion_at"] = _iso_utc(eta[1]) if eta else None
    elif state.get("status") == "processing":
        # Running-phase ETA (card t_d64bdf29): an honest completion
        # estimate derived from the job's OWN observed rate — elapsed time
        # since this attempt started against the published monotonic
        # progress. estimated_wait_seconds aliases the remaining seconds
        # (a running client is asking the same question a queued client
        # asks), estimated_completion_at is this job's own finish time.
        # Null until an honest estimate exists (fresh/warm-up job, stalled
        # past the freshness allowance, no start stamp) — never stale,
        # never inflated. Rate=0 disables the whole feature like the
        # queue side (t_b3ae9366).
        if _processing_rate() > 0:
            est = _running_eta_ratchet(state)
            state["estimated_remaining_seconds"] = (
                round(est[0], 1) if est else None)
            state["estimated_completion_at"] = _iso_utc(est[1]) if est else None
            state["estimated_wait_seconds"] = (
                round(est[0], 1) if est else None)
        else:
            state["estimated_wait_seconds"] = None
            state["estimated_completion_at"] = None
            state["estimated_remaining_seconds"] = None
    # AC6 (card t_d64bdf29): integer percent view of the banded float on
    # every status payload — queued, running, and terminal alike.
    state["progress_percent"] = _progress_percent(state)
    return JobStatus(**state)


@app.delete("/v1/jobs/{token}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(token: str):
    """Cancel a queued job (spec minimum for card t_daebf369).

    Running jobs get 409: GPU concurrency is 1 and the pipeline is mid-card,
    so graceful mid-chunk cancel is out of scope here. Cancelled jobs free
    both queue depth and session quota; the input audio is cleaned from disk
    immediately while status.json is kept so the 'cancelled' state remains
    observable. The V2 retention sweep (card t_5e6200fc) expires it — like
    every finished status — after RETENTION_DAYS.
    """
    validate_token(token)
    with _queue_lock:
        state = _read_state(token)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found")
        current = state.get("status")
        if current == "queued":
            state["status"] = "cancelled"
            state["cancelled_at"] = time.time()
            state["message"] = "Cancelled by client while queued"
            _write_state(state)
            (JOBS_DIR / token / "audio_source.wav").unlink(missing_ok=True)
            return {"token": token, "status": "cancelled"}
        if current == "cancelled":
            return {"token": token, "status": "cancelled"}
        if current == "processing":
            raise HTTPException(
                status_code=409,
                detail="Job is currently processing on the GPU (concurrency=1); cancellation of running jobs is not supported yet.",
            )
        raise HTTPException(
            status_code=409,
            detail=f"Job already finished with status '{current}'.",
        )


@app.get("/v1/jobs/{token}/download")
async def download_result(token: str, format: str = "json"):
    validate_token(token)
    job_dir = JOBS_DIR / token
    status_file = job_dir / "status.json"

    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    state = json.loads(status_file.read_text())
    if state.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Cannot download results when status is {state.get('status')}")

    if format == "json":
        res_file = job_dir / "output_result.json"
        if not res_file.exists():
            raise HTTPException(status_code=404, detail="Result file missing")
        return FileResponse(res_file, media_type="application/json", filename=f"{token}_result.json")
    
    elif format == "md":
        md_file = job_dir / "markdown_report.md"
        if not md_file.exists():
            raise HTTPException(status_code=404, detail="Markdown report missing")
        return FileResponse(md_file, media_type="text/markdown", filename=f"{token}_report.md")

    else:
        raise HTTPException(status_code=422, detail="Unknown format requested. Use format=json or format=md")
