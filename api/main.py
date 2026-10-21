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
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from pipeline.transcribe import decode_audio, transcribe_and_diarize
from pipeline.spellcheck import spell_check_segments
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
DEFAULT_MAX_JOBS_PER_SESSION = 2
DEFAULT_RETRY_AFTER_SECONDS = 30
DEFAULT_QUEUE_POLL_SECONDS = 0.2
# Which header names the session identity; default Authorization (bearer token).
SESSION_HEADER_DEFAULT = "Authorization"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


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


def _session_id(request: Request) -> str:
    """Bearer/session header if present, else client IP, else anonymous.

    Hashed so secrets/PII (tokens, IPs) never land in job state on disk.
    """
    header = os.environ.get("SESSION_HEADER", SESSION_HEADER_DEFAULT)
    value = request.headers.get(header)
    if value and value.strip():
        raw = f"header:{header}:{value.strip()}"
    elif request.client and request.client.host:
        raw = f"ip:{request.client.host}"
    else:
        raw = "anonymous"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


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

    def update_state(st, prog=None, msg="", err="", stage=None):
        with _queue_lock:
            if state_file.exists():
                try:
                    st_data = json.loads(state_file.read_text())
                except Exception:
                    st_data = {"token": token}
                # A cancelled/absent job must not be resurrected by a straggler.
                if st_data.get("status") in ("cancelled",):
                    return
                st_data["status"] = st
                prev = float(st_data.get("progress") or 0.0)
                if stage is not None:
                    st_data["stage"] = stage
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
                     stage="transcribing")
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
        if os.environ.get("SPELL_CHECK") == "1" and not mock:
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
    """Atomically pick the FIFO head and mark it processing (concurrency=1)."""
    with _queue_lock:
        queued = _queued_states()
        if not queued:
            return None
        job = queued[0]
        job["status"] = "processing"
        job["stage"] = "queued"
        job["progress"] = 0.05
        job["message"] = "Claimed by worker"
        job["started_at"] = time.time()
        _write_state(job)
        return job


async def _queue_worker_loop() -> None:
    """Single-GPU worker: one job at a time, FIFO by seq.

    The pipeline is CPU/GPU-bound synchronous code, so it runs in the default
    executor and the event loop stays responsive for admissions and polls.
    """
    loop = asyncio.get_running_loop()
    while True:
        job = _claim_next_queued()
        if job is not None:
            try:
                await loop.run_in_executor(
                    None,
                    process_job_background,
                    job["token"],
                    job.get("audio_path", str(JOBS_DIR / job["token"] / "audio_source.wav")),
                    job.get("filename", "unknown"),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker crashed processing %s", job["token"])
            continue
        wake = getattr(app.state, "wake", None)
        if wake is not None:
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), timeout=_env_float("QUEUE_POLL_SECONDS", DEFAULT_QUEUE_POLL_SECONDS))
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(_env_float("QUEUE_POLL_SECONDS", DEFAULT_QUEUE_POLL_SECONDS))


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    app.state.wake = asyncio.Event()
    worker_enabled = os.environ.get("QUEUE_WORKER_ENABLED", "1") == "1"
    worker_task = None
    if worker_enabled:
        worker_task = asyncio.create_task(_queue_worker_loop(), name="asr-queue-worker")
    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task


app = FastAPI(title="AutoDiarist", version="1.0.0", lifespan=lifespan)


class JobResponse(BaseModel):
    token: str
    status: str


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    token: str
    status: str


@app.post("/v1/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request, file: UploadFile = File(...)):
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

    # 3. Admission control under the queue lock: depth first (system-level
    #    backpressure, AC1), then the per-session quota (AC2). Rejected
    #    uploads leave no enqueued trace on disk.
    session = _session_id(request)
    depth_limit = _env_int("MAX_QUEUE_DEPTH", DEFAULT_MAX_QUEUE_DEPTH)
    session_limit = _env_int("MAX_JOBS_PER_SESSION", DEFAULT_MAX_JOBS_PER_SESSION)
    retry_after = str(_env_int("RETRY_AFTER_SECONDS", DEFAULT_RETRY_AFTER_SECONDS))
    with _queue_lock:
        states = _iter_states()
        queued = [s for s in states if s.get("status") == "queued"]
        if len(queued) >= depth_limit:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503,
                detail=f"Queue is full ({len(queued)}/{depth_limit} jobs queued). Retry later.",
                headers={"Retry-After": retry_after},
            )
        active = [
            s for s in states
            if s.get("status") in ("queued", "processing") and s.get("session") == session
        ]
        if len(active) >= session_limit:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Session limit reached: {len(active)}/{session_limit} active jobs "
                    "per session (queued+processing). Wait for a job to finish or cancel one."
                ),
            )
        seq = max((s.get("seq", 0) for s in states), default=0) + 1
        state = {
            "token": token,
            "status": "queued",
            "session": session,
            "queued_at": time.time(),
            "seq": seq,
            "progress": 0.0,
            "message": "Queued",
            "stage": "queued",
            "filename": name,
            "audio_path": str(audio_path),
        }
        _write_state(state)

    # Wake the worker if it's idle-waiting.
    wake = getattr(app.state, "wake", None)
    if wake is not None:
        wake.set()

    return JobResponse(token=token, status="queued")


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
        # queue_position is the 1-based FIFO rank among currently queued jobs:
        # exactly the worker's pickup order (both keyed on `seq`).
        with _queue_lock:
            rank = 1 + sum(
                1 for s in _queued_states() if s.get("seq", 0) < state.get("seq", 0)
            )
        state["queue_position"] = rank
    return JobStatus(**state)


@app.delete("/v1/jobs/{token}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(token: str):
    """Cancel a queued job (spec minimum for card t_daebf369).

    Running jobs get 409: GPU concurrency is 1 and the pipeline is mid-card,
    so graceful mid-chunk cancel is out of scope here. Cancelled jobs free
    both queue depth and session quota; the input audio is cleaned from disk
    immediately while status.json is kept so the 'cancelled' state remains
    observable (status retention sweep is a separate concern).
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
