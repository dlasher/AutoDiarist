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


def _write_state(state: dict) -> None:
    """Atomic state write; callers hold _queue_lock on transition paths."""
    token = state["token"]
    job_dir = JOBS_DIR / token
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, job_dir / "status.json")


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


def _chunk_progress(update_state):
    """Surface per-chunk pipeline progress in the 0.4..0.9 stage band.

    Values are strictly monotonic: every accepted transition must exceed the
    last, so status.json observers can count real progress.
    """
    last = {"value": 0.4}

    def on_progress(stage: str, done: int, total: int) -> None:
        fraction = min(1.0, max(0.0, done / max(1, total)))
        value = round(0.4 + 0.5 * fraction, 4)
        if value > last["value"]:
            last["value"] = value
            update_state("processing", value, f"{stage} pass {done}/{total}")

    return on_progress


def process_job_background(token: str, audio_path: str, filename: str):
    job_dir = JOBS_DIR / token
    state_file = job_dir / "status.json"

    def update_state(st, prog, msg="", err=""):
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
                st_data["progress"] = prog
                if msg:
                    st_data["message"] = msg
                if err:
                    st_data["error"] = err
                tmp = job_dir / "status.json.tmp"
                tmp.write_text(json.dumps(st_data))
                os.replace(tmp, state_file)

    try:
        mock = os.environ.get("MOCK_PIPELINE") == "1"
        update_state("processing", 0.1, "Decoding audio...")
        if mock:
            # Mock mode never reads the input bytes; skip ffmpeg so synthetic
            # test payloads (and GPU-free CI runs) short-circuit cleanly.
            wav_path = audio_path
            duration = 0.0
        else:
            wav_path = str(job_dir / "audio_16k.wav")
            decode_audio(audio_path, wav_path)
            duration = _probe_duration(wav_path)

        update_state("processing", 0.4, "Running transcription & diarization...")
        segments = transcribe_and_diarize(
            wav_path,
            mock=mock,
            duration=duration or None,
            progress=_chunk_progress(update_state),
        )

        if not segments:
            update_state("failed", 1.0, err="Empty transcript result.")
            return

        suggestions: dict = {}
        if os.environ.get("SPELL_CHECK") == "1" and not mock:
            update_state("processing", 0.92, "Spell-checking transcript...")
            segments = spell_check_segments(segments).segments

        update_state("processing", 0.95, "Suggesting speaker identities...")
        suggestions = suggest_speaker_names(segments)

        update_state("processing", 0.98, "Rendering markdown report...")
        meta = {
            "filename": filename,
            "source": filename,
            "duration_seconds": duration,
            "suggestions": suggestions,
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
        update_state("done", 1.0, "Transcription complete.")

    except Exception as exc:
        logger.exception("Background job failed for token %s", token)
        update_state("failed", 1.0, err=str(exc))


def _claim_next_queued() -> dict | None:
    """Atomically pick the FIFO head and mark it processing (concurrency=1)."""
    with _queue_lock:
        queued = _queued_states()
        if not queued:
            return None
        job = queued[0]
        job["status"] = "processing"
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
