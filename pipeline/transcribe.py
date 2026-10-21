"""Audio decoding plus CrispASR-backed ASR + diarization.

Backend per spike verdict (card t_7e17589e, GO): the ``crispasr`` CUDA CLI
(v0.8.43, Nemotron 3.5 ASR streaming 0.6B, Q4_K GGUF) is invoked as a
subprocess; foxnose provides the default globally-stable diarization pass,
``DIAR_METHOD=sortformer`` selectable later. The NeMo/torch in-process path
is retired — no ``nemo`` or ``torch`` import remains on the real path.

VRAM discipline inherited from card t_95726260:

* bounded per-pass memory — nemotron auto-slices audio into ~30 s windows
  inside the engine (measured 1085 MiB peak over a full 39-min run, well
  under the 3.5 GiB cap), so the adapter runs one whole-file invocation
  instead of local ``CHUNK_SECONDS`` windowing (card AC: upstream chunking
  is equivalent, existing chunking tests stay on ``pipeline.chunking``).
* per-stage VRAM instrumentation — an ``nvidia-smi`` sampler thread records
  GPU memory allocated during the invocation; the run logs
  ``allocated=/peak=`` lines (INFO) and ``TranscriptionOOM`` is raised when
  the peak exceeds ``VRAM_LIMIT_MIB`` (default 3584 = 3.5 GiB).
* per-chunk progress streaming (card t_1eb108e0) — the engine prints one
  timestamped chunk line per completed ~30 s window; the adapter consumes
  live stdout line-by-line and publishes ``("asr", done, total)`` reports on
  those real boundaries, so the ``transcribing`` band refines during the run
  without timers (mapping lives in ``api/main.py::_engine_progress``).
* the silent CPU fallback trap (missing libcudart/libcublas drops crispasr
  to CPU ~11x slower without error): the adapter pins ``--gpu-backend`` and
  probes ``crispasr --version`` for CUDA backend registration before
  spending minutes on a job, failing fast with a typed error.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ASR_BACKEND = "nemotron"
# SPEC 08 fixes en-US; the crispasr CLI accepts the region qualifier.
ASR_TARGET_LANG = "en-US"
DIAR_METHODS = {"foxnose", "ecapa", "sherpa", "sortformer", "none"}
DIAR_METHOD_DEFAULT = "foxnose"
GPU_BACKENDS = {"cpu", "cuda", "metal", "vulkan"}
GPU_BACKEND_DEFAULT = "cuda"
# nemotron Q4_K GGUF; see spike A3 — canonical repo is
# cstr/nemotron-3.5-asr-streaming-0.6b-GGUF (engine auto-download target).
Q4K_GGUF_NAME = "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
UNKNOWN_SPEAKER = "Unknown"
DEFAULT_VRAM_LIMIT_MIB = 3584.0  # 3.5 GiB cap from card t_95726260
DEFAULT_MIN_FRAGMENT_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 1800.0
_LANG_TAG = re.compile(r"\s*<\s*[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})?\s*>")
_SPEAKER_INDEX = re.compile(r"speaker\s*(\d+)", re.IGNORECASE)
# nemotron auto-slices audio into ~30 s windows inside the engine (spike
# measurement); streamed chunk-line end times map onto that window grid so
# progress advances on real engine boundaries, never timers.
ENGINE_CHUNK_WINDOW_SECONDS = 30.0
# Streamed diarized chunk line produced by the engine on stdout as each
# window completes: "[HH:MM:SS.mmm --> HH:MM:SS.mmm] (speaker N) ..."
# (format reference: spike archive spike-crispasr/run_full_stdout.log).
_CHUNK_LINE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{1,3})\]")
# Engine version token as printed by ``crispasr --version`` (three-part
# dotted, optional leading v/V).
_VERSION_TOKEN = re.compile(r"v?\d+\.\d+\.\d+", re.IGNORECASE)
# Display name baked into the renderer Pipeline stamp (card t_3275e056);
# matches the pinned engine + Q4_K model of the shipped build.
BACKEND_DISPLAY_NAME = "CrispASR nemotron Q4_K"
_CUDA_RUNTIME_HINT = (
    "install libcudart.so.13 and libcublas.so.13 (apt: cuda-cudart-13-0 "
    "libcublas-13-0) or include their directory in CRISPASR_LIB_DIR"
)

logger = logging.getLogger("pipeline.transcribe")

# Probe results and last-run instrumentation, keyed per process.
_BACKEND_PROBE_CACHE: dict[str, dict[str, Any]] = {}
LAST_RUN_STATS: dict[str, Any] = {}


class TranscodeError(Exception):
    pass


class TranscriptionOOM(Exception):
    pass


class CrispasrError(RuntimeError):
    """The CrispASR backend could not produce a transcription."""


class CrispasrConfigError(CrispasrError):
    """CRISPASR_BIN / GGUF_MODEL / DIAR_METHOD configuration is invalid."""


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise CrispasrConfigError(f"{name} must be numeric") from exc


def crispasr_bin() -> str:
    return _env_str("CRISPASR_BIN")


def gguf_model_path() -> str:
    """Q4_K GGUF path from GGUF_MODEL (empty => engine auto-download)."""
    return _env_str("GGUF_MODEL")


def diar_method() -> str:
    method = _env_str("DIAR_METHOD", DIAR_METHOD_DEFAULT).lower()
    if method not in DIAR_METHODS:
        raise CrispasrConfigError(
            f"DIAR_METHOD={method!r} not supported; choose from "
            + ", ".join(sorted(DIAR_METHODS))
        )
    return method


def gpu_backend() -> str:
    backend = _env_str("CRISPASR_GPU_BACKEND", GPU_BACKEND_DEFAULT).lower()
    if backend not in GPU_BACKENDS:
        raise CrispasrConfigError(
            f"CRISPASR_GPU_BACKEND={backend!r} not supported; choose from "
            + ", ".join(sorted(GPU_BACKENDS))
        )
    return backend


def asr_language() -> str:
    return _env_str("CRISPASR_LANGUAGE", ASR_TARGET_LANG)


def build_crispasr_command(wav_path: str, out_base: str) -> list[str]:
    """Compose the crispasr CLI invocation for one whole-file diarized run."""
    backend = gpu_backend()
    command = [
        crispasr_bin() or "crispasr",
        "-f", str(wav_path),
        "--backend", ASR_BACKEND,
    ]
    model = gguf_model_path()
    if model:
        command += ["-m", model]
    else:
        command += ["-m", "auto", "--auto-download"]
    command += [
        "-l", asr_language(),
        # first-run trap: the CLI silently falls back to CPU when the CUDA
        # runtime is missing; pin the backend so the probe can prove CUDA.
        "--gpu-backend", backend,
        # strip non-speech tokens (e.g. inline "<en-US>" language tags)
        "--suppress-nst",
        "-oj",
        "-of", str(out_base),
    ]
    cache_dir = _env_str("CRISPASR_CACHE_DIR")
    if cache_dir:
        command += ["--cache-dir", cache_dir]
    method = diar_method()
    if method != "none":
        command += ["--diarize", "--diarize-method", method]
        if method == "sortformer":
            command += ["--diarize-model", "auto"]
        else:
            # foxnose/ecapa/sherpa need the embedder for global speaker passes
            command += ["--diarize-embedder", "auto"]
    return command


def _crispasr_env() -> dict[str, str]:
    """Environment for the subprocess: bundled libs first in LD_LIBRARY_PATH."""
    env = dict(os.environ)
    prepend = [os.path.dirname(os.path.abspath(crispasr_bin()))] if crispasr_bin() else []
    extra = _env_str("CRISPASR_LIB_DIR")
    if extra:
        prepend.append(extra)
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*prepend, current]) if prepend else current
    cache_dir = _env_str("CRISPASR_CACHE_DIR")
    if cache_dir:
        env["CRISPASR_CACHE_DIR"] = cache_dir
    return env


def _parse_crispasr_version(output: str) -> str | None:
    """Extract the engine version (e.g. ``v0.8.43``) from ``--version`` output.

    Only lines naming the app or an explicit ``version`` field are trusted,
    so unrelated semver-shaped build stamps (ggml, CUDA) are never misread
    as the engine version. Returns ``None`` when no version is parseable.
    """
    for key in ("crispasr", "version"):
        for line in output.splitlines():
            if key in line.lower():
                match = _VERSION_TOKEN.search(line)
                if match:
                    token = match.group(0)
                    return "v" + (token[1:] if token[0] in "vV" else token)
    return None


def probe_crispasr_binary() -> dict[str, Any]:
    """Parse ``crispasr --version`` build info; guard against CPU fallback.

    Returns ``{"backends": [...], "path": ...}``. Raises CrispasrConfigError
    when the binary is unset/missing/not executable, and CrispasrError when a
    pinned backend (cuda) is not registered by the build — the spike's silent
    CPU fallback is an error for us, never a fallback.
    """
    path = crispasr_bin()
    if not path:
        raise CrispasrConfigError(
            "CRISPASR_BIN is not set; point it at the crispasr CUDA build "
            f"expected layout: <dir>/{Q4K_GGUF_NAME} next to the binary via GGUF_MODEL"
        )
    cached = _BACKEND_PROBE_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise CrispasrConfigError(f"CRISPASR_BIN={path!r} is not an executable file")
    try:
        res = subprocess.run(
            [path, "--version"], capture_output=True, text=True,
            timeout=30, env=_crispasr_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CrispasrError(f"cannot execute CRISPASR_BIN={path!r}: {exc}") from exc
    if res.returncode != 0:
        raise CrispasrError(
            f"crispasr --version failed (exit {res.returncode}): "
            f"{(res.stderr or res.stdout).strip()[-400:]}"
        )
    probe = {"path": path, "backends": []}
    match = re.search(r"ggml backends\s*:\s*([^\n]+)", res.stdout, re.IGNORECASE)
    if match:
        probe["backends"] = [b.strip().lower() for b in match.group(1).split(",") if b.strip()]
    probe["version"] = _parse_crispasr_version(res.stdout)
    pinned = gpu_backend()
    if pinned != "cpu" and pinned not in probe["backends"]:
        hint = f"; {_CUDA_RUNTIME_HINT}" if pinned == "cuda" else ""
        raise CrispasrError(
            f"crispasr build at {path!r} registers backends {probe['backends']!r} "
            f"but GPU backend {pinned!r} is required{hint}. Continuing on CPU "
            "would be an ~11x silent slowdown."
        )
    _BACKEND_PROBE_CACHE[path] = probe
    logger.info("crispasr probe: version-ok backends=%s pinned=%s", probe["backends"], pinned)
    return probe


def crispasr_version() -> str | None:
    """Engine version captured by a completed backend probe, else ``None``."""
    for probe in _BACKEND_PROBE_CACHE.values():
        if probe.get("version"):
            return str(probe["version"])
    return None


def backend_identity() -> str:
    """Backend stamp for the renderer, e.g. ``CrispASR nemotron Q4_K @v0.8.43``.

    Reads only the probe cache — never spawns the binary — so rendering stays
    side-effect free; falls back to ``@unknown`` until a probe has run.
    """
    return f"{BACKEND_DISPLAY_NAME} @{crispasr_version() or 'unknown'}"


def _validate_runtime_config() -> None:
    probe_crispasr_binary()
    diar_method()  # validates DIAR_METHOD
    model = gguf_model_path()
    if model and not os.path.isfile(model):
        raise CrispasrConfigError(
            f"GGUF_MODEL={model!r} not found; expected the nemotron {Q4K_GGUF_NAME} file"
        )


# ---------------------------------------------------------------------------
# Segment mapping (crispasr JSON -> pipeline Segment dicts)
# ---------------------------------------------------------------------------

def _strip_lang_tags(text: str) -> str:
    return _LANG_TAG.sub("", text).strip()


def _parse_hms(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def _stream_chunk_end_seconds(line: str) -> float | None:
    """End time (seconds) of a streamed diarized chunk line, else ``None``.

    Non-matching stdout (engine banners, plain-text runs) is not a work
    boundary and is ignored by the progress stream.
    """
    match = _CHUNK_LINE.match(line.strip())
    if not match:
        return None
    try:
        return _parse_hms(match.group(1))
    except ValueError:
        return None


def _engine_chunk_total(duration: float,
                        window: float = ENGINE_CHUNK_WINDOW_SECONDS) -> int:
    """Engine chunk windows for one whole-file invocation of ``duration`` s.

    Returns 0 for an unknown duration (probe failure): callers must then
    fall back to coarse reporting instead of fabricating boundaries.
    """
    if duration <= 0:
        return 0
    return max(1, math.ceil(duration / window))


def _segment_times(item: Mapping[str, Any]) -> tuple[float, float] | None:
    offsets = item.get("offsets") or {}
    try:
        return float(offsets["from"]) / 1000.0, float(offsets["to"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        pass
    stamps = item.get("timestamps") or {}
    try:
        return _parse_hms(stamps["from"]), _parse_hms(stamps["to"])
    except (KeyError, TypeError, ValueError):
        return None


def _speaker_raw(item: Mapping[str, Any]) -> int | None:
    match = _SPEAKER_INDEX.search(str(item.get("speaker") or ""))
    return int(match.group(1)) if match else None


def _merge_fragments(segments: list[dict[str, Any]], min_seconds: float) -> list[dict[str, Any]]:
    """Merge sub-``min_seconds`` fragments into the previous turn.

    Nemotron's fixed 30 s auto-chunks orphan tail words into micro-segments
    whose too-short embedding window mis-clusters them to a spurious extra
    speaker (spike finding: 15/16 of the sub-2 s fragments on the 39-min
    fixture). The fragment's text joins the neighbour and the neighbour's
    speaker label wins.
    """
    if not segments:
        return segments
    merged: list[dict[str, Any]] = []
    pending = segments[0]
    for item in segments[1:]:
        if item["t1"] - item["t0"] < min_seconds:
            if merged:
                target = merged[-1]
            else:
                target = pending
            target["text"] = f"{target['text']} {item['text']}".strip()
            target["t0"] = min(target["t0"], item["t0"])
            target["t1"] = max(target["t1"], item["t1"])
            continue
        merged.append(pending)
        pending = item
    merged.append(pending)
    return merged


def _clamp_monotonic(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_end = 0.0
    for item in segments:
        item["t0"] = max(item["t0"], previous_end)
        item["t1"] = max(item["t1"], item["t0"])
        previous_end = item["t1"]
    return segments


def _normalize_speakers(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Raw '(speaker N)' labels -> 'Speaker N' in first-appearance order."""
    order: dict[int, int] = {}
    seen_unlabeled = False
    for item in segments:
        raw = item.pop("_speaker_raw", None)
        if raw is None:
            seen_unlabeled = True
            label = UNKNOWN_SPEAKER if order else "Speaker 1"
        else:
            if raw not in order:
                order[raw] = len(order) + 1
            label = f"Speaker {order[raw]}"
        item["speaker"] = label
    if seen_unlabeled and not order:
        logger.info("crispasr output carried no speaker labels (DIAR_METHOD=none)")
    return segments


def parse_crispasr_output(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map crispasr JSON to Segment dicts: {text, t0, t1, speaker}.

    Raises CrispasrError when the payload carries no usable segments —
    never returns a silent empty/fallback transcript.
    """
    raw = payload.get("transcription")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise CrispasrError("crispasr JSON missing 'transcription' list")
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        times = _segment_times(entry)
        if times is None:
            continue
        text = _strip_lang_tags(str(entry.get("text") or ""))
        if not text:
            continue
        items.append({"text": text, "t0": times[0], "t1": times[1],
                      "_speaker_raw": _speaker_raw(entry)})
    if not items:
        raise CrispasrError(
            f"crispasr produced no usable segments from {len(raw)} rows "
            "(empty transcription is a failure, not a fallback)"
        )
    items.sort(key=lambda item: (item["t0"], item["t1"]))
    fragments = _env_float("MIN_FRAGMENT_SECONDS", DEFAULT_MIN_FRAGMENT_SECONDS)
    items = _merge_fragments(items, max(0.0, fragments))
    items = _clamp_monotonic(items)
    return _normalize_speakers(items)


# ---------------------------------------------------------------------------
# VRAM instrumentation (subprocess backend -> nvidia-smi sampling)
# ---------------------------------------------------------------------------

def _nvidia_smi_used_mib() -> float | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    try:
        return float(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


class VramSampler:
    """Background 1 Hz GPU-memory sampler for one crispasr invocation."""

    def __init__(self) -> None:
        self.sample_seconds = max(0.1, _env_float("CRISPASR_VRAM_SAMPLE_SECONDS", 1.0))
        self.peak_mib = 0.0
        self.last_mib = 0.0
        self.samples = 0
        self.active = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def observe(self, value: float | None) -> None:
        if value is None:
            return
        self.samples += 1
        self.last_mib = value
        self.peak_mib = max(self.peak_mib, value)

    def collect_once(self) -> float | None:
        value = _nvidia_smi_used_mib()
        self.observe(value)
        return value

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.collect_once()
            self._stop.wait(self.sample_seconds)

    def start(self) -> None:
        if not self.active:
            logger.info("VRAM sampling disabled: nvidia-smi not found")
            return
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_seconds * 5 + 5)
            self._thread = None

    def log_summary(self, stage: str, limit_mib: float) -> None:
        status = "ok"
        if self.samples and limit_mib > 0 and self.peak_mib > limit_mib:
            status = "EXCEEDED"
        elif not self.samples:
            status = "unobserved"
        logger.info(
            "VRAM stage=%s allocated=%.1fMiB peak=%.1fMiB samples=%d cap=%.1fMiB status=%s",
            stage, self.last_mib, self.peak_mib, self.samples, limit_mib, status,
        )


def _make_vram_sampler() -> VramSampler:
    return VramSampler()


# ---------------------------------------------------------------------------
# Backend invocation
# ---------------------------------------------------------------------------

def _stdout_pump(stream, sink: queue.Queue) -> None:
    """Read engine stdout line-by-line into ``sink``; ``None`` marks EOF."""
    try:
        for line in stream:
            sink.put(line)
    except (OSError, ValueError):
        pass  # stream torn down on timeout kill; the deadline owns shutdown
    finally:
        sink.put(None)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate the streamed engine's whole process group, TERM then KILL.

    ``subprocess.run``'s timeout only signals the direct child; a streamed
    run must also clear grandchildren that keep the stdout pipe (and the
    GPU) alive. The engine is spawned in its own session
    (``start_new_session=True``) so the group is exactly the invocation.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # already reaped
    for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 5.0)):
        with contextlib.suppress(OSError):
            os.killpg(pgid, sig)
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=5.0)


def _stderr_tail(path: str, limit: int = 800) -> str:
    try:
        text = Path(path).read_text(errors="replace").strip()
    except OSError:
        text = ""
    return text[-limit:] or "<no output>"


def _run_crispasr(wav_path: str, work_dir: str,
                  progress: Callable[[str, int, int], None] | None = None,
                  chunk_total: int = 0) -> list[dict[str, Any]]:
    """One whole-file crispasr invocation -> parsed Segment dicts.

    Engine stdout is streamed line-by-line: each diarized chunk line is a
    real work boundary and advances ``progress("asr", done, chunk_total)``
    when the caller supplied a chunk total. stdout/stderr are never
    accumulated in memory (an earlier fix banned ``capture_output`` due to
    a pipe-buffer deadlock): stderr lands in ``work_dir`` and failure
    paths read only its tail. ``CRISPASR_TIMEOUT_SECONDS`` keeps its wall-
    clock semantics; on expiry the whole process group is killed.
    """
    started = time.monotonic()
    out_base = os.path.join(work_dir, "crispasr_out")
    command = build_crispasr_command(wav_path, out_base)
    timeout = _env_float("CRISPASR_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    limit = _env_float("VRAM_LIMIT_MIB", DEFAULT_VRAM_LIMIT_MIB)
    sampler = _make_vram_sampler()
    sampler.start()
    stderr_path = os.path.join(work_dir, "crispasr_stderr.log")
    try:
        try:
            with open(stderr_path, "wb") as stderr_file:
                proc = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=stderr_file,
                    text=True, bufsize=1, env=_crispasr_env(), cwd=work_dir,
                    start_new_session=True,
                )
        except OSError as exc:
            raise CrispasrError(f"cannot execute crispasr at {command[0]!r}: {exc}") from exc
        deadline = started + timeout
        done = 0

        def _on_line(line: str) -> None:
            nonlocal done
            if progress is None or chunk_total <= 0:
                return
            end = _stream_chunk_end_seconds(line)
            if end is None:
                return
            index = min(chunk_total, max(1, math.ceil(end / ENGINE_CHUNK_WINDOW_SECONDS)))
            if index > done:  # monotone: interim/overlap lines never regress
                done = index
                progress("asr", done, chunk_total)

        lines: queue.Queue = queue.Queue()
        pump = threading.Thread(target=_stdout_pump, args=(proc.stdout, lines),
                                daemon=True)
        pump.start()
        timed_out = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = lines.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if line is None:  # stdout EOF — engine finished or closed the pipe
                    break
                _on_line(line)
            if not timed_out and proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if timed_out or proc.poll() is None:
                _kill_process_group(proc)
            with contextlib.suppress(OSError, ValueError):
                if proc.stdout is not None:
                    proc.stdout.close()
        if timed_out:
            raise CrispasrError(f"crispasr timed out after {timeout:.0f}s: {command[:1]}")
        returncode = proc.wait()
    finally:
        sampler.stop()
    engine_seconds = time.monotonic() - started
    if returncode != 0:
        raise CrispasrError(
            f"crispasr exited with code {returncode}: {_stderr_tail(stderr_path)}"
        )
    if progress is not None and chunk_total > 0 and done < chunk_total:
        # The engine exited cleanly: every audio window is by definition
        # transcribed, even if the streamed lines covered fewer boundaries.
        progress("asr", chunk_total, chunk_total)
    json_path = out_base + ".json"
    if not os.path.isfile(json_path):
        raise CrispasrError(
            f"crispasr exited 0 but wrote no {json_path!r}; refusing silent fallback"
        )
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CrispasrError(f"cannot parse crispasr JSON at {json_path!r}: {exc}") from exc
    segments = parse_crispasr_output(payload)
    sampler.log_summary("crispasr_diarized_asr", limit)
    LAST_RUN_STATS.update({
        "engine_seconds": round(engine_seconds, 2),
        "vram_peak_mib": round(sampler.peak_mib, 1),
        "vram_samples": sampler.samples,
        "vram_limit_mib": limit,
        "segments": len(segments),
    })
    if sampler.samples and limit > 0 and sampler.peak_mib > limit:
        raise TranscriptionOOM(
            f"peak VRAM {sampler.peak_mib:.0f} MiB exceeded cap {limit:.0f} MiB"
        )
    logger.info(
        "crispasr run ok: %.1fs (%.1fx realtime) segments=%d",
        engine_seconds, max(1e-9, _probe_duration_quiet(wav_path) / engine_seconds),
        len(segments),
    )
    return segments


def _probe_duration_quiet(path: str) -> float:
    try:
        return _probe_duration(path)
    except TranscodeError:
        return 0.0


def decode_audio(input_path: str, output_wav: str) -> str:
    """Decode audio file to 16kHz mono PCM wav via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_wav,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise TranscodeError(f"ffmpeg failed to run: {exc}") from exc
    if res.returncode != 0:
        raise TranscodeError(f"ffmpeg failed: {res.stderr}")
    return output_wav


def _probe_duration(path: str | Path) -> float:
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if res.returncode != 0:
            raise TranscodeError(f"ffprobe failed: {res.stderr}")
        return float(res.stdout.strip())
    except OSError as exc:
        raise TranscodeError(f"ffprobe failed to run: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscodeError("ffprobe timed out") from exc
    except ValueError as exc:
        raise TranscodeError("ffprobe returned non-numeric duration") from exc


def _mock_segments() -> list[dict[str, Any]]:
    return [
        {"text": "Hello and welcome to the AutoDiarist pipeline demo.",
         "t0": 0.0, "t1": 3.5, "speaker": "Speaker 1"},
        {"text": "Thanks for having me, let's review the diarization results.",
         "t0": 3.8, "t1": 7.2, "speaker": "Speaker 2"},
    ]


def transcribe_and_diarize(
    audio_path: str,
    mock: bool = False,
    duration: float | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Transcribe and diarize one audio file via the CrispASR backend.

    Returns Segment dicts ``{text, t0, t1, speaker}`` with monotonic times
    and ``Speaker N`` labels ordered by first appearance. The engine handles
    its own ~30 s chunking, so the whole file runs in one bounded-VRAM
    invocation. ``duration`` (or a cheap probe) fixes the engine chunk
    count; ``progress`` then receives real per-chunk reports
    ``("asr", done, total)`` from the streamed engine stdout, plus the
    final ``("diar", 1, 1)`` assembly report.
    """
    if mock or os.environ.get("MOCK_PIPELINE") == "1":
        return _mock_segments()

    _validate_runtime_config()
    chunk_total = 0
    if progress is not None:
        known = duration if duration and duration > 0 else _probe_duration_quiet(audio_path)
        chunk_total = _engine_chunk_total(known)
        if chunk_total == 0:
            # Duration unknown: keep the legacy coarse pass pair rather than
            # fabricating chunk boundaries.
            progress("asr", 1, 2)
    with tempfile.TemporaryDirectory(prefix="autodiarist-crispasr-") as work_dir:
        segments = _run_crispasr(str(audio_path), work_dir,
                                 progress=progress, chunk_total=chunk_total)
    if progress is not None:
        progress("diar", 1, 1)
    return segments


def transcribe_audio(audio_path: str, mock: bool = False) -> list[dict[str, Any]]:
    """Compatibility wrapper: returns [{'text', 't0', 't1', 'speaker'}, ...]."""
    return transcribe_and_diarize(audio_path, mock=mock)
