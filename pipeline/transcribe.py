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
  the peak exceeds the FLEET cap — ``VRAM_LIMIT_MIB`` (default 3584 = 3.5
  GiB) x ``MAX_PROCESSING_GLOBAL``. The sampler reads the CARD TOTAL, so
  under global concurrency (card t_dbad2235) the per-job share is scaled by
  the worker count; unset/invalid N is 1 and the serial math is unchanged.
* live progress streaming (cards t_1eb108e0, t_2e901945) — measured on the
  pinned binary, the engine's diarized stdout transcript is APP-HELD until
  the final merge (``stdbuf -oL`` and even a pty deliver nothing mid-run
  on long inputs), so stdout chunk lines alone left the ``transcribing``
  band frozen for hours on a 140-minute job. The in-run boundary source is
  the engine's ``-pp`` stderr stream — one unbuffered line per completed
  ~30 s slice ("crispasr: progress =  X% (k/N slices)") — which the adapter
  tails from the stderr file and publishes as ``("asr", done, total)``
  reports, so progress advances on real engine boundaries, never timers
  (mapping lives in ``api/main.py::_engine_progress``).
* the silent CPU fallback trap (missing libcudart/libcublas drops crispasr
  to CPU ~11x slower without error): the adapter pins ``--gpu-backend`` and
  probes ``crispasr --version`` for CUDA backend registration before
  spending minutes on a job, failing fast with a typed error.
"""

from __future__ import annotations

import contextlib
import inspect
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
# sherpa (external sherpa-onnx subprocess: segmentation + embedding +
# clustering) is the default since card t_5a287979: the foxnose global pass
# with its default BIC speaker counter collapses panel recordings
# (5 speakers -> 1, "cosine_similarity_single_speaker" veto) on hard audio;
# the engine's own --help routes reliable multi-speaker diarization to
# sherpa/ecapa. foxnose remains selectable (it is faster and was fine on
# well-separated audio).
DIAR_METHOD_DEFAULT = "sherpa"
# Methods that shell out to sherpa-onnx and REQUIRE both ONNX models;
# without them the engine emits segments with no speaker labels (silent
# degradation, rejected at startup since t_5a287979).
SHERPA_MODEL_METHODS = frozenset({"sherpa", "ecapa"})
# Canonical model artifacts baked at /opt/models by the Dockerfile (all
# three URLs are k2-fsa/sherpa-onnx GitHub release assets -- never guess
# them; the embedding filename is also the one the engine's own help text
# prints, "Download them from https://github.com/k2-fsa/sherpa-onnx").
# The SEGMENTATION model is reverb-diarization-v1, not the plain
# pyannote-segmentation-3-0: on NV-host A/Bs over the card fixtures it
# halves the single-speaker over-split (BestPractice 7 -> 3 labels at
# thr 0.35) while keeping the panel file at 5 speakers -- the reverb-
# robust training matches conference/panel audio.
SHERPA_SEGMENT_NAME = "sherpa-onnx-reverb-diarization-v1.onnx"
SHERPA_EMBEDDING_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
SHERPA_MODELS_DIR = "/opt/models"
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


class JobCancelled(Exception):
    """The client cancelled this job while it was running (card t_fef54150).

    A deliberate terminal class: the cancel kill makes the engine exit by
    signal (SIGTERM/15), which the generic failure path used to misread as
    an engine crash. Cancellation is never a timeout and never a failure,
    so no post-kill retry or fallback handler may treat it as one.
    """


class CrispasrConfigError(CrispasrError):
    """CRISPASR_BIN / GGUF_MODEL / DIAR_METHOD configuration is invalid."""


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise CrispasrConfigError(f"{name} must be numeric") from exc


def _env_positive_float(name: str, default: float) -> float:
    """Float knob with a house-style safe fallback: unset/blank keeps the
    default; non-numeric, zero or negative warn and keep the default.
    A heartbeat knob must never be able to crash a running stage or set
    an unbounded/forever cadence on a config typo (card t_868131e3)."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "config: %s=%r is not a number; using %.3f", name, raw, default)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "config: %s=%r must be a positive number; using %.3f",
            name, raw, default)
        return default
    return value


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


def sherpa_segment_model() -> str:
    return _env_str("SHERPA_SEGMENT_MODEL", f"{SHERPA_MODELS_DIR}/{SHERPA_SEGMENT_NAME}")


def sherpa_embedding_model() -> str:
    return _env_str("SHERPA_EMBEDDING_MODEL", f"{SHERPA_MODELS_DIR}/{SHERPA_EMBEDDING_NAME}")


def sherpa_bin() -> str:
    """Optional explicit sherpa-onnx CLI (image default is a wrapper script
    that injects --clustering.cluster-threshold; empty => engine PATH lookup)."""
    return _env_str("SHERPA_BIN")


def sherpa_split_enabled() -> bool:
    """True when the pipeline takes the diarization off the engine.

    Card t_7685c6da: the engine runs its external-sherpa diarization
    pre-pass BEFORE the ASR slice loop, so the -pp per-slice progress
    lines are withheld behind the pre-pass for the whole run. On the
    shipped path (DIAR_METHOD=sherpa/ecapa with an explicit SHERPA_BIN —
    always set by the Dockerfile ENV and both compose files) the pipeline
    instead runs ASR-only through the engine and calls the SAME wrapper
    itself (run_sherpa_diarization), restoring live progress. An unset
    SHERPA_BIN (host probes, bare engine runs) keeps the legacy
    single-invocation contract.
    """
    return diar_method() in SHERPA_MODEL_METHODS and bool(sherpa_bin())


def _require_sherpa_artifacts(method: str) -> None:
    """AC4 (t_5a287979): sherpa/ecapa without their models is a SILENT
    no-label failure mode at runtime; reject it at startup instead."""
    missing = [
        f"{var}={path!r}"
        for var, path in (
            ("SHERPA_SEGMENT_MODEL", sherpa_segment_model()),
            ("SHERPA_EMBEDDING_MODEL", sherpa_embedding_model()),
        )
        if not os.path.isfile(path)
    ]
    if missing:
        raise CrispasrConfigError(
            f"DIAR_METHOD={method!r} requires the sherpa-onnx models; missing: "
            + ", ".join(missing)
            + " — add them to the image (see README 'Diarization models') or "
            "choose another DIAR_METHOD"
        )
    binary = sherpa_bin()
    if binary and not os.access(binary, os.X_OK):
        raise CrispasrConfigError(
            f"SHERPA_BIN={binary!r} is not an executable file; DIAR_METHOD="
            f"{method!r} shells out to it"
        )


def build_crispasr_command(wav_path: str, out_base: str) -> list[str]:
    """Compose the crispasr CLI invocation for one whole-file run.

    Card t_7685c6da (long-file progress): for the external-sherpa methods
    with an explicit ``SHERPA_BIN`` (the shipped default), the invocation
    is ASR-ONLY — no ``--diarize``. The pinned engine runs
    ``crispasr_compute_sherpa_cache()`` BEFORE its slice loop
    (crispasr_run.cpp order, confirmed on v0.8.43), so on a long file the
    slow diarization pre-pass withholds every ``-pp`` per-slice progress
    line until ASR is effectively over: the bar froze at 30% and the ETA
    was absent for ~83% of the wait. With the diarization delegated to
    ``run_sherpa_diarization`` (invocation 2), the engine publishes its
    slice boundaries live, exactly as measured on the no-diarize path
    (42 lines on the 7128 s file, first at slice 1).

    ``DIAR_METHOD=none`` never passed ``--diarize`` and is untouched; the
    in-engine methods (foxnose, sortformer) keep the single-invocation
    contract — the split is scoped to the sherpa/ecapa subprocess path,
    and only when ``SHERPA_BIN`` names the wrapper explicitly (an unset
    SHERPA_BIN means engine PATH lookup and keeps the legacy command).
    """
    backend = gpu_backend()
    command = [
        crispasr_bin() or "crispasr",
        "-f", str(wav_path),
        "--backend", ASR_BACKEND,
        # Card t_2e901945: -pp makes the engine write one unbuffered
        # "progress = X% (k/N slices)" line to STDERR per completed audio
        # slice. Verified live on the pinned 0.8.43 binary (86/86 lines,
        # real-time cadence over 39 min). This is the only in-run work-
        # boundary channel the engine has: its diarized stdout transcript
        # is held until the merge stage and stdbuf/PTY tricks change that
        # (see _run_crispasr docstring).
        "-pp",
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
    if sherpa_split_enabled():
        # Card t_7685c6da: ASR-ONLY invocation. The engine's --diarize path
        # runs the external sherpa pre-pass BEFORE the slice loop that
        # emits -pp progress (crispasr_run.cpp order, pinned v0.8.43), so a
        # long file with a slow diarization published ZERO progress lines
        # (measured: 0 of 264 on the 7128 s panel, 42 with --diarize
        # removed). The pipeline does the diarization itself
        # (run_sherpa_diarization) and labels segments by time overlap
        # (_merge_diarization_regions). Only the diarization flags come off, so
        # the ASR decode path is the identical one (AC2 gate). -oj is
        # swapped for -ojf: nemotron populates word timestamps internally
        # on BOTH paths (the engine's own Phase-2 turn split consumes
        # seg.words whenever --diarize runs), and the JSON writer only
        # emits them in full mode — the pipeline-side split needs the same
        # word vector to reproduce the engine's segment boundaries. The
        # writer adds fields; the decoder is untouched (nemotron's words/
        # tokens are always computed, crispasr_backend_nemotron.cpp).
        command = list(command)
        command[command.index("-oj")] = "-ojf"
        return command
    if method != "none":
        command += ["--diarize", "--diarize-method", method]
        if method == "sortformer":
            command += ["--diarize-model", "auto"]
        else:
            # foxnose/ecapa/sherpa need the embedder for global speaker passes
            command += ["--diarize-embedder", "auto"]
        if method in SHERPA_MODEL_METHODS:
            # The external sherpa-onnx subprocess needs both ONNX models;
            # without them the engine emits speaker=None segments (the
            # silent-degradation mode rejected at startup, AC4 t_5a287979).
            command += [
                "--sherpa-segment-model", sherpa_segment_model(),
                "--sherpa-embedding-model", sherpa_embedding_model(),
            ]
            binary = sherpa_bin()
            if binary:
                command += ["--sherpa-bin", binary]
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
    method = diar_method()  # validates DIAR_METHOD
    if method in SHERPA_MODEL_METHODS:
        _require_sherpa_artifacts(method)
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


# Slice-progress line the engine writes to stderr per completed audio
# slice under -pp (card t_2e901945). Format captured verbatim from the
# pinned crispasr 0.8.43 on the NV host; the percent field carries either
# an int or one decimal and may be padded with multiple spaces.
_SLICE_PROGRESS = re.compile(
    r"progress = +\d+(?:\.\d+)?% \((\d+)/(\d+) slices\)")


def _slice_progress_report(line: str) -> tuple[int, int] | None:
    """``(slice_done, slice_total)`` from a -pp stderr line, else ``None``.

    Only engine slice completions count as work boundaries here; noise
    (banners, ggml backend lines, diarize chatter) returns ``None`` so the
    progress stream can never advance on a non-event.
    """
    match = _SLICE_PROGRESS.search(line)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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


def _normalize_speakers(
    segments: list[dict[str, Any]], method: str | None = None
) -> list[dict[str, Any]]:
    """Raw '(speaker N)' labels -> 'Speaker N' in first-appearance order.

    AC4 guard (card t_5a287979): when the configured DIAR_METHOD is a
    labeling method (anything but 'none') and the engine produced NO
    speaker labels anywhere, that is the silent sherpa/ecapa degradation
    (missing models -> speaker=None on every segment) -- a hard failure,
    never a fake single-speaker transcript. A PRESENT single label stays
    a legitimate one-speaker result (AC7: do not invent speakers).
    """
    if method is None:
        method = diar_method()
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
        if method == "none":
            logger.info("crispasr output carried no speaker labels (DIAR_METHOD=none)")
        else:
            raise CrispasrError(
                f"crispasr produced no speaker labels for DIAR_METHOD={method!r} "
                "— the diarization pass failed silently (missing or miswired "
                "--sherpa-* models?); refusing to emit a fake single-speaker "
                "transcript"
            )
    return segments


def parse_crispasr_output(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map crispasr JSON to Segment dicts: {text, t0, t1, speaker}.

    Raises CrispasrError when the payload carries no usable segments —
    never returns a silent empty/fallback transcript.
    """
    items = _parse_crispasr_items(payload)
    return _finalize_segments(items)


def _parse_crispasr_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """crispasr JSON -> raw item dicts (sorted by time, labels unmapped).

    Shared front-end of the single-invocation path (speaker labels come
    from the engine) and the split path (card t_7685c6da: the engine ran
    ASR-only, labels arrive later from _merge_diarization_regions). ``-ojf``
    word timestamps are kept as ``_words`` for the split path's
    speaker-turn splitting and stripped again in _finalize_segments.
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
        item: dict[str, Any] = {"text": text, "t0": times[0], "t1": times[1],
                                "_speaker_raw": _speaker_raw(entry)}
        words = entry.get("words")
        if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
            parsed: list[dict[str, Any]] = []
            for word in words:
                if not isinstance(word, Mapping):
                    continue
                try:
                    parsed.append({"text": str(word.get("text") or ""),
                                   "t0": float(word["t0"]) / 100.0,
                                   "t1": float(word["t1"]) / 100.0})
                except (KeyError, TypeError, ValueError):
                    continue
            if parsed:
                item["_words"] = parsed
        items.append(item)
    if not items:
        raise CrispasrError(
            f"crispasr produced no usable segments from {len(raw)} rows "
            "(empty transcription is a failure, not a fallback)"
        )
    items.sort(key=lambda item: (item["t0"], item["t1"]))
    return items


def _finalize_segments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tail half of parse_crispasr_output: fragment merge, clamp, labels."""
    fragments = _env_float("MIN_FRAGMENT_SECONDS", DEFAULT_MIN_FRAGMENT_SECONDS)
    items = _merge_fragments(items, max(0.0, fragments))
    items = _clamp_monotonic(items)
    for item in items:
        item.pop("_words", None)
    return _normalize_speakers(items)


# ---------------------------------------------------------------------------
# Split-path diarization (card t_7685c6da)
#
# The pinned engine (CrispASR v0.8.43) runs its external-sherpa diarization
# pre-pass BEFORE the ASR slice loop (crispasr_run.cpp:
# crispasr_compute_sherpa_cache -> slice loop -> tick_slice_progress), so on
# a long file the -pp per-slice progress lines are withheld behind the
# pre-pass and the job publishes nothing until ASR is essentially over
# (measured: 0 of 264 slice lines on the 7128 s panel; the same file
# WITHOUT --diarize published 42, the first at slice 1). The pipeline
# therefore runs the engine ASR-only and performs the SAME sherpa
# invocation the engine would have, then merges speaker labels by time
# overlap, replicating the engine's assign_speakers_from_global_sherpa.
# ---------------------------------------------------------------------------

# A sherpa region line, both shapes the engine parses
# (crispasr_diarize_cli.cpp parse_sherpa_line):
#   "0.320 -- 3.680 speaker_00 duration=3.360"   (newer)
#   "0.320 3.680 0"                               (older)
# The STRICT shape ($2 == "--", label at end) is the same one the wrapper
# counts with: the CLI banner echoes the embedding model path, which embeds
# "speaker_16" and would poison a loose match (measured, card t_38e52d89).
_SHERPA_REGION_STRICT = re.compile(
    r"^(\d+(?:\.\d+)?)\s+--\s+(\d+(?:\.\d+)?)\s+speaker_(\d+)(?:\s+\S+)?\s*$")
_SHERPA_REGION_LEGACY = re.compile(
    r"^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s*$")


def _parse_sherpa_regions(text: str) -> list[dict[str, Any]]:
    """Sherpa stdout -> [{t0, t1, speaker}] regions (seconds).

    Non-region lines (config banner, blanks) are ignored exactly as the
    engine ignores them; ordering follows the subprocess output, matching
    SherpaSegment vector order.
    """
    regions: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _SHERPA_REGION_STRICT.match(line.strip())
        if match:
            t0, t1, spk = match.groups()
        else:
            match = _SHERPA_REGION_LEGACY.match(line.strip())
            if not match:
                continue
            t0, t1, spk = match.groups()
        regions.append({"t0": float(t0), "t1": float(t1),
                        "speaker": int(spk)})
    return regions


def _merge_diarization_regions(
    segments: list[dict[str, Any]], regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label ASR-only items with sherpa speakers, engine-faithful.

    Replicates crispasr_diarize_cli.cpp assign_speakers_from_global_sherpa
    (#110 global cache + #107 word-turn splitting), because the transcript
    gate (AC2) compares against the engine's own diarized output:

    * Phase 1 - dominant speaker per segment: overlap summed per speaker,
      winner scan over speaker indices ascending with a STRICT '>', so the
      LOWEST index wins a tie; a segment outside every region keeps its
      label (None). Times compare in seconds; the engine compares
      centiseconds — the region boundaries are 3-decimal (ms) and segment
      times are ms-derived, so cs comparisons are exact at ms granularity
      either way.
    * Phase 2 - split at speaker turns when word timestamps exist (the
      engine's -ojf-equivalent word vector): per-word dominant region
      (max overlap, strict '>'), carry-forward for unaligned words,
      back-fill leading unknowns, no split when all words agree, sub-
      segments rebuilt from the word run exactly as the engine does.
    * MIN_FRAGMENT_SECONDS merge stays OUR rendering stage (it runs after
      splitting on both paths): _finalize_segments applies it exactly once
      hereafter, matching the legacy order where the engine had already
      split by the time our merge saw the JSON.
    """
    if not regions:
        raise CrispasrError(
            "sherpa diarization produced no speaker regions — refusing a "
            "label-less transcript (silent-degradation guard, AC4 parity "
            "with t_5a287979)")
    out: list[dict[str, Any]] = []
    for item in segments:
        best = _dominant_speaker(item["t0"], item["t1"], regions)
        item["_speaker_raw"] = best
        words = item.get("_words") or []
        if not words:
            out.append(item)
            continue
        word_spk = [_dominant_speaker(w["t0"], w["t1"], regions) for w in words]
        # carry-forward: an unaligned word inherits the previous label
        last = None
        for i, spk in enumerate(word_spk):
            if spk is not None:
                last = spk
            elif last is not None:
                word_spk[i] = last
        # back-fill leading unknowns from the first known label
        first_known = next((s for s in word_spk if s is not None), None)
        if first_known is not None:
            for i in range(len(word_spk)):
                if word_spk[i] is None:
                    word_spk[i] = first_known
                else:
                    break
        if all(s == word_spk[0] for s in word_spk):
            out.append(item)
            continue
        run_start = 0
        for i in range(1, len(words) + 1):
            if i < len(words) and word_spk[i] == word_spk[run_start]:
                continue
            run = words[run_start:i]
            sub = {"text": " ".join(w["text"] for w in run),
                   "t0": run[0]["t0"] if run[0]["t0"] > 0 else item["t0"],
                   "t1": run[-1]["t1"] if run[-1]["t1"] > 0 else item["t1"],
                   "_speaker_raw": (word_spk[run_start]
                                    if word_spk[run_start] is not None
                                    else item["_speaker_raw"]),
                   "_words": run}
            out.append(sub)
            run_start = i
    return out


def _dominant_speaker(t0: float, t1: float,
                      regions: list[dict[str, Any]]) -> int | None:
    """Max summed time-overlap speaker index over [t0, t1], engine order.

    The engine accumulates overlaps in region order, then scans speaker
    indices ascending with strict '>' — equal totals keep the earlier
    (lower) index. Zero-length overlap never counts (engine: hi > lo).
    """
    overlap: dict[int, float] = {}
    for region in regions:
        lo = max(t0, region["t0"])
        hi = min(t1, region["t1"])
        if hi > lo:
            overlap[region["speaker"]] = (
                overlap.get(region["speaker"], 0.0) + (hi - lo))
    best, best_overlap = None, 0.0
    for spk in sorted(overlap):
        if overlap[spk] > best_overlap:
            best, best_overlap = spk, overlap[spk]
    return best


def build_sherpa_diarize_command(wav_path: str) -> list[str]:
    """The diarization subprocess argv, engine-contract exact.

    Mirrors crispasr_compute_sherpa_cache + make_sherpa_args
    (crispasr_diarize_cli.cpp): the engine ALWAYS forwards
    --clustering.num-clusters=0 plus the two model paths and the 16 kHz
    mono wav it was given; the shipped SHERPA_BIN wrapper injects the
    cluster threshold, thread caps and the speaker-count ceiling around
    exactly this argv.
    """
    return [
        sherpa_bin(),
        "--clustering.num-clusters=0",
        f"--segmentation.pyannote-model={sherpa_segment_model()}",
        f"--embedding.model={sherpa_embedding_model()}",
        str(wav_path),
    ]


def _sherpa_timeout_seconds(duration: float) -> float:
    """Engine parity: crispasr_cli_process::timeout_from_audio_samples.

    CRISPASR_SHERPA_TIMEOUT_SEC overrides when parsed as a positive
    integer; else max(120, ceil(audio_s) * 5) — the same formula the
    engine applies to its own sherpa pre-pass, so a job that fit the
    engine's budget fits ours.
    """
    raw = _env_str("CRISPASR_SHERPA_TIMEOUT_SEC")
    if raw.isdigit() and int(raw) > 0:
        return float(int(raw))
    audio_s = max(1, math.ceil(duration if duration > 0 else 0.0))
    return max(120.0, audio_s * 5.0)


# ---------------------------------------------------------------------------
# Diarization-stage heartbeat (card t_868131e3)
#
# The sherpa CLI prints nothing while it runs, so between the last ASR
# slice boundary and the finished region table the stage published EXACTLY
# ONE event — ("diar", 1, 1) — and a 140-minute job went silent for ~50
# minutes. The fix is a time-based heartbeat: total counts uniform work
# chunks of the diarization stage's ESTIMATED cost, done advances with
# elapsed time toward total (never reaching it), and the true
# (total, total) boundary is published when the region table parses.
# ---------------------------------------------------------------------------

# The stage's cost model is the SAME rate constant the API's ETA uses
# (PROCESSING_SECONDS_PER_AUDIO_SECOND; card: "feed the same estimate
# rather than inventing a second source of truth").
DIAR_HEARTBEAT_INTERVAL_DEFAULT = 2.0     # seconds between published ticks
DIAR_CHUNK_ESTIMATE_DEFAULT = 2.0         # estimated seconds per tick
DEFAULT_PROCESSING_RATE = 0.185           # s CPU per audio s (T400, api/main twin)


def processing_rate_per_audio_second() -> float:
    """Seconds of processing per second of audio, from the shared ETA
    knob (default 0.185, the measured T400 rate). Exact 0 or junk falls
    back to the default here — the heartbeat needs A rate, unlike the
    API surface where 0 disables ETA emission."""
    raw = (os.environ.get("PROCESSING_SECONDS_PER_AUDIO_SECOND") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_PROCESSING_RATE
    except ValueError:
        logger.warning(
            "config: PROCESSING_SECONDS_PER_AUDIO_SECOND=%r is not a number;"
            " using %.3f", raw, DEFAULT_PROCESSING_RATE)
        value = DEFAULT_PROCESSING_RATE
    return value if value > 0 else DEFAULT_PROCESSING_RATE


def diar_heartbeat_settings() -> tuple[float, float]:
    """(interval_seconds, chunk_estimate_seconds), both positive."""
    return (
        _env_positive_float("DIAR_HEARTBEAT_INTERVAL_SECONDS",
                            DIAR_HEARTBEAT_INTERVAL_DEFAULT),
        _env_positive_float("DIAR_CHUNK_ESTIMATE_SECONDS",
                            DIAR_CHUNK_ESTIMATE_DEFAULT),
    )


def diar_tick_plan(*, duration: float, rate: float,
                   chunk_seconds: float = DIAR_CHUNK_ESTIMATE_DEFAULT,
                   ) -> tuple[int, float]:
    """(total_ticks, seconds_per_tick) for the diarization stage.

    total = ceil(duration x rate / chunk_seconds) — uniform estimated work
    chunks of the stage's linear-model cost. seconds_per_tick = chunk /
    rate, so a job that runs AT the model advances exactly one tick per
    interval; the cap-then-final design keeps honesty when it runs slower
    or faster. duration/rate at or below zero (unknown duration) yields
    total 0: the caller must emit NO ticks rather than fabricate a
    counter for a file it cannot measure.
    """
    if duration is None or duration <= 0 or rate <= 0 or chunk_seconds <= 0:
        return 0, 0.0
    cost = float(duration) * float(rate)
    total = max(1, math.ceil(cost / float(chunk_seconds)))
    return total, float(chunk_seconds) / float(rate)


def _callback_accepts_kwarg(cb, name: str) -> bool:
    """True when `cb` can receive the keyword `name` (a named parameter or
    **kwargs). Legacy 3-arg callbacks (cards t_2e901945/t_7685c6da wired
    `lambda stage, done, total: ...`) keep working unchanged — they simply
    do not see the estimate label."""
    try:
        params = inspect.signature(cb).parameters
    except (TypeError, ValueError):
        return True
    for param in params.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name:
            return True
    return False


class _DiarHeartbeat:
    """Background tick publisher for one diarization stage (AC1-AC4).

    Driven by the STAGE lifetime, not the subprocess's: the wrapper's
    ceiling re-pass restarts the diarization underneath the wait loop
    (pass 1 -> label count -> re-pass), so the clock starts once, when
    the stage spawns, and never resets — the count cannot rewind across
    a restart. Ticks are strictly below total (progress must not fake
    completion), monotonic, and carry estimated=True + the method label
    so no client can mistake a heartbeat for a real work boundary. The
    thread touches no API lock; it only calls the progress callback,
    which takes the queue lock for its own short write.
    """

    def __init__(self, progress, total: int, interval: float,
                 tick_seconds: float) -> None:
        self._progress = progress
        self._labelled = _callback_accepts_kwarg(progress, "estimated")
        self.total = total
        self._interval = interval
        self._tick = tick_seconds
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._published = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.total <= 0:
            return
        self._thread = threading.Thread(target=self._loop,
                                         name="diar-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.publish_now()

    def publish_now(self) -> bool:
        """Advance the elapsed-time model and publish if the value moved.
        Returns True when a new tick was published. Monotonic by
        construction: the model value only grows with elapsed time, and
        an advance beyond total - 1 is held there until the real boundary
        lands. EVERY publication here is labelled estimated=True — even
        one triggered by an observed region line, because the VALUE is
        model-derived, not a measured count of work (AC4: a client must
        never mistake it for a real slice boundary)."""
        if self.total <= 0:
            return False
        elapsed = time.monotonic() - self._started
        model = int(elapsed / self._tick) if self._tick > 0 else 0
        done = min(model, self.total - 1)
        if done <= self._published:
            return False
        self._published = done
        try:
            if self._labelled:
                self._progress("diar", done, self.total,
                               estimated=True,
                               estimate_method="elapsed_time_model")
            else:
                self._progress("diar", done, self.total)
        except Exception:  # a progress sink must never kill the stage
            logger.exception("diarization heartbeat publish failed")
        return True

    def last_published(self) -> int:
        return self._published

    def stop(self) -> None:
        """Idempotent: signal the loop AND join it, so the heartbeat never
        outlives the stage (AC5: nothing holds the process or a lock open
        after the stage exits, on any path — success, cancel, timeout)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None


def run_sherpa_diarization(
    wav_path: str,
    progress: Callable[..., None] | None = None,
    cancel: CancelControl | None = None,
    duration: float = 0.0,
) -> list[dict[str, Any]]:
    """Invocation 2: the diarization subprocess, pipelined as its own stage.

    Spawns build_sherpa_diarize_command directly (SHERPA_BIN is validated
    executable at startup, AC4 t_5a287979), reads the region table from
    stdout, and publishes the ``("diar", done, total)`` series. Two
    signals, one band (card t_868131e3):

      * REAL boundaries — every strict region line the stdout pump
        delivers while the stage is in flight (the wrapper's ceiling
        re-pass streams its table before exit) — advances `done` and is
        NOT labelled.
      * A TIME-BASED HEARTBEAT for the silence between them: total counts
        uniform work chunks of the stage's linear-model cost (the SAME
        rate constant as the API ETA), `done` advances with elapsed time
        toward it, capped at total - 1 so a tick never claims completion.
        Every tick carries estimated=True + estimate_method, so a client
        can never mistake it for a measured boundary.

    The heartbeat clock starts with the STAGE, not the subprocess, so the
    wrapper's re-pass restart cannot rewind the count. The final event is
    always the true (total, total) boundary. Honour cancel kills the whole
    process group; the heartbeat thread stops on every exit path. A
    non-zero exit or an empty region table raises CrispasrError with the
    output tail — never a silent no-label transcript (the failure mode
    that card was built to refuse).
    """
    command = build_sherpa_diarize_command(wav_path)
    timeout = _sherpa_timeout_seconds(duration)
    if cancel is not None and cancel.cancelled():
        raise JobCancelled(
            "cancellation requested before the diarization stage started: "
            f"{cancel.token}")
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=_crispasr_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise CrispasrError(
            f"cannot execute diarization subprocess {command[0]!r}: {exc}") from exc
    if cancel is not None:
        cancel.attach(proc)
    heartbeat: _DiarHeartbeat | None = None
    if progress is not None:
        interval, chunk = diar_heartbeat_settings()
        total, tick_seconds = diar_tick_plan(duration=duration,
                                             rate=processing_rate_per_audio_second(),
                                             chunk_seconds=chunk)
        if total > 0:
            heartbeat = _DiarHeartbeat(progress, total, interval, tick_seconds)
            heartbeat.start()
    collected: list[str] = []
    timed_out = False
    cancelled = False
    try:
        lines: queue.Queue = queue.Queue()
        reader = threading.Thread(target=_stdout_pump,
                                  args=(proc.stdout, lines), daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        eof = False
        while not eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if cancel is not None and cancel.cancelled():
                cancelled = True
                break
            try:
                line = lines.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
            if line is None:  # stdout EOF
                eof = True
                continue
            collected.append(line)
            # REAL in-flight boundary: a strict region line is the
            # diarizer's own output, so it advances the count immediately
            # and is never labelled as an estimate.
            if heartbeat is not None and _SHERPA_REGION_STRICT.match(line.strip()):
                heartbeat.publish_now()
        reader.join(timeout=1.0)
        if not eof and not timed_out and not cancelled:
            # stdout closed but the child still wedged: bounded drain wait
            wait_for = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=min(wait_for, 0.5) if wait_for else 0.01)
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.cancelled():
                    cancelled = True
                elif time.monotonic() >= deadline:
                    timed_out = True
    finally:
        if timed_out or cancelled or proc.poll() is None:
            _kill_process_group(proc)
        with contextlib.suppress(OSError, ValueError):
            if proc.stdout is not None:
                proc.stdout.close()
        if cancel is not None:
            cancel.detach()
        if heartbeat is not None:
            heartbeat.stop()
    if cancelled or (cancel is not None and cancel.cancelled()):
        raise JobCancelled(
            f"cancellation requested during the diarization stage: {cancel.token}")
    if timed_out:
        raise CrispasrError(
            f"diarization timed out after {timeout:.0f}s: {command[:1]}")
    returncode = proc.wait()
    text = "".join(collected)
    if returncode != 0:
        raise CrispasrError(
            f"diarization subprocess failed (exit {returncode}): "
            f"{text[-800:] or '<no output>'}")
    regions = _parse_sherpa_regions(text)
    if not regions:
        raise CrispasrError(
            "diarization subprocess produced no speaker regions: "
            f"{text[-800:] or '<no output>'}")
    if progress is not None:
        # The TRUE completion boundary, never labelled as an estimate.
        # If no tick ever published (short path finishes inside one
        # interval, or the duration is unknown), the legacy ("diar", 1, 1)
        # event stays byte-identical (AC6); a moved band must land on
        # (total, total) so the final done == total == the true total (AC1).
        if heartbeat is not None and heartbeat.last_published() > 0:
            progress("diar", heartbeat.total, heartbeat.total)
        else:
            progress("diar", 1, 1)
    return regions


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


def _vram_fleet_multiplier() -> int:
    """Service-wide worker count N configured for this deployment
    (card t_dbad2235, planner addition on the concurrency card).

    The ``nvidia-smi`` sampler reads the CARD TOTAL — with N concurrent jobs,
    every job's sampler observes the aggregate of all N. VRAM_LIMIT_MIB keeps
    its documented meaning (a PER-JOB share); the ceiling a check actually
    compares against is the fleet cap, ``limit x N`` (approach (a)). Unset or
    invalid input falls back to 1 with a warning — the serial default — so
    the CLI path (which never sets the knob) and every existing deployment
    behave byte-for-byte as before. The env var name must match api/main.py's
    exactly (compose env keys are read by both surfaces).
    """
    raw = (os.environ.get("MAX_PROCESSING_GLOBAL") or "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "config: MAX_PROCESSING_GLOBAL=%r is not an integer; VRAM fleet "
            "cap uses 1x (serial semantics)", raw,
        )
        return 1
    if value < 1:
        logger.warning(
            "config: MAX_PROCESSING_GLOBAL=%d is invalid; VRAM fleet cap "
            "uses 1x (serial semantics)", value,
        )
        return 1
    return value


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


def _stderr_progress_pump(path: str, proc: subprocess.Popen, sink,
                          stop: threading.Event) -> None:
    """Follow the engine's stderr file, feeding every line to ``sink``.

    Card t_2e901945: the diarized stdout transcript is held by the engine
    until the merge stage, but stderr (a regular file here) is unbuffered,
    so ``-pp`` slice lines land the moment a slice completes. Polling the
    file is merely the read mechanism — progress values still come ONLY
    from real engine output; a silent engine yields no reports and the
    published progress stays frozen (an honest "no new work yet").
    Terminates when the engine exits (draining the rest of the file) or
    ``stop`` is set.
    """
    try:
        handle = open(path, "r", errors="replace")
    except OSError:
        return
    with handle:
        while True:
            line = handle.readline()
            if line:
                sink(line)
                continue
            if stop.is_set() or proc.poll() is not None:
                # One exit path only: drain every remaining byte first, so
                # the last slice lines of a finishing engine are never lost
                # to a stop request that races the final writes.
                for rest in handle:
                    sink(rest)
                return
            stop.wait(0.05)


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


class CancelControl:
    """Per-attempt cancellation handle wiring a client request to the live
    engine subprocess (card t_fef54150).

    Two signals, one truth: the engine adapter polls ``cancelled()`` at
    short granularity (stdout-read iterations and bounded process waits),
    while ``request()`` additionally terminates the whole process group of
    an already-attached subprocess, so a mid-chunk kill is immediate rather
    than waiting for the next poll. ``attach()`` closes the race where the
    cancel lands between claim and spawn: if the flag is set when the
    process appears, it is killed on arrival.

    The subprocess runs in its own session (``start_new_session=True``), so
    the group is exactly the one engine invocation — the same guarantee the
    timeout kill already relies on. ``detach()`` clears the reference before
    any cleanup can reap the pid: a stale pgid must never be signalled after
    the process is gone (pid reuse).
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def cancelled(self) -> bool:
        return self._event.is_set()

    def attach(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
            kill_now = self._event.is_set()
        if kill_now:
            _kill_process_group(proc)

    def detach(self) -> None:
        with self._lock:
            self._proc = None

    def request(self) -> bool:
        """Flag the job and kill the attached engine, if any. Returns True
        when a subprocess handle was present (live or just-exited); False
        means the caller must decide the orphan path."""
        self._event.set()
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        if proc.poll() is None:
            _kill_process_group(proc)
        return True


def _stderr_tail(path: str, limit: int = 800) -> str:
    try:
        text = Path(path).read_text(errors="replace").strip()
    except OSError:
        text = ""
    return text[-limit:] or "<no output>"


def _run_crispasr(wav_path: str, work_dir: str,
                  progress: Callable[[str, int, int], None] | None = None,
                  chunk_total: int = 0,
                  cancel: CancelControl | None = None,
                  raw_items: bool = False) -> list[dict[str, Any]]:
    """One whole-file crispasr invocation -> parsed Segment dicts.

    Live progress needs TWO channels (card t_2e901945, measured on the
    pinned 0.8.43 binary): the diarized stdout transcript is app-held by
    the engine until the final merge — neither ``stdbuf -oL`` nor a pty
    releases a single line earlier on a long input — so stdout chunk
    lines only refine progress at the end of the run. The in-run signal
    is the engine's ``-pp`` stderr stream (always passed by
    ``build_crispasr_command``): one unbuffered line per completed audio
    slice, tailed from the stderr file and reported as
    ``progress("asr", done, total)`` real work boundaries. Short jobs,
    whose stdout burst arrives promptly, behave exactly as before.
    stdout/stderr are never accumulated in memory (an earlier fix banned
    ``capture_output`` due to a pipe-buffer deadlock): stderr lands in
    ``work_dir`` and failure paths read only its tail.
    ``CRISPASR_TIMEOUT_SECONDS`` keeps its wall-clock semantics; on
    expiry the whole process group is killed.
    ``cancel`` (card t_fef54150) wires the client-facing cancel to this
    subprocess: the handle is attached the moment the engine spawns (and
    killed on arrival if the cancel landed first), the stdout-read loop
    polls the flag while the engine is chatty-quiet, and the bounded waits
    wake on it too. The ``finally`` then reuses the timeout kill — TERM
    then KILL over the whole group — and the exit is reported as
    ``JobCancelled``, never as a timeout and never as a generic engine
    failure (a signal-killed exit code 15/−15 must not be reclassified or
    retried into a resurrection).
    """
    started = time.monotonic()
    out_base = os.path.join(work_dir, "crispasr_out")
    command = build_crispasr_command(wav_path, out_base)
    timeout = _env_float("CRISPASR_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    limit = _env_float("VRAM_LIMIT_MIB", DEFAULT_VRAM_LIMIT_MIB)
    # The sampled quantity is the CARD TOTAL, so under global concurrency the
    # effective ceiling is the fleet cap (per-job share x N, card t_dbad2235
    # planner note): without this, N-1 sibling jobs could each fail a job for
    # memory that is not its own. Serial default (N=1) keeps the old math.
    fleet_mult = _vram_fleet_multiplier()
    fleet_cap = limit * fleet_mult if limit > 0 else 0.0
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
        if cancel is not None:
            # Live handle for the DELETE path (card t_fef54150): if the
            # cancel landed between claim and spawn, attach kills this
            # engine on arrival instead of letting it chew GPU time.
            cancel.attach(proc)
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
        # Card t_2e901945: tail the engine's stderr FILE for -pp slice
        # completions — the only channel that delivers work boundaries
        # DURING the run on long inputs (see _stderr_progress_pump).
        stderr_done = 0
        stderr_stop = threading.Event()

        def _on_stderr_line(line: str) -> None:
            nonlocal stderr_done
            if progress is None or chunk_total <= 0:
                return
            report = _slice_progress_report(line)
            if report is None:
                return
            slice_done, slice_total = report
            if slice_done > stderr_done:  # monotone: never regress
                stderr_done = slice_done
                progress("asr", slice_done, slice_total)

        stderr_pump = threading.Thread(
            target=_stderr_progress_pump,
            args=(stderr_path, proc, _on_stderr_line, stderr_stop),
            daemon=True,
        )
        stderr_pump.start()
        timed_out = False
        cancelled = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if cancel is not None and cancel.cancelled():
                    # Card t_fef54150: the handler-side kill normally beats
                    # this poll; the poll is the belt for a flag set without
                    # a live handle attached (e.g. cancel during decode).
                    cancelled = True
                    break
                try:
                    line = lines.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if line is None:  # stdout EOF — engine finished or closed the pipe
                    break
                _on_line(line)
            if not timed_out and not cancelled and proc.poll() is None:
                wait_for = deadline - time.monotonic()
                if cancel is not None:
                    # Bounded wait so a cancel that lands while parked here
                    # is observed within ~0.5s even without the direct kill.
                    wait_for = min(wait_for, 0.5)
                while True:
                    try:
                        proc.wait(timeout=max(0.0, wait_for))
                        break
                    except subprocess.TimeoutExpired:
                        if cancel is not None and cancel.cancelled():
                            cancelled = True
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            break
                        wait_for = deadline - time.monotonic()
                        if cancel is not None:
                            wait_for = min(wait_for, 0.5)
        finally:
            if timed_out or cancelled or proc.poll() is None:
                _kill_process_group(proc)
            with contextlib.suppress(OSError, ValueError):
                if proc.stdout is not None:
                    proc.stdout.close()
            stderr_stop.set()
            stderr_pump.join(timeout=2.0)
            if cancel is not None:
                # Drop the (possibly reaped) handle before anything else can
                # signal a recycled pid through this control.
                cancel.detach()
        if cancelled or (cancel is not None and cancel.cancelled()):
            # Cancel-wins: whatever the engine's exit code ended up being
            # (TERM-killed -15, SIGKILL -9, or a race to exit 0), the
            # client asked this job to stop and the terminal is cancelled —
            # never a timeout, never the generic failure class. The kill
            # from request() can beat the next poll iteration (EOF break),
            # so the control's own flag is authoritative here, not `cancelled`.
            raise JobCancelled(
                f"crispasr cancelled by client request after "
                f"{time.monotonic() - started:.1f}s (exit {proc.returncode}): "
                f"{command[:1]}")
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
    segments = _parse_crispasr_items(payload) if raw_items \
        else _finalize_segments(_parse_crispasr_items(payload))
    sampler.log_summary("crispasr_diarized_asr", fleet_cap)
    LAST_RUN_STATS.update({
        "engine_seconds": round(engine_seconds, 2),
        "vram_peak_mib": round(sampler.peak_mib, 1),
        "vram_samples": sampler.samples,
        "vram_limit_mib": limit,
        "vram_fleet_multiplier": fleet_mult,
        "vram_fleet_cap_mib": round(fleet_cap, 1),
        "segments": len(segments),
    })
    if sampler.samples and fleet_cap > 0 and sampler.peak_mib > fleet_cap:
        raise TranscriptionOOM(
            f"peak VRAM {sampler.peak_mib:.0f} MiB exceeded fleet cap "
            f"{fleet_cap:.0f} MiB (per-job {limit:.0f} MiB x {fleet_mult} "
            "concurrent workers; the nvidia-smi sampler reads the card "
            "total, not this job's share)"
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


def _mock_stage_hold(cancel: CancelControl | None) -> None:
    """Parking helper for the mock transcribe stage (card t_fef54150).

    ``MOCK_TRANSCRIBE_SECONDS`` (default 0 — today's instant behaviour) parks
    the mock stage so the queue/API layer can exercise mid-run cancellation
    without a GPU. The flag is polled at ~10 ms cadence, so the mock honours
    the same contract as the real engine: cancel wins promptly.
    """
    try:
        hold = float(os.environ.get("MOCK_TRANSCRIBE_SECONDS", "0") or 0)
    except ValueError:
        hold = 0.0
    end = time.monotonic() + max(0.0, hold)
    while True:
        if cancel is not None and cancel.cancelled():
            raise JobCancelled(
                "cancellation requested during the mock transcribe stage: "
                f"{cancel.token}")
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.01, remaining))


def transcribe_and_diarize(
    audio_path: str,
    mock: bool = False,
    duration: float | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    cancel: CancelControl | None = None,
) -> list[dict[str, Any]]:
    """Transcribe and diarize one audio file via the CrispASR backend.

    Returns Segment dicts ``{text, t0, t1, speaker}`` with monotonic times
    and ``Speaker N`` labels ordered by first appearance. The engine handles
    its own ~30 s chunking, so the whole file runs in one bounded-VRAM
    invocation. ``duration`` (or a cheap probe) fixes the engine chunk
    count; ``progress`` then receives real per-chunk reports
    ``("asr", done, total)`` from the streamed engine stdout, plus the
    ``("diar", ...)`` series from the pipeline-side diarization stage —
    heartbeat ticks labelled ``estimated=True`` while the subprocess
    runs, then the final true boundary (card t_868131e3). The legacy
    single-invocation path keeps its ``("diar", 1, 1)`` assembly report.
    ``cancel`` (card t_fef54150) is polled at this stage boundary BEFORE the
    engine spawns — a retry or requeue can never resurrect an already
    cancelled attempt by re-spawning it — threaded into the engine run for
    the mid-chunk kill, and polled again at the assembly boundary so a
    cancelled job never continues into the downstream passes (cancel wins
    even if the engine raced to finish).
    """
    if cancel is not None and cancel.cancelled():
        raise JobCancelled(
            "cancellation requested before the transcribe stage started: "
            f"{cancel.token}")
    if mock or os.environ.get("MOCK_PIPELINE") == "1":
        _mock_stage_hold(cancel)
        return _mock_segments()

    _validate_runtime_config()
    chunk_total = 0
    known = 0.0
    if progress is not None or sherpa_split_enabled():
        known = duration if duration and duration > 0 else _probe_duration_quiet(audio_path)
    if progress is not None:
        chunk_total = _engine_chunk_total(known)
        if chunk_total == 0:
            # Duration unknown: keep the legacy coarse pass pair rather than
            # fabricating chunk boundaries.
            progress("asr", 1, 2)
    split = sherpa_split_enabled()
    with tempfile.TemporaryDirectory(prefix="autodiarist-crispasr-") as work_dir:
        if split:
            # Card t_7685c6da: ASR publishes live; diarization is its own
            # stage on the SAME 16 kHz mono wav the engine was handed.
            items = _run_crispasr(str(audio_path), work_dir,
                                  progress=progress, chunk_total=chunk_total,
                                  cancel=cancel, raw_items=True)
            regions = run_sherpa_diarization(str(audio_path),
                                             progress=progress, cancel=cancel,
                                             duration=known)
            # CONTRACT: _merge_diarization_regions RETURNS a new list — the
            # Phase-2 word-split sub-segments exist only there, never on the
            # input items. Discarding the return (gate t_7685c6da regression)
            # finalized the original one-segment ASR item and collapsed the
            # panel clip to 1 speaker instead of 2; the direct-call merge
            # tests could not see it because non-split segments mutate
            # _speaker_raw in place.
            items = _merge_diarization_regions(items, regions)
            segments = _finalize_segments(items)
        else:
            segments = _run_crispasr(str(audio_path), work_dir,
                                     progress=progress, chunk_total=chunk_total,
                                     cancel=cancel)
        if progress is not None and not split:
            progress("diar", 1, 1)
    if cancel is not None and cancel.cancelled():
        raise JobCancelled(
            "cancellation accepted at the diarization boundary: "
            f"{cancel.token}")
    return segments


def transcribe_audio(audio_path: str, mock: bool = False) -> list[dict[str, Any]]:
    """Compatibility wrapper: returns [{'text', 't0', 't1', 'speaker'}, ...]."""
    return transcribe_and_diarize(audio_path, mock=mock)
