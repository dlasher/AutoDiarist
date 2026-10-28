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


class DiarizationDegradedError(CrispasrError):
    """The labeling diarization pass degraded to collapsed speaker
    regions (card t_dc5c2abb).

    Distinct from the zero-label CrispasrError (t_5a287979 AC4): there the
    engine emitted NO labels at all (missing models); here the engine
    exited 0 with a plausible-looking transcript whose speaker separation
    is meaningless — the measured shape is 2 regions for 8411 s after the
    global sherpa pre-pass was OOM-killed (exit 137) and the engine fell
    back per-slice. Both refuse loudly; this one names the degradation,
    the measured numbers, and the remedy.
    """


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


def sherpa_segment_model() -> str:
    return _env_str("SHERPA_SEGMENT_MODEL", f"{SHERPA_MODELS_DIR}/{SHERPA_SEGMENT_NAME}")


def sherpa_embedding_model() -> str:
    return _env_str("SHERPA_EMBEDDING_MODEL", f"{SHERPA_MODELS_DIR}/{SHERPA_EMBEDDING_NAME}")


def sherpa_bin() -> str:
    """Optional explicit sherpa-onnx CLI (image default is a wrapper script
    that injects --clustering.cluster-threshold; empty => engine PATH lookup)."""
    return _env_str("SHERPA_BIN")


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
    """Compose the crispasr CLI invocation for one whole-file diarized run."""
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


# ---------------------------------------------------------------------------
# Sherpa degradation detection (card t_dc5c2abb)
# ---------------------------------------------------------------------------
#
# The engine's GLOBAL sherpa pre-pass buffers the whole file; on long
# audio it can be OOM-killed (exit 137), after which the engine silently
# FALLS BACK to per-slice sherpa. Measured on the NV host (8411.072 s
# panel file): the fallback produced 2 speaker regions over 1 ASR segment
# and the run completed "successfully" with a collapsed-speaker transcript
# the zero-label guard does not catch.
#
# Engine stderr signatures (source-confirmed in crispasr_diarize_cli.cpp,
# v0.8.43: the 'over M ASR segments' line prints ONLY on the per-slice
# fallback path; the global success path prints 'sherpa global -> N'):
_SHERPA_GLOBAL_FAILED = re.compile(
    r"sherpa global run (?:failed with exit code \d+"
    r"|timed out after \d+ s|produced no segments)")
_SHERPA_FALLBACK_REGIONS = re.compile(
    r"sherpa →\s+(\d+) speaker regions over\s+(\d+) ASR segments")
_SHERPA_GLOBAL_REGIONS = re.compile(
    r"sherpa global →\s+(\d+) speaker regions")

# Calibration (measured legit counts, card body + planner receipt):
#   8411 s clean run  -> 472 regions, 6 speakers  (3.37 segments/min)
#   8411 s DEFECT     ->   2 regions              (0.0143 regions/min)
#   249 s panel       -> 18 segments, 3 speakers  (legit, exempt anyway)
#   137 s control     ->  1 region,  1 speaker    (legit single-speaker;
#         a BARE rate floor would reject it at 0.44/min — the exemption
#         below is what keeps AC3 true)
# The gap legit-vs-defect on long audio is a factor of 235, so the floor
# constants are not a knife's edge. Refuse only when BOTH conditions
# hold: audio long enough that a tiny region count cannot be a legitimate
# recording, AND the region count below a floor that scales with
# duration. Operator-tunable via the two knobs (catalogued in
# pipeline/config_vars.py); defaults are the measured shape.
DIARIZATION_MIN_REGIONS_PER_MINUTE_DEFAULT = 0.5
DIARIZATION_MIN_CHECK_SECONDS_DEFAULT = 600.0


def _parse_sherpa_stderr(text: str) -> dict[str, Any]:
    """Extract the engine's own sherpa degradation signals from stderr.

    Returns ``{global_failed, fallback_regions, fallback_asr_segments,
    global_regions}``; the *_regions keys are None when the line was never
    printed. Total on arbitrary input (never raises). LAST matching line
    wins per pattern: the fallback path prints one 'over M ASR segments'
    line per slice (slice-local numbers — see _reject_degraded_sherpa,
    where PRESENCE of the line is the signal, not its count), and the
    captured number is message evidence only.
    """
    signal: dict[str, Any] = {"global_failed": None,
                              "fallback_regions": None,
                              "fallback_asr_segments": None,
                              "global_regions": None}
    if not text:
        return signal
    for line in text.splitlines():
        match = _SHERPA_GLOBAL_FAILED.search(line)
        if match:
            signal["global_failed"] = line.strip()
        match = _SHERPA_FALLBACK_REGIONS.search(line)
        if match:
            signal["fallback_regions"] = int(match.group(1))
            signal["fallback_asr_segments"] = int(match.group(2))
        match = _SHERPA_GLOBAL_REGIONS.search(line)
        if match:
            signal["global_regions"] = int(match.group(1))
    return signal


def _diarization_region_floor(duration_seconds: float) -> int:
    """Minimum speaker regions a duration-``duration_seconds`` diarization
    must produce to be believable (card t_dc5c2abb, AC1 threshold shape:
    absolute floor scaled with duration with the small-audio exemption
    applied by the caller — see calibration block above)."""
    rate = _env_float("DIARIZATION_MIN_REGIONS_PER_MINUTE",
                      DIARIZATION_MIN_REGIONS_PER_MINUTE_DEFAULT)
    if rate < 0:
        rate = 0.0
    return max(2, int(duration_seconds / 60.0 * rate))


def _check_diarization_plausibility(
    regions: Sequence[Any] | int,
    duration_seconds: float | None,
    *,
    engine_signal: str | None = None,
    transcript_summary: str | None = None,
) -> None:
    """Refuse a collapsed speaker-region table; loud, typed, diagnostic.

    ``regions`` is the DIARIZER's speaker-region table (the split path's
    wrapper output; on the engine path a stand-in carrying the engine's
    own reported region count) — NOT the transcript segment count: the
    two are different quantities (472 segments came from 6 speakers on
    the clean 8411 s run). ``duration_seconds`` is the audio length;
    None/short audio falls under the small-audio exemption.

    Raises ``DiarizationDegradedError`` (maps to the NEW
    ``DIARIZATION_DEGRADED`` error code, AC4: distinct from the zero-label
    refusal and from OUT_OF_MEMORY even though OOM is the usual root
    cause). Never rejects a legitimate single-speaker recording below the
    check threshold (AC3)."""
    if duration_seconds is None or duration_seconds <= 0:
        return
    min_check = _env_float("DIARIZATION_MIN_CHECK_SECONDS",
                           DIARIZATION_MIN_CHECK_SECONDS_DEFAULT)
    if duration_seconds < min_check:
        # Small-audio exemption: a short recording with one or two
        # regions is exactly what a single-speaker upload looks like.
        return
    count = len(regions) if isinstance(regions, Sequence) else int(regions)
    floor = _diarization_region_floor(duration_seconds)
    if count >= floor:
        return
    message = (
        f"crispasr diarization degraded: {count} speaker regions for "
        f"{duration_seconds:.0f} s of audio "
        f"(floor {floor} regions at {duration_seconds / 60.0:.1f} min) — "
        "the global sherpa pre-pass failed and the engine's per-slice "
        "fallback collapsed the speakers; refusing to emit a "
        "collapsed-speaker transcript. Likely cause: the pre-pass was "
        "OOM-killed — raise the container memory limit (or lower "
        "concurrency), or use a shorter file"
    )
    if engine_signal:
        message += f". Engine signal: {engine_signal}"
    if transcript_summary:
        message += f". {transcript_summary}"
    raise DiarizationDegradedError(message)


def _reject_degraded_sherpa(stderr_path: str,
                            duration_seconds: float | None) -> None:
    """Engine-path degradation check from the captured stderr FILE (AC5).

    Source invariant (crispasr 0.8.43 diarize_cli, confirmed by reading
    the code): ``sherpa → N speaker regions over M ASR segments`` prints
    ONLY when the global pre-pass cache was invalid and the engine fell
    back to per-slice sherpa — the path that re-inits speaker identity
    every slice; the healthy path prints ``sherpa global → N`` instead.
    So on this path EITHER degradation line is conclusive for long
    audio, independent of the region count: N here is slice-local and
    small by construction. Short audio stays under the same
    small-audio-exemption as the region-table floor: a single-slice file
    degraded to per-slice is byte-for-byte the global result.
    """
    try:
        text = Path(stderr_path).read_text(errors="replace")
    except OSError:
        return  # stderr unreadable: nothing to inspect, never invent
    signal = _parse_sherpa_stderr(text)
    degraded = signal["global_failed"] is not None \
        or signal["fallback_regions"] is not None
    if not degraded:
        return
    min_check = _env_float("DIARIZATION_MIN_CHECK_SECONDS",
                           DIARIZATION_MIN_CHECK_SECONDS_DEFAULT)
    if not duration_seconds or duration_seconds < min_check:
        return
    regions = signal["fallback_regions"]
    parts = [
        (f"crispasr diarization degraded: the global sherpa pre-pass failed "
         f"and the engine fell back to per-slice diarization for "
         f"{duration_seconds:.0f} s of audio")
    ]
    if regions is not None:
        floor = _diarization_region_floor(duration_seconds)
        parts.append(
            f"last fallback line carried {regions} slice-local speaker "
            f"regions (a duration-scaled floor of {floor} whole-file "
            f"regions is the calibration from the measured 8411 s run: "
            f"472 segments clean vs 2 regions collapsed)")
    if signal["global_failed"]:
        parts.append(f"Engine signal: {signal['global_failed']}")
    parts.append(
        "refusing to emit a collapsed-speaker transcript. Likely cause: "
        "the pre-pass was OOM-killed (exit 137 = SIGKILL) — raise the "
        "container memory limit (or lower concurrency), or use a "
        "shorter file")
    raise DiarizationDegradedError(". ".join(parts))


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
                  duration_seconds: float | None = None) -> list[dict[str, Any]]:
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
    # Card t_dc5c2abb: the engine exited 0 — but did its GLOBAL sherpa
    # pre-pass silently degrade to the per-slice fallback on the way? The
    # fallback line prints only on that path (source-confirmed), so its
    # presence IS the degradation signal: the transcript is a fake with
    # per-slice speaker identities, whatever the region count says.
    # The duration-scaled region FLOOR is the split path's own detection
    # (there the wrapper region table is the inspected quantity, not the
    # engine's line); on this path the signal check is strictly stronger
    # evidence than the floor, and the floor is what makes the message
    # diagnostic with the measured numbers.
    known_duration = (duration_seconds if duration_seconds and duration_seconds > 0
                      else _probe_duration_quiet(wav_path))
    _reject_degraded_sherpa(stderr_path, known_duration)
    segments = parse_crispasr_output(payload)
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
    final ``("diar", 1, 1)`` assembly report.
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
    if progress is not None:
        known = duration if duration and duration > 0 else _probe_duration_quiet(audio_path)
        chunk_total = _engine_chunk_total(known)
        if chunk_total == 0:
            # Duration unknown: keep the legacy coarse pass pair rather than
            # fabricating chunk boundaries.
            progress("asr", 1, 2)
    with tempfile.TemporaryDirectory(prefix="autodiarist-crispasr-") as work_dir:
        segments = _run_crispasr(str(audio_path), work_dir,
                                 progress=progress, chunk_total=chunk_total,
                                 cancel=cancel,
                                 duration_seconds=(duration
                                                   if duration and duration > 0
                                                   else None))
    if progress is not None:
        progress("diar", 1, 1)
    if cancel is not None and cancel.cancelled():
        raise JobCancelled(
            "cancellation accepted at the diarization boundary: "
            f"{cancel.token}")
    return segments


def transcribe_audio(audio_path: str, mock: bool = False) -> list[dict[str, Any]]:
    """Compatibility wrapper: returns [{'text', 't0', 't1', 'speaker'}, ...]."""
    return transcribe_and_diarize(audio_path, mock=mock)
