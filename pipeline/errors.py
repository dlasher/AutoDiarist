"""Machine-readable error codes for failed jobs (card t_1799d07f).

A failed job used to expose only a raw exception string. A client could
not tell "the GPU ran out of memory" from "the job hit the timeout
ceiling" from "the engine crashed" — and each class needs a different
response: lower concurrency, raise the timeout, re-encode the input, or
report a bug.

The vocabulary below is ADDITIVE: ``error`` and ``message`` keep their
existing content and semantics; ``error_code`` joins them on every
``failed`` job record and response. CANCELLED jobs are NOT failures and
carry no code (AC7) — cancel and failure stay distinct signals.

  OUT_OF_MEMORY     VRAM exhaustion. The measured deterministic N=4
                    signature on the 4 GB card
                    (``ggml_gallocr_reserve_n_impl: failed to allocate
                    CUDA0 buffer ...`` + ``nemotron: sched alloc encoder
                    graph failed``) and the ``VRAM_LIMIT_MIB`` sampler
                    kill. Action: lower ``MAX_PROCESSING_GLOBAL`` (3 is
                    the measured ceiling on that host).
  TIMEOUT           ``CRISPASR_TIMEOUT_SECONDS`` ceiling expired (or a
                    decode-phase probe timed out). Action: raise the
                    timeout or split the input.
  UNSUPPORTED_AUDIO The upload could not be decoded (ffmpeg/ffprobe
                    failure — corrupt file, unsupported codec). Action:
                    re-encode or reject client-side.
  INPUT_TOO_LARGE   The upload exceeded ``MAX_UPLOAD_BYTES`` (HTTP 413
                    text). Action: smaller input; identical bytes fail
                    identically.
  ENGINE_CRASHED    The engine ran and failed with no known signature
                    (non-zero exit, missing/unusable output). Action:
                    report a bug, include the ``error`` string.
  DIARIZATION_DEGRADED
                    The engine exited 0 but the speaker-diarization pass
                    degraded silently (card t_dc5c2abb: the global sherpa
                    pre-pass was OOM-killed and the per-slice fallback
                    collapsed the speakers — measured 2 regions over a
                    8411 s 7-speaker file). The service REFUSED to emit
                    the transcript. Action: raise the container memory
                    limit (or lower concurrency), or use a shorter file.
  CONFIG_ERROR      Deployment configuration is wrong — e.g. the loud
                    missing-sherpa-assets guard or a bad DIAR_METHOD.
                    Action: fix the env; retrying the same job cannot
                    help.
  INTERNAL          Honest catch-all for everything else (unexpected
                    exceptions, empty-transcript results). Action: read
                    the ``error`` string; classify the raw message
                    FIRST, this code is for when nothing else matches.

Every emitted code must appear in the README *Error codes* table; the
pairing is pinned by ``tests/pipeline/test_error_codes.py`` (AC6).
"""
from __future__ import annotations

import re
import subprocess

from pipeline.transcribe import (
    CrispasrConfigError,
    CrispasrError,
    DiarizationDegradedError,
    TranscodeError,
    TranscriptionOOM,
)

OUT_OF_MEMORY = "OUT_OF_MEMORY"
TIMEOUT = "TIMEOUT"
UNSUPPORTED_AUDIO = "UNSUPPORTED_AUDIO"
INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
ENGINE_CRASHED = "ENGINE_CRASHED"
CONFIG_ERROR = "CONFIG_ERROR"
DIARIZATION_DEGRADED = "DIARIZATION_DEGRADED"
INTERNAL = "INTERNAL"

VOCABULARY = frozenset({
    OUT_OF_MEMORY, TIMEOUT, UNSUPPORTED_AUDIO, INPUT_TOO_LARGE,
    ENGINE_CRASHED, CONFIG_ERROR, DIARIZATION_DEGRADED, INTERNAL,
})

# Signatures are matched case-insensitively against the RAW recorded
# string. The VRAM list leads with the exact measured crash text from the
# concurrency findings (3/4 of every N=4 wave on the 4 GB card) plus the
# sampler's own limit message; the rest are the ggml/CUDA alloc-failure
# phrasings around them.
_OOM_PATTERNS = re.compile(
    r"failed to allocate cuda|ggml_gallocr|sched alloc encoder graph failed"
    r"|out of memory|cuda(malloc|memcpy|free) .*failed|insufficient memory"
    r"|vram limit exceeded",
    re.IGNORECASE,
)
_TIMEOUT_PATTERNS = re.compile(r"timed out|timeout(?: [=>]\d|\b )", re.IGNORECASE)
# Card t_dc5c2abb: the silent-speaker-collapse refusal. Checked FIRST —
# its message mentions the OOM kill as the *cause*, but the failure the
# client must act on is the degraded diarization (the transcript), not a
# bare memory crash: the remedy is memory limit/shorter file for the
# diarization specifically, and callers branch on the distinct code.
_DEGRADED_PATTERNS = re.compile(
    r"diarization degraded|diarization produced \d+ speaker regions"
    r"|\d+ speaker regions for [\d.]+ s of audio"
    r"|sherpa pre-pass failed and the engine fell back",
    re.IGNORECASE,
)
# Upload-over-cap text: the POST 413 detail ("File exceeds maximum size of
# N bytes") and any HTTP 413 envelope that reached a job record.
_SIZE_PATTERNS = re.compile(
    r"exceeds maximum size|too large|request body exceeds|http 413",
    re.IGNORECASE,
)
# Text produced by the decode/probe phase on an undecodable upload. The
# TranscodeError type covers the normal path; these patterns catch the
# same failures when only the recorded string survives (status.json
# replay, journal, pre-card records classified on read).
_UNSUPPORTED_PATTERNS = re.compile(
    r"ffmpeg failed|ffprobe failed|invalid data found"
    r"|moov atom not found|does not contain any stream",
    re.IGNORECASE,
)
# The engine's own failure phrasings (exit code, unusable output), for
# string-only classification. Checked LAST, after the specific signatures.
_ENGINE_PATTERNS = re.compile(
    r"crispasr exited|cannot parse crispasr|crispasr json missing"
    r"|refusing silent fallback|crispasr --version failed",
    re.IGNORECASE,
)


def _text(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - classification must be total, see below
        return exc.__class__.__name__


def _classify_text(text: str) -> str | None:
    """Pattern rules; None when nothing specific matched."""
    if _DEGRADED_PATTERNS.search(text):
        # First: a degraded-diarization message may quote the OOM kill or
        # the pre-pass timeout as the CAUSE; the failure class the client
        # branches on is the diarization degradation itself.
        return DIARIZATION_DEGRADED
    if _OOM_PATTERNS.search(text):
        return OUT_OF_MEMORY
    if _TIMEOUT_PATTERNS.search(text):
        return TIMEOUT
    if _SIZE_PATTERNS.search(text):
        return INPUT_TOO_LARGE
    if _UNSUPPORTED_PATTERNS.search(text):
        return UNSUPPORTED_AUDIO
    if _ENGINE_PATTERNS.search(text):
        return ENGINE_CRASHED
    return None


def classify(exc: BaseException | None = None, message: str = "") -> str:
    """Return the failure code for an exception and/or its raw string.

    Rules are ordered and deterministic: typed facts first (the sampler's
    OOM, the config guard), measured-text signatures next (they upgrade a
    crash whose stderr tail carries a known signature), an honest
    catch-all last. Never raises; always returns a member of VOCABULARY.
    """
    text = message or _text(exc)
    if isinstance(exc, TranscriptionOOM):
        return OUT_OF_MEMORY
    if isinstance(exc, DiarizationDegradedError):
        # Typed first: the guard raised this exact refusal; never let a
        # trailing quoted OOM line steal the classification.
        return DIARIZATION_DEGRADED
    specific = _classify_text(text)
    if specific is not None:
        return specific
    if isinstance(exc, subprocess.TimeoutExpired):
        return TIMEOUT
    if isinstance(exc, CrispasrConfigError):
        # Deployment configuration, not a job failure — e.g. the loud
        # missing-sherpa-assets guard.
        return CONFIG_ERROR
    if isinstance(exc, TranscodeError):
        return UNSUPPORTED_AUDIO
    if isinstance(exc, CrispasrError):
        return ENGINE_CRASHED
    return INTERNAL


def classify_message(message: str) -> str:
    """classify() restricted to the raw recorded string (no exception).

    Total on arbitrary input: any string in, exactly one vocabulary code
    out. A bare word like "timeout" alone is deliberately NOT evidence of
    a timed-out job (the engine phrasing is "timed out after <N>s"); the
    pattern only fires in sentence context.
    """
    try:
        text = message if isinstance(message, str) else str(message)
    except Exception:  # noqa: BLE001 - totality beats fidelity here
        text = ""
    specific = _classify_text(text)
    return specific if specific is not None else INTERNAL
