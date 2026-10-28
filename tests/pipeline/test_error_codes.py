"""Machine-readable job error codes (card t_1799d07f) — classifier tests.

AC2/AC3: the code vocabulary covers the failure classes this project
actually produces, classification runs against the RAW recorded strings
of the measured failures, and it is total on arbitrary input — never
raises, never emits an undocumented code. AC5/AC6: the emitted set is
pinned against the README documentation.
"""
import re
import subprocess
from pathlib import Path

import pytest

from pipeline import errors
from pipeline.transcribe import (
    CrispasrConfigError,
    CrispasrError,
    TranscodeError,
    TranscriptionOOM,
)

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The measured failures, verbatim (card AC3: classification runs against
# real strings, never invented examples).
# ---------------------------------------------------------------------------

# N=4 on the 4 GB NVIDIA card, 3/4 of every wave (concurrency findings).
MEASURED_VRAM_CRASH = (
    "crispasr exited with code 1: ggml_gallocr_reserve_n_impl: failed to "
    "allocate CUDA0 buffer of 647530880 bytes\nnemotron: sched alloc "
    "encoder graph failed"
)
MEASURED_TIMEOUT = "crispasr timed out after 1800s: ['/usr/local/bin/crispasr']"
MEASURED_EMPTY = "Empty transcript result."
MEASURED_413_BODY = "File exceeds maximum size of 524288000 bytes"
MEASURED_FFMPEG_FAILURE = (
    "ffmpeg failed: [mp3 @ 0x55f7b0] Invalid data found when processing input"
)
# Verbatim shape of _require_sherpa_artifacts' guard (pipeline/transcribe.py).
MEASURED_MISSING_ASSETS = (
    "DIAR_METHOD='sherpa' requires the sherpa-onnx models; missing: "
    "SHERPA_SEGMENT_MODEL='/opt/models/sherpa-onnx-reverb-diarization-v1.onnx' "
    "— add them to the image (see README 'Diarization models') or choose "
    "another DIAR_METHOD"
)


# --- AC2: every named class maps from its real string ----------------------


def test_vram_exhaustion_maps_to_out_of_memory():
    assert errors.classify_message(MEASURED_VRAM_CRASH) == errors.OUT_OF_MEMORY
    # same signature arriving as a generic exception object (engine crash
    # path wraps the stderr tail in CrispasrError)
    assert errors.classify(CrispasrError(MEASURED_VRAM_CRASH)) == errors.OUT_OF_MEMORY
    # and as the sampler's own typed OOM
    assert errors.classify(TranscriptionOOM(MEASURED_VRAM_CRASH)) == errors.OUT_OF_MEMORY


def test_timeout_ceiling_maps_to_timeout():
    assert errors.classify_message(MEASURED_TIMEOUT) == errors.TIMEOUT
    assert errors.classify(CrispasrError(MEASURED_TIMEOUT)) == errors.TIMEOUT
    exc = subprocess.TimeoutExpired(cmd=["crispasr"], timeout=1800)
    assert errors.classify(exc) == errors.TIMEOUT
    # the decode-phase probe timeout is a timeout too, despite the type
    assert errors.classify(TranscodeError("ffprobe timed out")) == errors.TIMEOUT


def test_unsupported_audio_maps_from_the_decode_failures():
    assert (errors.classify(TranscodeError(MEASURED_FFMPEG_FAILURE))
            == errors.UNSUPPORTED_AUDIO)
    assert errors.classify(TranscodeError(
        "ffprobe failed: moov atom not found")) == errors.UNSUPPORTED_AUDIO
    assert errors.classify_message(MEASURED_FFMPEG_FAILURE) == errors.UNSUPPORTED_AUDIO


def test_input_too_large_maps_from_the_measured_413():
    assert errors.classify_message(MEASURED_413_BODY) == errors.INPUT_TOO_LARGE
    from fastapi import HTTPException
    assert errors.classify(HTTPException(
        status_code=413, detail=MEASURED_413_BODY)) == errors.INPUT_TOO_LARGE


def test_engine_crashed_maps_from_engine_failures():
    # exit != 0 with a tail carrying NO specific signature stays a crash
    assert (errors.classify(CrispasrError(
        "crispasr exited with code 139: segmentation fault in decoder"))
        == errors.ENGINE_CRASHED)
    assert errors.classify(CrispasrError(
        "crispasr exited 0 but wrote no 'crispasr_out.json'; refusing silent fallback"
    )) == errors.ENGINE_CRASHED
    assert errors.classify_message(
        "crispasr JSON missing 'transcription' list") == errors.ENGINE_CRASHED


def test_config_error_maps_from_the_loud_asset_guard():
    assert errors.classify(CrispasrConfigError(MEASURED_MISSING_ASSETS)) \
        == errors.CONFIG_ERROR
    assert errors.classify(CrispasrConfigError(
        "DIAR_METHOD='fox' not supported; choose from sherpa")) == errors.CONFIG_ERROR


def test_internal_is_the_honest_catch_all():
    assert errors.classify(RuntimeError("disk exploded")) == errors.INTERNAL
    assert errors.classify(OSError(28, "No space left on device")) == errors.INTERNAL
    # "Empty transcript result." names no class: catch-all, not a guess.
    assert errors.classify_message(MEASURED_EMPTY) == errors.INTERNAL


# --- AC3: classification is total on arbitrary strings ---------------------

_JUNK = [
    "", " ", "\x00\x01binary", "null", "None",
    "OUT_OF_MEMORY", "TIMEOUT", "timeout", "out of memory",
    "a" * 10_000, "unicode ✓ émoji 💥 failed to allocate",
    "42", "{}", "[]", "\n\n\n",
]


@pytest.mark.parametrize("junk", _JUNK)
def test_classify_message_total_never_raises(junk):
    # The hard guarantees: no exception, always a documented code. (The
    # literal phrases above legitimately match their class by pattern —
    # they are specific enough to be evidence.)
    assert errors.classify_message(junk) in errors.VOCABULARY


def test_classify_accepts_bare_exception_and_none():
    assert errors.classify(None) == errors.INTERNAL
    assert errors.classify(ValueError()) == errors.INTERNAL


def test_classify_survives_exception_with_broken_str():
    class Broken(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    assert errors.classify(Broken()) in errors.VOCABULARY


def test_codes_are_stable_uppercase_names():
    # machine-readable contract: the value IS the name, documented as such
    for code in errors.VOCABULARY:
        assert code == code.upper()
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", code)


# --- AC5/AC6: documentation paired with the code ---------------------------


def _readme_documented_codes():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    section = text.split("#### Error codes", 1)[1].split("\n#### ", 1)[0]
    return {m.group(1) for m in re.finditer(r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|", section,
                                            re.MULTILINE)}


def test_every_emitted_code_is_documented_in_readme():
    # AC6: any code the implementation can emit (VOCABULARY is exactly the
    # return set of classify/classify_message) is pinned against the
    # README table — code drift fails the test, doc drift fails the test.
    documented = _readme_documented_codes()
    assert documented == set(errors.VOCABULARY), (
        f"undocumented codes: {sorted(errors.VOCABULARY - documented)}, "
        f"undeliverable docs: {sorted(documented - errors.VOCABULARY)}")


def test_module_docstring_documents_the_vocabulary():
    doc = errors.__doc__ or ""
    for code in errors.VOCABULARY:
        assert code in doc, f"{code} missing from the pipeline/errors.py doc block"
