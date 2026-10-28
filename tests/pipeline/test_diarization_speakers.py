"""Multi-speaker diarization regression (card t_5a287979).

Root cause established on the NV host (engine runs direct, bypassing the
service): the foxnose global pass with its default BIC counter collapses
the 249 s NASA SEWP panel file (5 real speakers) to one label --
`crispasr[diarize]: foxnose windows=357 -> 1 speakers
(cosine_similarity_single_speaker)`. The engine's own --help names
"sherpa/ecapa: external sherpa-onnx subprocess with segmentation +
speaker embedding + clustering" as the reliable path; those methods were
unusable on the shipped image because the two ONNX models and the
sherpa-onnx binary are absent -- and worse, the failure was SILENT: the
engine emitted segments without speaker labels and the pipeline
normalized them all to "Speaker 1" (a successful-looking single-speaker
transcript).

These tests pin the fix:

* AC4 (startup reject): `DIAR_METHOD=sherpa|ecapa` without the required
  model files on disk raises CrispasrConfigError BEFORE any job runs,
  instead of degrading silently at runtime.
* command wiring: the sherpa models/binary paths are passed to the
  engine as `--sherpa-segment-model` / `--sherpa-embedding-model` /
  `--sherpa-bin`.
* AC4 (runtime reject): engine output that carries NO speaker labels for
  a labeling method is a hard error -- it never becomes a fake
  single-speaker transcript. A true single-speaker result (one label,
  present) must still map to exactly one speaker (AC7, no invented
  speakers).
* AC1 (engine integration, GPU host only): the 249 s panel fixture yields
  >= 3 distinct speakers via the service pipeline, and the pyannote
  2-speaker control keeps yielding >= 2. Runs with the real engine when
  `CRISPASR_BIN` + fixture env vars are present (inside the container on
  the NV host); skipped in CI without them.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline import transcribe as tr

CONFIG_ENVS = (
    "CRISPASR_BIN", "GGUF_MODEL", "DIAR_METHOD", "CRISPASR_LANGUAGE",
    "CRISPASR_GPU_BACKEND", "CRISPASR_CACHE_DIR", "CRISPASR_LIB_DIR",
    "CRISPASR_TIMEOUT_SECONDS", "MIN_FRAGMENT_SECONDS", "VRAM_LIMIT_MIB",
    "MOCK_PIPELINE", "CHUNK_SECONDS", "OVERLAP_SECONDS",
    "SHERPA_SEGMENT_MODEL", "SHERPA_EMBEDDING_MODEL", "SHERPA_BIN",
)

# Canonical model artifacts (k2-fsa/sherpa-onnx GitHub release assets;
# the Dockerfile pins them by sha256 -- see README "Diarization models").
# The segmentation model is reverb-trained: on NV-host A/Bs it halves the
# single-speaker over-split vs plain pyannote-segmentation-3-0 while the
# panel file keeps 5 speakers at thr 0.35.
SHERPA_SEGMENT_NAME = "sherpa-onnx-reverb-diarization-v1.onnx"
SHERPA_EMBEDDING_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in CONFIG_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRISPASR_BIN", "/opt/crispasr/crispasr")
    monkeypatch.setenv("GGUF_MODEL", "/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf")
    tr._BACKEND_PROBE_CACHE.clear()
    yield


def _fake_binary(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model-stub")
    return str(path)


# ---------------------------------------------------------------------------
# AC4: startup validation rejects silently-degrading methods
# ---------------------------------------------------------------------------

def _stub_gguf_and_backend(monkeypatch, tmp_path):
    """Make probe/gguf checks pass so only the sherpa validation can fail."""
    monkeypatch.setattr(tr, "probe_crispasr_binary", lambda *a, **k: {"backends": ["cuda"], "path": "/x"})
    gguf = _touch(tmp_path / "model.gguf")
    monkeypatch.setenv("GGUF_MODEL", gguf)


def test_sherpa_without_models_rejected_at_startup(tmp_path, monkeypatch):
    _stub_gguf_and_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    # The models are NOT on disk -- today the config is accepted and the job
    # silently degrades to speaker=None -> "Speaker 1" (the exact defect).
    with pytest.raises(tr.CrispasrConfigError):
        tr._validate_runtime_config()


def test_ecapa_without_models_rejected_at_startup(tmp_path, monkeypatch):
    _stub_gguf_and_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("DIAR_METHOD", "ecapa")
    with pytest.raises(tr.CrispasrConfigError):
        tr._validate_runtime_config()


def test_sherpa_with_models_present_accepts(tmp_path, monkeypatch):
    _stub_gguf_and_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", _touch(tmp_path / SHERPA_SEGMENT_NAME))
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", _touch(tmp_path / SHERPA_EMBEDDING_NAME))
    tr._validate_runtime_config()  # no raise


def test_sherpa_rejects_non_executable_sherpa_bin(tmp_path, monkeypatch):
    """AC4 refusal, second half: SHERPA_BIN pointing at a non-executable
    file is a miswired wrapper -- fail at startup, not per job."""
    _stub_gguf_and_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", _touch(tmp_path / SHERPA_SEGMENT_NAME))
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", _touch(tmp_path / SHERPA_EMBEDDING_NAME))
    dead = tmp_path / "sherpa-diarization-wrapper"
    dead.write_text("#!/bin/sh\nexit 0\n")  # NOT chmod +x
    monkeypatch.setenv("SHERPA_BIN", str(dead))
    with pytest.raises(tr.CrispasrConfigError, match="SHERPA_BIN"):
        tr._validate_runtime_config()


# ---------------------------------------------------------------------------
# command wiring: the model paths reach the engine
# ---------------------------------------------------------------------------

def test_build_command_wires_sherpa_models(tmp_path, monkeypatch):
    seg = _touch(tmp_path / SHERPA_SEGMENT_NAME)
    emb = _touch(tmp_path / SHERPA_EMBEDDING_NAME)
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", seg)
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", emb)
    cmd = tr.build_crispasr_command("/tmp/a.wav", "/tmp/out")
    assert "--sherpa-segment-model" in cmd and seg in cmd
    assert "--sherpa-embedding-model" in cmd and emb in cmd


def test_build_command_wires_sherpa_bin_wrapper(tmp_path, monkeypatch):
    """Card t_7685c6da supersedes the pre-split contract.

    The wrapper no longer reaches the engine via --sherpa-bin: the split
    runs the engine ASR-only and the pipeline spawns SHERPA_BIN itself
    (build_sherpa_diarize_command), which is what restores live -pp
    progress on long files. The wiring requirement survives, in the
    second invocation's argv.
    """
    seg = _touch(tmp_path / SHERPA_SEGMENT_NAME)
    emb = _touch(tmp_path / SHERPA_EMBEDDING_NAME)
    wrapper = _fake_binary(tmp_path, "sherpa-wrapper", "exit 0")
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", seg)
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", emb)
    monkeypatch.setenv("SHERPA_BIN", str(wrapper))
    cmd = tr.build_crispasr_command("/tmp/a.wav", "/tmp/out")
    assert "--sherpa-bin" not in cmd and "--diarize" not in cmd
    diar_cmd = tr.build_sherpa_diarize_command("/tmp/full16k.wav")
    assert diar_cmd[0] == str(wrapper)
    assert seg in " ".join(diar_cmd) and emb in " ".join(diar_cmd)


def test_shipped_default_is_multi_speaker_capable_method():
    """The shipped defaults must not be the collapsing foxnose path.

    The pipeline default (DIAR_METHOD_DEFAULT) AND both backend compose
    files must select the same multi-speaker-capable method (AC4/AC5/AC6):
    foxnose collapses 5 real speakers to 1 on panel audio; the engine's
    own help routes reliable diarization to sherpa/ecapa.
    """
    assert tr.DIAR_METHOD_DEFAULT == "sherpa"
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "DIAR_METHOD: sherpa" in compose


# ---------------------------------------------------------------------------
# AC4 runtime guard: unlabeled output for a labeling method is an error.
# parse_crispasr_output normalizes internally using the configured
# DIAR_METHOD, so the guard fires through the parse entry point.
# ---------------------------------------------------------------------------

def _payload(entries):
    return {"transcription": entries}


def _seg(t0, t1, text, speaker=None):
    entry = {
        "timestamps": {"from": t0, "to": t1},
        "offsets": {"from": t0, "to": t1},
        "text": text,
    }
    if speaker is not None:
        entry["speaker"] = speaker
    return entry


def test_unlabeled_segments_raise_for_labeling_method(monkeypatch):
    """The old sherpa failure shape: engine succeeds, no labels anywhere."""
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    payload = _payload([_seg("00:00:01,000", "00:00:05,000", "hello there"),
                        _seg("00:00:05,000", "00:00:09,000", "and more")])
    with pytest.raises(tr.CrispasrError):
        tr.parse_crispasr_output(payload)


def test_none_method_keeps_single_unlabeled_output(monkeypatch):
    """DIAR_METHOD=none remains the documented no-diarization escape hatch."""
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = _payload([_seg("00:00:01,000", "00:00:05,000", "hello there")])
    out = tr.parse_crispasr_output(payload)
    assert {s["speaker"] for s in out} == {"Speaker 1"}


def test_true_single_speaker_maps_to_exactly_one_label(monkeypatch):
    """AC7 guard: a PRESENT single label is one speaker, never invented extras."""
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    payload = _payload([
        _seg("00:00:01,000", "00:00:05,000", "one", "(speaker 0) "),
        _seg("00:00:05,000", "00:00:09,000", "two", "(speaker 0) "),
    ])
    out = tr.parse_crispasr_output(payload)
    assert {s["speaker"] for s in out} == {"Speaker 1"}


def test_multi_label_output_maps_to_multi_speakers(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    payload = _payload([
        _seg("00:00:01,000", "00:00:05,000", "one", "(speaker 0) "),
        _seg("00:00:05,000", "00:00:09,000", "two", "(speaker 1) "),
        _seg("00:00:09,000", "00:00:13,000", "three", "(speaker 0) "),
    ])
    out = tr.parse_crispasr_output(payload)
    assert {s["speaker"] for s in out} == {"Speaker 1", "Speaker 2"}


# ---------------------------------------------------------------------------
# AC1/AC2/AC6: real-engine integration on the GPU host (skipped in CI)
# ---------------------------------------------------------------------------

ENGINE_AVAILABLE = shutil.which(os.environ.get("CRISPASR_BIN", "/opt/crispasr/crispasr")) is not None \
    or Path(os.environ.get("CRISPASR_BIN", "/opt/crispasr/crispasr")).is_file()

PANEL_WAV = os.environ.get("DIAR_TEST_PANEL_WAV", "/tmp/panel_ref.wav")
CONTROL_WAV = os.environ.get("DIAR_TEST_MARKLEX_WAV", "/tmp/marklex1min.wav")

requires_engine = pytest.mark.skipif(
    not (ENGINE_AVAILABLE and Path(PANEL_WAV).is_file()),
    reason="real engine + 249s panel fixture required (run inside the image on the NV host)",
)


def _distinct_service_speakers(wav: Path, tmp: Path) -> set[str]:
    """Run the exact shipped command build -> engine -> parse (+normalize)."""
    out_base = tmp / "out"
    cmd = tr.build_crispasr_command(str(wav), str(out_base))
    subprocess.run(cmd, check=True, capture_output=True, timeout=1200, env=tr._crispasr_env())
    payload = json.loads(Path(f"{out_base}.json").read_text(encoding="utf-8"))
    return {s["speaker"] for s in tr.parse_crispasr_output(payload)}


@requires_engine
def test_panel_file_yields_three_or_more_speakers(tmp_path, monkeypatch):
    """AC2: engine + chosen default method on the 249 s panel file -> >= 3.

    Fails on current main (foxnose default: one 'Speaker 1' for 5 people);
    passes once the sherpa method + models ship. Run with the real binary:
    DIAR_TEST_PANEL_WAV=/tmp/panel_ref.wav pytest tests/pipeline/test_diarization_speakers.py -k panel
    """
    monkeypatch.setenv("CRISPASR_BIN", os.environ.get("CRISPASR_BIN", "/opt/crispasr/crispasr"))
    monkeypatch.setenv("GGUF_MODEL", os.environ.get("GGUF_MODEL", "/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"))
    monkeypatch.setenv("CRISPASR_GPU_BACKEND", os.environ.get("CRISPASR_GPU_BACKEND", "cuda"))
    speakers = _distinct_service_speakers(Path(PANEL_WAV), tmp_path)
    assert len(speakers) >= 3, f"panel file collapsed to {sorted(speakers)}"


@requires_engine
@pytest.mark.skipif(not Path(CONTROL_WAV).is_file(), reason="marklex control fixture absent")
def test_marklex_control_keeps_two_speakers(tmp_path, monkeypatch):
    """Control: pyannote's published 2-speaker file must keep separating 2."""
    monkeypatch.setenv("CRISPASR_BIN", os.environ.get("CRISPASR_BIN", "/opt/crispasr/crispasr"))
    monkeypatch.setenv("GGUF_MODEL", os.environ.get("GGUF_MODEL", "/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"))
    monkeypatch.setenv("CRISPASR_GPU_BACKEND", os.environ.get("CRISPASR_GPU_BACKEND", "cuda"))
    speakers = _distinct_service_speakers(Path(CONTROL_WAV), tmp_path)
    assert len(speakers) >= 2, f"marklex control collapsed to {sorted(speakers)}"
