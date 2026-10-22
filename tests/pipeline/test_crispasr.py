"""Unit tests for the CrispASR backend adapter (card t_4d5f7d08).

The subprocess boundary is mocked two ways:
* pure command-build / output-parse tests (no process spawn), and
* fake crispasr executables (tiny shell scripts) for invocation-level tests,
  following the same stub pattern as the fake-embeddings unit tests.
No NeMo, no GPU, and no network is required.
"""

import json
import re
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


def _stub_sherpa_models(tmp_path: Path, monkeypatch) -> None:
    """DIAR_METHOD=sherpa is the shipped default; the startup validation
    (t_5a287979 AC4) demands the two ONNX models on disk. Fake-engine
    tests stub them exactly like the GGUF stub above."""
    for env, fname in (
        ("SHERPA_SEGMENT_MODEL", "sherpa-pyannote-segmentation-3.0.onnx"),
        ("SHERPA_EMBEDDING_MODEL", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
    ):
        stub = tmp_path / fname
        stub.write_bytes(b"onnx-stub")
        monkeypatch.setenv(env, str(stub))


# ---------------------------------------------------------------------------
# AC: config surface — CRISPASR_BIN, GGUF_MODEL, DIAR_METHOD=foxnose default,
#     language en-US, Q4_K GGUF invocation shape
# ---------------------------------------------------------------------------

def test_build_command_defaults_to_sherpa_and_en_us(tmp_path):
    """t_5a287979: the shipped default is sherpa (foxnose's BIC counter
    collapses 5-speaker panel audio to one label; sherpa is the engine's
    'reliable diarization' path -- see engine --help)."""
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out_base")
    assert cmd[0] == "/opt/crispasr/crispasr"
    joined = " ".join(cmd)
    assert "--backend nemotron" in joined
    assert cmd[cmd.index("-m") + 1] == "/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
    assert cmd[cmd.index("-l") + 1] == "en-US"
    assert "--diarize" in cmd
    assert cmd[cmd.index("--diarize-method") + 1] == "sherpa"
    assert "--diarize-embedder" in cmd  # embedder auto for stable global IDs
    assert "--sherpa-segment-model" in cmd
    assert "--sherpa-embedding-model" in cmd
    assert cmd[cmd.index("--gpu-backend") + 1] == "cuda"  # no silent CPU fallback
    assert "--suppress-nst" in cmd
    assert "-oj" in cmd
    assert cmd[cmd.index("-of") + 1] == "/tmp/out_base"
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "/tmp/in.wav"


def test_build_command_foxnose_still_selectable(tmp_path, monkeypatch):
    """foxnose stays a supported, explicitly-selectable method (fast path
    on well-separated audio) -- t_5a287979 changed only the default."""
    monkeypatch.setenv("DIAR_METHOD", "foxnose")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out_base")
    assert cmd[cmd.index("--diarize-method") + 1] == "foxnose"
    assert "--sherpa-segment-model" not in cmd  # models only for sherpa paths


def test_build_command_sortformer_selectable():
    import os
    os.environ["DIAR_METHOD"] = "sortformer"
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out_base")
    assert cmd[cmd.index("--diarize-method") + 1] == "sortformer"
    assert "--diarize-model" in cmd  # sortformer needs its own auto model
    assert "--diarize-embedder" not in cmd


def test_build_command_auto_download_when_gguf_unset(monkeypatch):
    monkeypatch.delenv("GGUF_MODEL")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")
    assert cmd[cmd.index("-m") + 1] == "auto"
    assert "--auto-download" in cmd


def test_diar_method_none_skips_diarization(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")
    assert "--diarize" not in cmd


def test_unknown_diar_method_rejected(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "whisper")
    with pytest.raises(tr.CrispasrError):
        tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")


# ---------------------------------------------------------------------------
# AC5: failure with bad binary path raises typed error; no silent fallback
# ---------------------------------------------------------------------------

def test_missing_binary_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


def test_unset_binary_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CRISPASR_BIN")
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


def test_missing_gguf_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GGUF_MODEL", str(tmp_path / "absent.gguf"))
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


def test_bad_backend_flag_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("CRISPASR_GPU_BACKEND", "qcd")
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


def test_failing_binary_propagates_stderr(tmp_path, monkeypatch):
    bad = _fake_binary(tmp_path, "crispasr-bad", 'echo "boom: cannot open library" >&2\nexit 3')
    monkeypatch.setenv("CRISPASR_BIN", str(bad))
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))
    assert "boom" in str(exc.value)


def test_cpu_fallback_detected(tmp_path, monkeypatch):
    # Simulates the first-run trap: ggml backends cpu only -> pinned cuda
    # unavailable -> typed error instead of a silent 11x CPU fallback.
    fake = _fake_binary(
        tmp_path, "crispasr-cpuonly",
        'if [ "$1" = "--version" ]; then echo "ggml backends : cpu"; exit 0; fi\nexit 1',
    )
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


def test_missing_output_json_raises(tmp_path, monkeypatch):
    fake = _fake_binary(
        tmp_path, "crispasr-nojson",
        'if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi\nexit 0',
    )
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    with pytest.raises(tr.CrispasrError):
        tr.transcribe_and_diarize(str(tmp_path / "a.wav"))


# ---------------------------------------------------------------------------
# End-to-end adapter against a fake crispasr binary (subprocess boundary real,
# engine stubbed): AC1 schema, AC3 no nemo, AC5 no silent fallback
# ---------------------------------------------------------------------------

FULL_JSON = {
    "crispasr": {"backend": "nemotron", "model": "q4_k", "language": "en-us"},
    "transcription": [
        {"offsets": {"from": 0, "to": 28000}, "timestamps": {"from": "00:00:00,000", "to": "00:00:28,000"},
         "speaker": "(speaker 0) ", "chunk_id": 0,
         "text": "<en-US> Welcome to the briefing on contract ratings."},
        {"offsets": {"from": 28000, "to": 28400}, "timestamps": {"from": "00:00:28,000", "to": "00:00:28,400"},
         "speaker": "(speaker 1) ", "chunk_id": 1, "text": "clauses."},
        {"offsets": {"from": 28200, "to": 40000}, "timestamps": {"from": "00:00:28,200", "to": "00:00:40,000"},
         "speaker": "(speaker 1) ", "chunk_id": 1, "text": "Thanks, let us review the results."},
    ],
}


@pytest.fixture()
def fake_crispasr(tmp_path, monkeypatch):
    script = f"""
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
cat > "$OF.json" <<'EOF'
{json.dumps(FULL_JSON)}
EOF
"""
    fake = _fake_binary(tmp_path, "crispasr-ok", script)
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    gguf = tmp_path / "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
    gguf.write_bytes(b"ggmf")
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    _stub_sherpa_models(tmp_path, monkeypatch)
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF")
    return wav


def test_transcribe_and_diarize_maps_segments(fake_crispasr):
    segments = tr.transcribe_and_diarize(str(fake_crispasr))
    assert segments, "no silent fallback to empty"
    assert all({"text", "t0", "t1", "speaker"} <= set(s) for s in segments)
    assert all(isinstance(s["text"], str) and s["text"].strip() for s in segments)
    # language tokens stripped
    assert all("<" not in s["text"] for s in segments)
    # monotonic t0/t1 and t0 <= t1
    assert all(s["t1"] >= s["t0"] for s in segments)
    assert all(segments[i]["t0"] >= segments[i - 1]["t1"] for i in range(1, len(segments)))
    # Speaker N labels, first-appearance stable ordering
    assert all(re.fullmatch(r"Speaker \d+", s["speaker"]) for s in segments)
    assert segments[0]["speaker"] == "Speaker 1"
    # sub-2s orphan fragment merged into previous (spurious tail-word
    # speaker absorbed); the genuine >2s second turn keeps Speaker 2
    assert len(segments) == 2
    assert segments[0]["speaker"] == "Speaker 1"
    assert segments[1]["speaker"] == "Speaker 2"
    assert "clauses." in segments[0]["text"]
    assert segments[0]["t0"] == pytest.approx(0.0)
    assert segments[0]["t1"] == pytest.approx(28.4)
    assert segments[1]["t0"] >= 28.4


def test_progress_callback_invoked(fake_crispasr):
    calls = []
    tr.transcribe_and_diarize(str(fake_crispasr), progress=lambda *a: calls.append(a))
    assert calls, "progress events expected"
    assert calls[-1][1] == calls[-1][2]  # final event is complete


def test_transcribe_audio_wrapper(fake_crispasr):
    segments = tr.transcribe_audio(str(fake_crispasr))
    assert isinstance(segments, list) and segments[0]["speaker"] == "Speaker 1"


# ---------------------------------------------------------------------------
# parse_crispasr_output — pure unit coverage of the mapping rules
# ---------------------------------------------------------------------------

def _seg(frm, to, text, speaker=None):
    d = {"offsets": {"from": frm, "to": to}, "text": text}
    if speaker is not None:
        d["speaker"] = f"(speaker {speaker}) "
    return d


def test_parse_relabels_speakers_by_first_appearance():
    payload = {"transcription": [
        _seg(0, 5000, "one", 1),
        _seg(5000, 9000, "two", 1),
        _seg(9000, 15000, "three", 0),
        _seg(15000, 21000, "four", 1),
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert [s["speaker"] for s in segments] == ["Speaker 1"] * 2 + ["Speaker 2", "Speaker 1"]


def test_parse_merges_short_fragments_into_previous(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = {"transcription": [
        _seg(0, 10000, "Long turn text."),
        _seg(10000, 10300, "day.", 1),
        _seg(10300, 20000, "Next turn."),
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert len(segments) == 2
    assert segments[0]["text"] == "Long turn text. day."
    assert segments[0]["t1"] == pytest.approx(10.3)


def test_parse_clamps_non_monotonic(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = {"transcription": [
        _seg(0, 10000, "first"),
        _seg(9000, 20000, "overlap"),
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert segments[1]["t0"] >= segments[0]["t1"]
    assert segments[1]["t1"] >= segments[1]["t0"]


def test_parse_drops_empty_and_strips_lang_tags(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = {"transcription": [
        _seg(0, 1000, " "),
        _seg(1000, 5000, "<en-US> hello <de-DE>"),
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert [s["text"] for s in segments] == ["hello"]


def test_parse_sorts_unordered_input(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = {"transcription": [
        _seg(30000, 40000, "later"),
        _seg(0, 10000, "earlier"),
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert [s["text"] for s in segments] == ["earlier", "later"]


def test_parse_falls_back_to_timestamps(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "none")
    payload = {"transcription": [
        {"timestamps": {"from": "00:01:30,500", "to": "01:00:00,000"}, "text": "deep"},
    ]}
    segments = tr.parse_crispasr_output(payload)
    assert segments[0]["t0"] == pytest.approx(90.5)
    assert segments[0]["t1"] == pytest.approx(3600.0)


def test_parse_empty_transcription_raises():
    with pytest.raises(tr.CrispasrError):
        tr.parse_crispasr_output({"transcription": []})


# ---------------------------------------------------------------------------
# AC: VRAM instrumentation carried over (peak logging + typed OOM)
# ---------------------------------------------------------------------------

def test_sampler_records_peak(monkeypatch):
    reads = iter([512.0, 1024.0, 900.0])
    monkeypatch.setattr(tr, "_nvidia_smi_used_mib", lambda: next(reads))
    sampler = tr.VramSampler()
    for _ in range(3):
        sampler.collect_once()
    assert sampler.peak_mib == 1024.0


def test_vram_oom_when_peak_exceeds_limit(fake_crispasr, monkeypatch):
    monkeypatch.setenv("VRAM_LIMIT_MIB", "1")  # any real usage exceeds
    fake = tr.VramSampler()
    fake.observe(9999.0)
    monkeypatch.setattr(tr, "_make_vram_sampler", lambda: fake)
    with pytest.raises(tr.TranscriptionOOM):
        tr.transcribe_and_diarize(str(fake_crispasr))


def test_vram_peak_logged(fake_crispasr, caplog, monkeypatch):
    import logging
    fake = tr.VramSampler()
    fake.observe(1085.0)
    monkeypatch.setattr(tr, "_make_vram_sampler", lambda: fake)
    with caplog.at_level(logging.INFO, logger="pipeline.transcribe"):
        tr.transcribe_and_diarize(str(fake_crispasr))
    assert "VRAM" in caplog.text and "peak=1085" in caplog.text


# ---------------------------------------------------------------------------
# AC6: no nemo import remains on the real path; AC7: MOCK_PIPELINE unchanged
# ---------------------------------------------------------------------------

def test_no_nemo_import_on_real_path():
    source = Path(tr.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+nemo[\w.]*", source, re.M), "nemo import remains"
    assert "import torch" not in source, "torch import remains on real path"


def test_mock_path_unchanged(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("real backend invoked in mock mode")
    monkeypatch.setattr(tr, "_run_crispasr", _boom)
    segments = tr.transcribe_and_diarize("dummy.wav", mock=True)
    assert len(segments) == 2
    assert segments[0]["speaker"] == "Speaker 1"
    import os
    os.environ["MOCK_PIPELINE"] = "1"
    assert tr.transcribe_and_diarize("dummy.wav") == segments


# ---------------------------------------------------------------------------
# Pipeline stamp support (card t_3275e056): probe captures the engine version;
# backend_identity() renders the label used by pipeline/renderer.py.
# ---------------------------------------------------------------------------

def test_probe_captures_version_for_backend_identity(tmp_path, monkeypatch):
    fake = _fake_binary(
        tmp_path, "crispasr-ver",
        'if [ "$1" = "--version" ]; then printf '
        '"crispasr v0.8.43 (ggml build)\\nggml backends : cpu,cuda\\n"; '
        'exit 0; fi\nexit 1',
    )
    monkeypatch.setenv("CRISPASR_BIN", str(fake))

    probe = tr.probe_crispasr_binary()

    assert probe["version"] == "v0.8.43"
    assert tr.crispasr_version() == "v0.8.43"
    assert tr.backend_identity() == "CrispASR nemotron Q4_K @v0.8.43"


def test_backend_identity_unknown_when_version_unparseable(tmp_path, monkeypatch):
    fake = _fake_binary(
        tmp_path, "crispasr-nover",
        'if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; '
        'exit 0; fi\nexit 1',
    )
    monkeypatch.setenv("CRISPASR_BIN", str(fake))

    tr.probe_crispasr_binary()

    assert tr.crispasr_version() is None
    assert tr.backend_identity() == "CrispASR nemotron Q4_K @unknown"
