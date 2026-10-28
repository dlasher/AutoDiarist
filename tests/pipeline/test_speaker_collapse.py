"""Silent speaker collapse (card t_dc5c2abb): the loud-failure guard tests.

The measured defect (NV host, 8411.072 s SEWP panel file, 08:07 control
run): the engine's GLOBAL sherpa pre-pass was OOM-killed (exit 137 =
SIGKILL), the engine fell back to per-slice sherpa SILENTLY, and the run
"successfully" produced a transcript with 2 speaker regions over 1 ASR
segment. The existing zero-label guard in _normalize_speakers does not
fire: 2 regions is not zero, so the service returns a collapsed-speaker
transcript and nothing reports an error.

Calibration (measured counts from the card body + planner receipt):
    249 s panel    -> 18 segments, 3 speakers   (legit)
    137 s control  ->  1 segment,  1 speaker    (legit — a 1-speaker
                                                  recording is NOT a
                                                  defect; a bare
                                                  rate floor rejects
                                                  this: 0.44/min)
     27 s clip     ->  4 segments, 2 speakers   (legit)
   8411 s clean    -> 472 segments, 6 speakers  (legit, 3.37 seg/min)
   8411 s DEFECT   ->   2 regions over 1 segment (0.0143 regions/min)
The gap between legit and defect on the long file is a factor of 235.

The guard therefore refuses ONLY when the audio is long enough that a
tiny region count cannot be a legitimate recording (small-audio
exemption) AND the region count is below a floor that scales with
duration. Constants are pinned here from the measurements above.
"""
import json
from pathlib import Path

import pytest

from pipeline import transcribe as tr

# The engine's own degradation lines, captured verbatim from the control
# run's stderr (card body): used for the AC5 parser tests.
MEASURED_ENGINE_STDERR = """\
crispasr: processing 306 slice(s)
crispasr[diarize]: computing global sherpa timeline over 134577152 samples (8411.1 s)...
crispasr[diarize]: '/opt/sherpa/sherpa-diarization-wrapper' '--clustering.num-clusters=0' '--segmentation.pyannote-model=/opt/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx' '--embedding.model=/opt/models/sherpa-onnx-eres2net_base.onnx' '/tmp/tmpsherpa.wav'
crispasr[diarize]: sherpa global run failed with exit code 137
crispasr[diarize]: '/opt/sherpa/sherpa-diarization-wrapper' '--clustering.num-clusters=0' '--segmentation.pyannote-model=/opt/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx' '--embedding.model=/opt/models/sherpa-onnx-eres2net_base.onnx' '/tmp/tmpslice.wav'
crispasr[diarize]: sherpa \u2192 2 speaker regions over 1 ASR segments
crispasr: progress =   0% (1/306 slices)
"""


# ---------------------------------------------------------------------------
# AC1/AC3 — the plausibility guard itself (unit level, no GPU, no engine)
# ---------------------------------------------------------------------------

def test_long_audio_with_two_regions_is_refused():
    # 8411 s, 2 regions -> the OOM-fallback shape (the measured control
    # run). Must NOT be accepted. This is the RED on today's code: there
    # is no plausibility guard yet, only the zero-label check.
    with pytest.raises(tr.CrispasrError) as exc:
        tr._check_diarization_plausibility(
            regions=[{"t0": 0.0, "t1": 100.0, "speaker": 0},
                     {"t0": 500.0, "t1": 600.0, "speaker": 1}],
            duration_seconds=8411.072)
    msg = str(exc.value)
    assert "region" in msg.lower()


def test_short_single_speaker_audio_is_accepted():
    # 137 s, 1 region -> a legitimate single-speaker recording. MUST NOT
    # raise. A naive rate floor (0.44 regions/min) gets this WRONG; the
    # small-audio exemption is what saves it (card AC3, "Do NOT").
    tr._check_diarization_plausibility(
        regions=[{"t0": 0.0, "t1": 137.0, "speaker": 0}],
        duration_seconds=137.45)


def test_measured_panel_is_accepted():
    # 249 s, 3 regions -> the measured panel calibration (18 segments,
    # 3 speakers). Must not raise.
    tr._check_diarization_plausibility(
        regions=[{"t0": 0.0, "t1": 80.0, "speaker": 0},
                 {"t0": 90.0, "t1": 170.0, "speaker": 1},
                 {"t0": 180.0, "t1": 249.0, "speaker": 2}],
        duration_seconds=249.003)


def test_clean_long_run_is_accepted():
    # The planner's measured clean run: 472 regions over 8411.072 s
    # (3.37/min). Far above the floor; must pass.
    regions = [{"t0": float(i), "t1": float(i) + 0.5,
                "speaker": i % 6} for i in range(472)]
    tr._check_diarization_plausibility(regions, duration_seconds=8411.072)


def test_guard_message_is_diagnostic():
    # AC4: the refusal names the degradation, the measured numbers
    # (region count, audio duration) and the remedy (container memory
    # limit / shorter file). A client must not need container logs.
    with pytest.raises(tr.CrispasrError) as exc:
        tr._check_diarization_plausibility(
            [{"t0": 0.0, "t1": 1.0, "speaker": 0},
             {"t0": 2.0, "t1": 3.0, "speaker": 1}],
            duration_seconds=8411.072)
    msg = str(exc.value)
    assert "2" in msg                      # region count
    assert "8411" in msg                   # audio duration
    assert "memory" in msg.lower()         # the likely cause/remedy
    assert "shorter" in msg.lower()        # the alternative remedy


# ---------------------------------------------------------------------------
# AC5 — the engine's own degradation signal is captured from stderr
# ---------------------------------------------------------------------------

def test_parse_sherpa_stderr_global_failure_line():
    # The measured OOM-killed pre-pass line, verbatim (card body line 18).
    signal = tr._parse_sherpa_stderr(MEASURED_ENGINE_STDERR)
    assert signal["global_failed"] is not None
    assert "137" in signal["global_failed"]


def test_parse_sherpa_stderr_region_fallback_line():
    # 'crispasr[diarize]: sherpa -> N speaker regions over M ASR segments'
    # is printed ONLY on the per-slice fallback path (source-confirmed:
    # crispasr_diarize_cli.cpp prints 'sherpa global -> N speaker regions'
    # on the global success path instead). The parser must capture both.
    signal = tr._parse_sherpa_stderr(MEASURED_ENGINE_STDERR)
    assert signal["fallback_regions"] == 2
    assert signal["fallback_asr_segments"] == 1


def test_parse_sherpa_stderr_healthy_run_has_no_signal():
    # The global success path prints 'sherpa global -> 6 speaker regions'
    # and NO failure line; the parser must not invent degradation.
    healthy = (
        "crispasr[diarize]: computing global sherpa timeline over "
        "4464000 samples (279.0 s)...\n"
        "crispasr[diarize]: sherpa global \u2192 6 speaker regions\n"
        "crispasr: progress = 100% (306/306 slices)\n"
    )
    signal = tr._parse_sherpa_stderr(healthy)
    assert signal["global_failed"] is None
    assert signal["fallback_regions"] is None
    assert signal["global_regions"] == 6


def test_parse_sherpa_stderr_survives_junk():
    # Totality: arbitrary stderr text never raises.
    for junk in ("", "x", "\x00\x01", "sherpa global run failed",
                 "crispasr[diarize]: sherpa \u2192 NaN speaker regions"):
        signal = tr._parse_sherpa_stderr(junk)
        assert set(signal) >= {"global_failed", "fallback_regions",
                               "fallback_asr_segments", "global_regions"}


# ---------------------------------------------------------------------------
# AC1/AC5 end-to-end — fake engine replaying the measured degraded run
# ---------------------------------------------------------------------------

def _fake_binary(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _stub_sherpa_models(tmp_path: Path, monkeypatch) -> None:
    for env, fname in (
        ("SHERPA_SEGMENT_MODEL", "sherpa-seg.onnx"),
        ("SHERPA_EMBEDDING_MODEL", "sherpa-emb.onnx"),
    ):
        (tmp_path / fname).write_bytes(b"onnx")
        monkeypatch.setenv(env, str(tmp_path / fname))


@pytest.fixture()
def configured_env(tmp_path, monkeypatch):
    """Config env as in tests/pipeline/test_crispasr.py's clean_env."""
    for name in (
        "CRISPASR_BIN", "GGUF_MODEL", "DIAR_METHOD", "CRISPASR_LANGUAGE",
        "CRISPASR_GPU_BACKEND", "CRISPASR_CACHE_DIR", "CRISPASR_LIB_DIR",
        "CRISPASR_TIMEOUT_SECONDS", "MIN_FRAGMENT_SECONDS",
        "VRAM_LIMIT_MIB", "MOCK_PIPELINE", "CHUNK_SECONDS",
        "OVERLAP_SECONDS", "SHERPA_SEGMENT_MODEL", "SHERPA_EMBEDDING_MODEL",
        "SHERPA_BIN", "MAX_PROCESSING_GLOBAL",
        "DIARIZATION_MIN_REGIONS_PER_MINUTE",
        "DIARIZATION_MIN_CHECK_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    gguf = tmp_path / "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
    gguf.write_bytes(b"ggmf")
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    monkeypatch.setenv("CRISPASR_BIN", str(tmp_path / "absent"))  # overridden
    _stub_sherpa_models(tmp_path, monkeypatch)
    tr._BACKEND_PROBE_CACHE.clear()
    yield tmp_path


DEGRADED_JSON = {
    "transcription": [
        # '2 speaker regions over 1 ASR segments': one collapsed ASR
        # segment carrying a single label — the measured control shape.
        {"offsets": {"from": 0, "to": 841107},
         "timestamps": {"from": "00:00:00,000", "to": "02:20:11,072"},
         "speaker": "(speaker 0) ", "chunk_id": 0,
         "text": "collapsed panel transcript"}
    ],
}


def _engine_replay(tmp_path: Path, payload, stderr_text: str) -> Path:
    script = f'''
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
cat > "$OF.json" <<'EOF'
{json.dumps(payload)}
EOF
cat >&2 <<'ERR'
{stderr_text}
ERR
'''
    return _fake_binary(tmp_path, "crispasr-replay", script)


def test_degraded_engine_run_is_refused_end_to_end(configured_env, monkeypatch):
    # AC1 + AC5 wired: the engine exits 0 with a collapsed transcript and
    # prints its degradation on stderr. transcribe_and_diarize must RAISE
    # and NO transcript may be returned.
    tmp = configured_env
    fake = _engine_replay(tmp, DEGRADED_JSON, MEASURED_ENGINE_STDERR)
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp / "in.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(str(wav), duration=8411.072)
    msg = str(exc.value)
    assert "region" in msg.lower()
    assert "8411" in msg


def test_healthy_long_engine_run_completes_end_to_end(configured_env, monkeypatch):
    # AC3 e2e: same duration, but stderr shows the global pass succeeded
    # ('sherpa global -> 6 speaker regions') and the JSON carries six
    # speakers' worth of labels. Must return the transcript unchanged.
    tmp = configured_env
    healthy_stderr = (
        "crispasr: processing 306 slice(s)\n"
        "crispasr[diarize]: computing global sherpa timeline over "
        "134577152 samples (8411.1 s)...\n"
        "crispasr[diarize]: sherpa global \u2192 6 speaker regions\n"
        "crispasr: progress = 100% (306/306 slices)\n"
    )
    payload = {"transcription": [
        # 12 s per segment: above MIN_FRAGMENT_SECONDS (2.0) so the
        # fragment merge keeps all six distinct.
        {"offsets": {"from": i * 15000, "to": i * 15000 + 12000},
         "speaker": f"(speaker {i}) ", "chunk_id": i,
         "text": f"segment number {i}"}
        for i in range(6)
    ]}
    fake = _engine_replay(tmp, payload, healthy_stderr)
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp / "in.wav"
    wav.write_bytes(b"RIFF")
    segments = tr.transcribe_and_diarize(str(wav), duration=8411.072)
    assert len(segments) == 6
    assert segments[0]["speaker"] == "Speaker 1"


def test_zero_label_refusal_message_is_unchanged(configured_env, monkeypatch):
    # AC2: the existing guard keeps its exact message and still fires.
    tmp = configured_env
    payload = {"transcription": [
        {"offsets": {"from": 0, "to": 1000},
         "timestamps": {"from": "00:00:00,000", "to": "00:00:10,000"},
         "text": "no labels at all"}
    ]}
    fake = _engine_replay(tmp, payload, "crispasr: done\n")
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp / "in.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(str(wav), duration=10.0)
    assert "produced no speaker labels" in str(exc.value)


# ---------------------------------------------------------------------------
# AC4 — a NEW machine-readable error code, distinct from every existing one
# ---------------------------------------------------------------------------

def test_degraded_diarization_maps_to_new_error_code():
    from pipeline import errors
    assert "DIARIZATION_DEGRADED" in errors.VOCABULARY
    # Distinct: not a reuse of OUT_OF_MEMORY even though OOM is the likely
    # root cause (planner: this is a different failure), and not
    # ENGINE_CRASHED (the engine exited 0).
    assert errors.DIARIZATION_DEGRADED not in {
        errors.OUT_OF_MEMORY, errors.ENGINE_CRASHED, errors.INTERNAL}
    exc = tr.DiarizationDegradedError(
        "crispasr diarization produced 2 speaker regions for 8411 s of "
        "audio (floor 70): the global sherpa pre-pass degraded; raise the "
        "container memory limit or use a shorter file")
    assert errors.classify(exc) == errors.DIARIZATION_DEGRADED
    assert errors.classify_message(str(exc)) == errors.DIARIZATION_DEGRADED


def test_degraded_signal_makes_error_diagnostic(configured_env, monkeypatch):
    # AC5 feeding AC4: the engine's own failure text reaches the client
    # without container logs.
    tmp = configured_env
    script_body = f'''
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
cat > "$OF.json" <<'EOF'
{json.dumps(DEGRADED_JSON)}
EOF
cat >&2 <<'ERR'
{MEASURED_ENGINE_STDERR}
ERR
'''
    fake = _fake_binary(tmp, "crispasr-diag", script_body)
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp / "in.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(str(wav), duration=8411.072)
    msg = str(exc.value)
    assert "2" in msg and "8411" in msg
    assert "memory" in msg.lower() and "shorter" in msg.lower()
    assert "137" in msg


def test_engine_exit_137_maps_to_out_of_memory():
    # The classification boundary between the two codes: an engine that
    # DIES with a SIGKILL/exit-137 signature in its stderr tail is an
    # OUT_OF_MEMORY job (existing t_1799d07f rule); the exit-0 run whose
    # stderr shows the SILENT fallback — whose message itself mentions
    # "OOM-killed (exit 137 = SIGKILL)" — must classify as
    # DIARIZATION_DEGRADED, never be stolen by the OOM pattern.
    from pipeline import errors
    assert errors.classify_message(
        "crispasr exited with code 1: killed process 137 out of memory "
        "failure") == errors.OUT_OF_MEMORY
    assert errors.classify_message(
        "crispasr diarization degraded: the global sherpa pre-pass failed "
        "and the engine fell back to per-slice diarization for 8411 s of "
        "audio. Likely cause: the pre-pass was OOM-killed (exit 137 = "
        "SIGKILL)") == errors.DIARIZATION_DEGRADED
    assert errors.classify(tr.DiarizationDegradedError(
        "…OOM-killed (exit 137)…")) == errors.DIARIZATION_DEGRADED


# ---------------------------------------------------------------------------
# Threshold is operator-tunable (knob catalogued) but measured-default safe
# ---------------------------------------------------------------------------

def test_threshold_env_override_rejects_more(configured_env, monkeypatch):
    # The floor constants are derived from measurements, but an operator
    # can raise the floor; the knob must be read, not hardcoded only.
    monkeypatch.setenv("DIARIZATION_MIN_REGIONS_PER_MINUTE", "1.0")
    with pytest.raises(tr.CrispasrError):
        tr._check_diarization_plausibility(
            [{"t0": 0.0, "t1": 1.0, "speaker": 0},
             {"t0": 2.0, "t1": 3.0, "speaker": 1},
             {"t0": 4.0, "t1": 5.0, "speaker": 2}],
            duration_seconds=8411.072)
    tr._check_diarization_plausibility(
        [{"t0": 0.0, "t1": 1.0, "speaker": 0},
         {"t0": 2.0, "t1": 3.0, "speaker": 1}],
        duration_seconds=27.0)


def test_threshold_knobs_are_catalogued():
    from pipeline.config_vars import CATALOG
    names = {v.name for v in CATALOG}
    assert "DIARIZATION_MIN_REGIONS_PER_MINUTE" in names
    assert "DIARIZATION_MIN_CHECK_SECONDS" in names
