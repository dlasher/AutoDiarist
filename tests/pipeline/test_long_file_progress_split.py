"""Long-file progress: split the ASR and diarization invocations (card t_7685c6da).

The engine's ORDER (confirmed in the pinned crispasr source, crispasr_run.cpp):
``crispasr_compute_sherpa_cache()`` — the external sherpa diarization
pre-pass — runs BEFORE the slice loop that emits the per-slice
``crispasr: progress = X% (k/N slices)`` stderr lines. A 119-minute file
therefore publishes ZERO progress during ASR whenever the diarization is
slow enough to matter, while a short file publishes fine — which is why the
defect hid behind the 249 s panel fixture.

The fix is the invocation SPLIT:
  * invocation 1: ASR only — ``build_crispasr_command`` must NOT pass
    ``--diarize``/``--sherpa-*`` when the diarization is delegated to the
    pipeline (DIAR_METHOD=sherpa/ecapa, SHERPA_BIN set).
  * invocation 2: the pipeline spawns the sherpa wrapper itself, exactly as
    ``crispasr_compute_sherpa_cache`` does (engine contract:
    ``--clustering.num-clusters=0 --segmentation.pyannote-model=SEG
    --embedding.model=EMB <wav>``), published as the separate
    ``diarizing`` stage.
  * merge: replicate ``assign_speakers_from_sherpa`` (max time-overlap
    dominant speaker, strict '>' scan = lowest index wins ties, segment
    fully outside every region keeps speaker=None).

DIAR_METHOD=none is untouched: it passes no ``--diarize`` to the engine
anyway and is diarization-free end to end.
"""

import json
import os
import re
import time
from pathlib import Path

import pytest

from pipeline import transcribe as tr


CONFIG_ENVS = (
    "CRISPASR_BIN", "GGUF_MODEL", "DIAR_METHOD", "CRISPASR_LANGUAGE",
    "CRISPASR_GPU_BACKEND", "CRISPASR_CACHE_DIR", "CRISPASR_LIB_DIR",
    "CRISPASR_TIMEOUT_SECONDS", "MIN_FRAGMENT_SECONDS", "VRAM_LIMIT_MIB",
    "MOCK_PIPELINE", "CHUNK_SECONDS", "OVERLAP_SECONDS",
    "SHERPA_SEGMENT_MODEL", "SHERPA_EMBEDDING_MODEL", "SHERPA_BIN",
    "CRISPASR_SHERPA_TIMEOUT_SEC",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in CONFIG_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRISPASR_BIN", "/opt/crispasr/crispasr")
    monkeypatch.setenv("GGUF_MODEL", "/models/nemotron.gguf")
    tr._BACKEND_PROBE_CACHE.clear()
    yield


def _fake_binary(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _asr_fake(tmp_path: Path, argv_dump: Path) -> Path:
    """ASR-only fake engine: per-slice -pp progress on stderr WHILE ALIVE.

    Emits slice 1, sleeps, emits slice 2, writes the JSON and exits,
    touching ``engine-exited`` as its very last act. The adapter must have
    consumed BOTH slice boundaries before that marker exists — i.e. during
    the run, not drained after exit — which is exactly what the un-split
    invocation cannot deliver once the diarization pre-pass is slow.
    """
    out = {
        "transcription": [
            {"timestamps": {"from": "00:00:01,000", "to": "00:00:04,000"},
             "offsets": {"from": 1000, "to": 4000}, "text": "hello there",
             "chunk_id": 0},
            {"timestamps": {"from": "00:00:31,000", "to": "00:00:35,000"},
             "offsets": {"from": 31000, "to": 35000}, "text": "goodbye now",
             "chunk_id": 1},
        ]
    }
    marker = str(Path(argv_dump).with_name("engine-exited"))
    script = f"""
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
printf '%s\\n' "$@" > {argv_dump}
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
echo "crispasr: progress =   0% (1/2 slices)" >&2
sleep 0.5
echo "crispasr: progress = 100% (2/2 slices)" >&2
sleep 0.3
cat > "$OF.json" <<'EOF'
{json.dumps(out)}
EOF
touch {marker}
"""
    return _fake_binary(tmp_path, "crispasr-split", script)


def _sleeping_sherpa_stub(tmp_path: Path) -> Path:
    """Stub sherpa diarizer: sleeps, then prints a region table.

    The sleep discriminates the measured defect condition — 'external
    sherpa subprocess + a diarization slow enough to matter' — without a
    long fixture or a GPU.
    """
    script = """
sleep 0.4
printf '0.500 -- 3.000 speaker_00 duration=2.500\\n'
printf '3.000 -- 36.000 speaker_01 duration=33.000\\n'
exit 0
"""
    return _fake_binary(tmp_path, "sherpa-sleep", script)


def _wav_fixture(tmp_path: Path) -> Path:
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFFfake")
    return wav


# ---------------------------------------------------------------------------
# AC1 structural: the ASR invocation must NOT carry the engine diarization
# ---------------------------------------------------------------------------

def test_split_asr_command_omits_engine_diarization(monkeypatch):
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_BIN", "/usr/local/bin/sherpa-diarization-wrapper.sh")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")
    assert "--diarize" not in cmd
    assert "--diarize-method" not in cmd
    assert "--sherpa-bin" not in cmd
    assert "--sherpa-segment-model" not in cmd
    assert "-pp" in cmd


def test_split_asr_command_unchanged_for_none_method(monkeypatch):
    # AC6: the none path never passed --diarize and stays untouched.
    monkeypatch.setenv("DIAR_METHOD", "none")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")
    assert "--diarize" not in cmd and "-pp" in cmd


def test_no_diarize_engine_still_gets_the_diarize_flag(monkeypatch):
    # Methods with no pipeline-side split (foxnose, sortformer) keep the
    # single-invocation contract — the split is scoped to sherpa/ecapa.
    monkeypatch.setenv("DIAR_METHOD", "foxnose")
    cmd = tr.build_crispasr_command("/tmp/in.wav", "/tmp/out")
    assert "--diarize" in cmd and "--diarize-method" in cmd


# ---------------------------------------------------------------------------
# AC3 structural: invocation 2 matches crispasr_compute_sherpa_cache's
# engine contract exactly (make_sherpa_args + the wav the cache was fed).
# ---------------------------------------------------------------------------

def test_build_sherpa_diarize_command_contract(monkeypatch, tmp_path):
    seg = tmp_path / "seg.onnx"
    emb = tmp_path / "emb.onnx"
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", str(seg))
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", str(emb))
    cmd = tr.build_sherpa_diarize_command("/tmp/full16k.wav")
    assert cmd == [
        tr.sherpa_bin(),
        "--clustering.num-clusters=0",
        f"--segmentation.pyannote-model={seg}",
        f"--embedding.model={emb}",
        "/tmp/full16k.wav",
    ]


# ---------------------------------------------------------------------------
# AC1/AC3 behaviour: while the sleeping diarizer is in flight, the
# transcribing stage publishes MORE THAN ONE progress value, and the
# diarizing stage publishes its own progress as a separate phase.
# ---------------------------------------------------------------------------

def test_long_job_publishes_progress_during_slow_diarization(tmp_path, monkeypatch):
    argv_dump = tmp_path / "engine-argv.txt"
    engine = _asr_fake(tmp_path, argv_dump)
    sherpa = _sleeping_sherpa_stub(tmp_path)
    monkeypatch.setenv("CRISPASR_BIN", str(engine))
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_BIN", str(sherpa))
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 60.0)

    events: list[tuple[float, str, int, int]] = []
    marker = tmp_path / "engine-exited"
    t0 = time.monotonic()

    def cb(stage, done, total):
        if stage == "asr" and marker.exists():
            events.append((time.monotonic() - t0, "asr-after-exit",
                           done, total))
        events.append((time.monotonic() - t0, stage, done, total))

    segments = tr.transcribe_and_diarize(str(_wav_fixture(tmp_path)),
                                         progress=cb)

    asr_events = [e for e in events if e[1] == "asr"]
    diar_events = [e for e in events if e[1] == "diar"]
    # AC1: more than ONE distinct ASR boundary, published DURING the run
    # (never after the engine exited — the pump's exit-drain must not be
    # what carries the boundaries).
    assert len({(d, t) for _, _, d, t in asr_events}) > 1, asr_events
    assert not [e for e in events if e[1] == "asr-after-exit"], events
    # AC3: diarizing is a separate published phase, after the ASR band.
    assert diar_events, events
    # AC4: the public sequence is non-decreasing in done/total.
    ratios = [(e[2] / e[3]) for e in events if e[1] in ("asr", "diar")]
    assert ratios == sorted(ratios), events
    # The ASR invocation really ran without the engine diarization flags.
    argv_text = argv_dump.read_text()
    assert "--diarize" not in argv_text and "--sherpa-bin" not in argv_text
    # The merged transcript carries the diarizer's labels: region 1
    # (speaker_00) covers segment A, region 2 (speaker_01) segment B, and
    # the normalizer re-labels by first appearance -> Speaker 1, Speaker 2.
    assert [s["speaker"] for s in segments] == ["Speaker 1", "Speaker 2"]


# ---------------------------------------------------------------------------
# Merge fidelity: pure replication of assign_speakers_from_sherpa
# ---------------------------------------------------------------------------

def _region(t0, t1, spk):
    return {"t0": t0, "t1": t1, "speaker": spk}


def test_merge_dominant_speaker_max_overlap():
    segments = [{"text": "a", "t0": 0.0, "t1": 10.0},
                {"text": "b", "t0": 10.0, "t1": 20.0}]
    regions = [_region(0.0, 10.0, 3), _region(5.0, 15.0, 1),
               _region(10.0, 20.0, 5)]
    tr._merge_diarization_regions(segments, regions)
    # seg A: spk3 overlap 10.0 > spk1 overlap 5.0 -> spk3.
    assert segments[0]["_speaker_raw"] == 3
    # seg B: spk1 overlap 5.0, spk5 overlap 10.0 -> spk5.
    assert segments[1]["_speaker_raw"] == 5


def test_merge_tie_lowest_speaker_index_wins():
    # The engine scans speaker indices ascending with a STRICT '>'
    # (crispasr_diarize_cli.cpp assign_speakers_from_sherpa), so on equal
    # overlap the lowest index keeps the label.
    segments = [{"text": "a", "t0": 0.0, "t1": 10.0}]
    regions = [_region(0.0, 5.0, 4), _region(5.0, 10.0, 4),
               _region(0.0, 5.0, 2), _region(5.0, 10.0, 2)]
    # spk2 total 10.0 vs spk4 total 10.0 -> spk2 wins the tie.
    tr._merge_diarization_regions(segments, regions)
    assert segments[0]["_speaker_raw"] == 2


def test_merge_segment_outside_every_region_stays_unassigned():
    segments = [{"text": "a", "t0": 40.0, "t1": 45.0}]
    tr._merge_diarization_regions(segments, [_region(0.0, 10.0, 0)])
    assert segments[0]["_speaker_raw"] is None


def test_merge_parser_accepts_both_sherpa_line_shapes():
    lines = [
        "config banner mentioning 3dspeaker_...3dspeaker_16k.onnx",
        "0.320 -- 3.680 speaker_00 duration=3.360",
        "0.320 3.680 7",
        "",
    ]
    parsed = tr._parse_sherpa_regions("\n".join(lines))
    assert parsed == [_region(0.320, 3.680, 0), _region(0.320, 3.680, 7)]


def test_merge_normalizes_through_pipeline_labels():
    # (speaker N) strings feed _normalize_speakers exactly as the engine's
    # own label shape does: first-appearance order -> Speaker 1, 2...
    segments = [{"text": "x", "t0": 0.0, "t1": 2.0, "_speaker_raw": 3},
                {"text": "y", "t0": 2.0, "t1": 4.0, "_speaker_raw": 0}]
    out = tr._normalize_speakers(segments, method="sherpa")
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2"]


# ---------------------------------------------------------------------------
# Fail-loud parity with apply_sherpa: an empty/failed region table is a
# diarization FAILURE, never a silent single-speaker transcript.
# ---------------------------------------------------------------------------

def test_sherpa_failure_raises_not_silent(monkeypatch, tmp_path):
    fail = _fake_binary(tmp_path, "sherpa-fail", "exit 3\n")
    monkeypatch.setenv("SHERPA_SEGMENT_MODEL", str(tmp_path / "s.onnx"))
    monkeypatch.setenv("SHERPA_EMBEDDING_MODEL", str(tmp_path / "e.onnx"))
    monkeypatch.setenv("SHERPA_BIN", str(fail))
    with pytest.raises(tr.CrispasrError):
        tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)))
