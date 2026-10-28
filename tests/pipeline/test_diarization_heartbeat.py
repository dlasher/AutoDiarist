"""Diarization-stage heartbeat: a moving, labelled-estimate progress signal
(card t_868131e3).

Measured on the 8411 s SEWP file (branch 34e33c4): the ASR phase publishes
307 events over 855.9 s, then the pipeline-side sherpa diarization runs
3033 s publishing EXACTLY ONE event — ("diar", 1, 1) — at its end. The
sherpa CLI has no progress output, so the pipeline cannot observe real
boundaries; the fix is a TIME-BASED HEARTBEAT that advances a synthetic
tick counter toward the stage's total while the subprocess is alive, always
capped below the total, and publishes the true (total, total) boundary at
completion.

Honesty contract (planner ruling + AC4): every heartbeat tick is labelled
``estimated=True`` with ``estimate_method="elapsed_time_model"`` — the
labelled, monotonic estimate the card permits — and the final boundary is
NOT labelled (it is the real completion). ``total`` is derived from the
SAME rate constant the API ETA uses (PROCESSING_SECONDS_PER_AUDIO_SECOND,
default 0.185 s CPU per audio second — the card: "feed the same estimate
rather than inventing a second source of truth").

The heartbeat is driven by the STAGE's lifetime, not the subprocess's:
the wrapper's ceiling re-pass restarts the diarization underneath the
wait loop, and a restart must NOT rewind or reset the published count.
"""

import math
import threading
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
    "DIAR_HEARTBEAT_INTERVAL_SECONDS", "DIAR_CHUNK_ESTIMATE_SECONDS",
    "PROCESSING_SECONDS_PER_AUDIO_SECOND",
)

RATE = 0.185  # seconds of diarization CPU per second of audio (NV-measured)


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


def _sherpa_models(tmp_path: Path, monkeypatch) -> Path:
    """Point SHERPA_* model env at stub files; return the seg model path."""
    for env, fname in (("SHERPA_SEGMENT_MODEL", "seg.onnx"),
                       ("SHERPA_EMBEDDING_MODEL", "emb.onnx")):
        (tmp_path / fname).write_bytes(b"onnx-stub")
        monkeypatch.setenv(env, str(tmp_path / fname))
    return tmp_path / "seg.onnx"


def _region_lines(count: int = 2):
    """Region table for the stub diarizer.

    ``count`` regions spread over 36 s. The default 2 matches the
    short-audio fixtures; long-audio fixtures must pass a count above
    card t_dc5c2abb's duration-scaled floor, because a 2-region table over
    long audio is the measured OOM-fallback shape and is refused.
    """
    if count <= 2:
        return ("printf '0.500 -- 3.000 speaker_00 duration=2.500\\n'\n"
                "printf '3.000 -- 36.000 speaker_01 duration=33.000\\n'\n")
    lines = []
    step = 36.0 / count
    for i in range(count):
        start = i * step
        end = (i + 1) * step
        # speaker_N with NO zero padding: the parser's strict regex is
        # r"(\d+(?:\.\d+)?)\s+--\s+(\d+(?:\.\d+)?)\s+speaker_(\d+)"
        lines.append("printf '%.3f -- %.3f speaker_%d duration=%.3f\\n'\n"
                     % (start, end, i, step))
    return "".join(lines)


def _collector():
    """4-arg collector: (t, stage, done, total, kwargs) per event."""
    events: list = []
    t0 = time.monotonic()

    def cb(stage, done, total, **kw):
        events.append((time.monotonic() - t0, stage, done, total, kw))
    return events, cb


def _fast_plan(monkeypatch):
    """FP-exact tick math: rate 0.5 x chunk 0.0625 -> one model tick per
    0.125 s of wall time, heartbeat interval 0.1 s. duration 600 -> total
    4800. Powers of two keep float//floor exact: model=int(t/0.125)."""
    monkeypatch.setenv("DIAR_HEARTBEAT_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("DIAR_CHUNK_ESTIMATE_SECONDS", "0.0625")
    monkeypatch.setenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", "0.5")


def _wav_fixture(tmp_path: Path) -> Path:
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFFfake")
    return wav


# ---------------------------------------------------------------------------
# AC1 — the diarization stage publishes MORE THAN ONE event, strictly
# increasing `done`, ending at done == total == the true total.
# ---------------------------------------------------------------------------

def test_slow_diarization_publishes_moving_diar_series(tmp_path, monkeypatch):
    # FP-exact knobs: rate 0.5 s/audio-s x 600 s = 300 s cost estimate;
    # chunk 0.0625 -> total 4800 ticks, one tick per 0.125 s of wall time.
    # Stub sleeps 0.6 s; heartbeat every 0.1 s -> several model ticks
    # advance BEFORE the (4800, 4800) true boundary.
    stub = _fake_binary(tmp_path, "sherpa-slow", "sleep 0.6\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    events, cb = _collector()
    regions = tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                                        progress=cb, duration=600.0)
    assert regions  # real regions still returned
    diar = [(e[2], e[3]) for e in events if e[1] == "diar"]
    assert len(diar) >= 2, events
    dones = [d for d, _ in diar]
    assert dones == sorted(dones) and len(set(dones)) > 1, events
    assert all(d < t for d, t in diar[:-1]), \
        "heartbeat must cap strictly below the total (never claim completion)"
    assert diar[-1] == (diar[-1][1], diar[-1][1]), \
        f"final event must be the true (total, total): {diar[-1]}"
    assert diar[-1][1] == 4800, events


def test_heartbeat_ticks_live_during_the_run(tmp_path, monkeypatch):
    """The events move DURING the subprocess, not drained after exit —
    that timing is exactly what the freeze defeated."""
    ready = tmp_path / "in-flight"
    stub = _fake_binary(
        tmp_path, "sherpa-marker",
        f"touch {ready}\nsleep 0.6\nrm -f {ready}\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    events, cb = _collector()
    tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                              progress=cb, duration=600.0)
    # The t in each tuple is monotonic-since-start: the subprocess slept
    # 0.6 s, so any event recorded well before the final boundary was
    # published while the stub was in flight — the freeze made every
    # event land only after the exit.
    diar = [e for e in events if e[1] == "diar"]
    assert len(diar) >= 2
    assert diar[0][0] < diar[-1][0] - 0.2, \
        f"ticks must move over time, not fire at the end: {diar}"


# ---------------------------------------------------------------------------
# AC1/AC3 — survival of the wrapper's ceiling RE-PASS. The heartbeat is
# driven by the STAGE lifetime, not one subprocess: the published count
# must keep advancing across (and after) a mid-stage restart and never
# rewind. Stub: sleep, print one region line (pass 1 boundary), sleep more,
# print the rest (re-pass completion).
# ---------------------------------------------------------------------------

def test_heartbeat_survives_subprocess_restart_without_rewinding(
        tmp_path, monkeypatch):
    stub = _fake_binary(
        tmp_path, "sherpa-restart",
        "sleep 0.35\n"
        "printf '0.500 -- 3.000 speaker_00 duration=2.500\\n'\n"
        "sleep 0.35\n" + _region_lines() +
        "exit 0\n")
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    events, cb = _collector()
    regions = tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                                        progress=cb, duration=600.0)
    assert regions
    dones = [e[2] for e in events if e[1] == "diar"]
    assert dones == sorted(dones), f"restart must not rewind the count: {dones}"
    assert len(set(dones)) > 1, events
    assert dones[-1] == [e[3] for e in events if e[1] == "diar"][-1]


# ---------------------------------------------------------------------------
# AC4 — the heartbeat is LABELLED as an estimate; the final boundary is not.
# A client must never mistake a tick for a real work boundary.
# ---------------------------------------------------------------------------

def test_ticks_are_labelled_estimated_and_final_is_not(tmp_path, monkeypatch):
    stub = _fake_binary(tmp_path, "sherpa-slow2", "sleep 0.6\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    events, cb = _collector()
    tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                              progress=cb, duration=600.0)
    diar = [e for e in events if e[1] == "diar"]
    ticks, final = diar[:-1], diar[-1]
    assert ticks, "expected at least one heartbeat tick in the run window"
    for _, _, _, _, kw in ticks:
        assert kw.get("estimated") is True, kw
        assert kw.get("estimate_method") == "elapsed_time_model", kw
    assert not final[4].get("estimated"), \
        "the completion boundary is a real event, not an estimate"


def test_legacy_three_arg_callback_still_works(tmp_path, monkeypatch):
    """The publish signature must stay call-compatible with the 3-arg
    callbacks already wired by cards t_2e901945/t_7685c6da."""
    stub = _fake_binary(tmp_path, "sherpa-slow3", "sleep 0.4\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    seen: list = []
    regions = tr.run_sherpa_diarization(
        str(_wav_fixture(tmp_path)),
        progress=lambda stage, done, total: seen.append((stage, done, total)),
        duration=600.0)
    assert regions
    assert len(seen) >= 2
    assert seen[-1][1] == seen[-1][2]


# ---------------------------------------------------------------------------
# AC2 — the diar band keeps moving with the stage, and no diar event is
# published before the ASR phase completes. End-to-end through
# transcribe_and_diarize (the real split call site) with the API's band
# mapping mirrored.
# ---------------------------------------------------------------------------

def _asr_only_fake(tmp_path: Path, marker: Path) -> Path:
    out = {"transcription": [
        {"timestamps": {"from": "00:00:01,000", "to": "00:00:04,000"},
         "offsets": {"from": 1000, "to": 4000}, "text": "hello there",
         "chunk_id": 0},
        {"timestamps": {"from": "00:00:31,000", "to": "00:00:35,000"},
         "offsets": {"from": 31000, "to": 35000}, "text": "goodbye now",
         "chunk_id": 1},
    ]}
    import json as _json
    script = f"""
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
echo "crispasr: progress =   0% (1/2 slices)" >&2
sleep 0.3
echo "crispasr: progress = 100% (2/2 slices)" >&2
touch {marker}
cat > "$OF.json" <<'EOF'
{_json.dumps(out)}
EOF
"""
    return _fake_binary(tmp_path, "crispasr-hb-engine", script)


def test_split_run_diar_band_moves_and_starts_after_asr(tmp_path, monkeypatch):
    marker = tmp_path / "engine-exited"
    engine = _asr_only_fake(tmp_path, marker)
    # 600 s of audio needs a REALISTIC region count: card t_dc5c2abb's
    # guard refuses a 2-region table for long audio (the measured
    # OOM-fallback shape), so a 2-region stub over 600 s would be
    # correctly refused. 12 regions over 10 minutes is well above the
    # floor (5) and keeps the fixture legitimate.
    stub = _fake_binary(tmp_path, "sherpa-slow4", "sleep 0.6\n" + _region_lines(12))
    monkeypatch.setenv("CRISPASR_BIN", str(engine))
    monkeypatch.setenv("DIAR_METHOD", "sherpa")
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    gguf = tmp_path / tr.Q4K_GGUF_NAME
    gguf.write_bytes(b"ggmf")
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 600.0)
    _fast_plan(monkeypatch)

    events, cb = _collector()
    segments = tr.transcribe_and_diarize(str(_wav_fixture(tmp_path)),
                                         progress=cb)
    assert segments
    asr = [e for e in events if e[1] == "asr"]
    diar = [e for e in events if e[1] == "diar"]
    assert diar, events
    # AC2 (event order): no diar event before the ASR phase completes.
    asr_end = max(e[0] for e in asr)
    assert all(e[0] >= asr_end for e in diar), events
    # AC2 (band mapping): mirror transcribing [0.30,0.60), diarizing
    # [0.60,0.75) and require a NON-CONSTANT moving value inside diarizing.
    bands = {"asr": (0.30, 0.60), "diar": (0.60, 0.75)}
    public = [round(lo + min(1.0, e[2] / e[3]) * (hi - lo), 4)
              for e in events if e[1] in bands for lo, hi in [bands[e[1]]]]
    assert public == sorted(public), public
    diar_public = [round(0.60 + min(1.0, e[2] / e[3]) * 0.15, 4)
                   for e in diar]
    assert len(set(diar_public)) > 1, \
        f"diarizing band must MOVE (AC3 enabler): {diar_public}"


# ---------------------------------------------------------------------------
# AC5 — the stage stays CANCELLABLE while the heartbeat runs, and the
# heartbeat thread does not outlive the stage (no lock/process retention).
# ---------------------------------------------------------------------------

def test_cancel_during_heartbeat_kills_process_group(tmp_path, monkeypatch):
    stub = _fake_binary(tmp_path, "sherpa-sleepy",
                        "sleep 5\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    _fast_plan(monkeypatch)

    control = tr.CancelControl("tok-hb")
    events, cb = _collector()
    threads_before = threading.active_count()

    def cancel_soon():
        time.sleep(0.4)
        control.request()

    killer = threading.Thread(target=cancel_soon)
    killer.start()
    t0 = time.monotonic()
    with pytest.raises(tr.JobCancelled):
        tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                                  progress=cb, cancel=control, duration=600.0)
    killer.join()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"cancel must win promptly, took {elapsed:.1f}s"
    # Heartbeat ticks did move before the cancel.
    assert len([e for e in events if e[1] == "diar"]) >= 2, events
    # No thread leak: the heartbeat is stopped (joinable) on every exit path.
    time.sleep(0.3)
    assert threading.active_count() <= threads_before, \
        "heartbeat thread must not outlive the stage"


# ---------------------------------------------------------------------------
# AC6 — the short-file path is unchanged: a diarization that finishes
# inside ONE heartbeat interval publishes exactly the pre-card ("diar",1,1)
# final event — no ticks, same total of 1.
# ---------------------------------------------------------------------------

def test_short_file_publishes_single_legacy_diar_event(tmp_path, monkeypatch):
    stub = _fake_binary(tmp_path, "sherpa-fast", _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    # default interval (2.0 s) — the stub finishes far inside one tick.

    events, cb = _collector()
    regions = tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)),
                                        progress=cb, duration=27.0)
    assert regions
    diar = [(e[2], e[3]) for e in events if e[1] == "diar"]
    assert diar == [(1, 1)], f"short path must stay byte-identical: {diar}"
    assert not events[0][4].get("estimated")


# ---------------------------------------------------------------------------
# AC3 — the ETA feeds from the same moving progress: the heartbeat's tick
# plan uses the API's rate constant, and the API band math on a tick emits
# a strictly-lower remaining estimate than the previous tick (same observed
# rate model card t_d64bdf29 implements; here we pin that the DIAR band
# moves, which is what unlocks the ETA ratchet during the silent phase).
# ---------------------------------------------------------------------------

def test_tick_plan_uses_the_shared_rate_constant(monkeypatch):
    # 1000 s of audio at the shipped 0.185 s/audio-s rate = 185 s cost
    # estimate; 2 s default chunk granularity -> 500 total ticks.
    total, tick_seconds = tr.diar_tick_plan(duration=1000.0, rate=RATE,
                                            chunk_seconds=2.0)
    assert total == math.ceil(1000.0 * RATE / 2.0)  # 93
    assert abs(tick_seconds - 2.0 / RATE) < 1e-9


def test_rate_knob_override_flows_into_the_plan(monkeypatch):
    monkeypatch.setenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", "0.37")
    rate = tr.processing_rate_per_audio_second()
    assert abs(rate - 0.37) < 1e-12
    total, tick_seconds = tr.diar_tick_plan(duration=100.0, rate=rate,
                                            chunk_seconds=1.0)
    assert total == 37  # ceil(100 x 0.37 / 1)
    assert abs(tick_seconds - 1.0 / 0.37) < 1e-12


def test_diar_band_tick_progress_advances_the_eta_ratchet(tmp_path, monkeypatch):
    """AC3 integration: with the API band mapping, consecutive heartbeat
    totals of the real measured run (8411 s @ RATE, total 4206 ticks)
    produce decreasing estimated_remaining_seconds under the t_d64bdf29
    observed-rate math. Pin the math the API will actually run."""
    from api import main as api_main
    _sherpa_models(tmp_path, monkeypatch)  # not needed; keeps env hygiene
    now = time.time()
    # A job whose attempt started 2500 s ago and whose monotonic band
    # progress the heartbeat moved from 0.66 to 0.69.
    dur = 8411.0
    f1 = 0.60 + min(1.0, 1678 / 4206) * 0.15   # tick 1678 -> 0.65994
    f2 = 0.60 + min(1.0, 1893 / 4206) * 0.15   # tick 1893 -> 0.66759
    est1 = api_main._running_estimate(
        {"status": "processing", "progress": round(f1, 4),
         "started_at": now - 2500.0, "duration_seconds": dur}, now)
    est2 = api_main._running_estimate(
        {"status": "processing", "progress": round(f2, 4),
         "started_at": now - 2500.0, "duration_seconds": dur}, now)
    assert est1 is not None and est2 is not None
    assert 0 < est2[0] < est1[0], (est1, est2)


# ---------------------------------------------------------------------------
# AC7 — knobs are catalogued (t_1799d07f checker) and honest about bounds:
# a total of 0 (duration unknown) disables the heartbeat entirely rather
# than fabricating a counter; junk chunk/interval values fall back.
# ---------------------------------------------------------------------------

def test_unknown_duration_emits_no_ticks(tmp_path, monkeypatch):
    stub = _fake_binary(tmp_path, "sherpa-slow5", "sleep 0.5\n" + _region_lines())
    _sherpa_models(tmp_path, monkeypatch)
    monkeypatch.setenv("SHERPA_BIN", str(stub))
    monkeypatch.setenv("DIAR_HEARTBEAT_INTERVAL_SECONDS", "0.1")

    events, cb = _collector()
    tr.run_sherpa_diarization(str(_wav_fixture(tmp_path)), progress=cb)
    diar = [(e[2], e[3]) for e in events if e[1] == "diar"]
    assert diar == [(1, 1)], f"no duration -> no fabricated ticks: {diar}"


def test_heartbeat_knobs_are_catalogued():
    from pipeline import config_vars
    by_name = {e.name: e for e in config_vars.CATALOG}
    assert "DIAR_HEARTBEAT_INTERVAL_SECONDS" in by_name
    assert "DIAR_CHUNK_ESTIMATE_SECONDS" in by_name
    for name in ("DIAR_HEARTBEAT_INTERVAL_SECONDS",
                 "DIAR_CHUNK_ESTIMATE_SECONDS"):
        rec = by_name[name]
        assert rec.default is not None and float(rec.default) > 0
        assert "estimat" in rec.effect.lower(), \
            f"{name} effect must label the ticks as estimates"


def test_junk_knob_values_fall_back(tmp_path, monkeypatch):
    """Negative/zero/junk interval and chunk values never freeze a
    forever-tick; the helper falls back to the defaults with the house
    warning pattern (config errors must not crash the stage)."""
    monkeypatch.setenv("DIAR_HEARTBEAT_INTERVAL_SECONDS", "-4")
    monkeypatch.setenv("DIAR_CHUNK_ESTIMATE_SECONDS", "junk")
    interval, chunk = tr.diar_heartbeat_settings()
    assert interval > 0
    assert chunk > 0
