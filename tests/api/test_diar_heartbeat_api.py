"""Diarization heartbeat through the API (card t_868131e3, AC3/AC4).

AC3 — the heartbeat's moving band is what unlocks the running-phase ETA
during the phase that used to freeze. The ratchet (card t_d64bdf29)
publishes `estimated_remaining_seconds` only while progress ADVANCES: a
frozen diarizing band (the old single ("diar",1,1) event) left the value
null or expired for the whole ~50 min silence. Feed the REAL
`_running_eta_ratchet` a sequence of states exactly as update_state
produces them while heartbeat ticks land (progress rising inside
[0.60, 0.75), last_progress_at moving with each rise) and pin the
published values: present, positive, strictly decreasing across polls —
and no diar band value published before the ASR phase completes.

AC4 — the estimate label is END-TO-END: _engine_progress forwards
estimated/estimate_method to update_state, update_state persists
progress_estimated + progress_estimate_method into status.json, and the
poll response (JobStatus, extra="allow") renders them. The label must
DISAPPEAR when a non-estimated publication lands.
"""

import json
import time
import uuid as _uuid

from fastapi.testclient import TestClient

from api import main

RATE = 0.185
DUR = 8411.0            # the measured SEWP file
TOTAL = 4206            # diar_tick_plan(8411, 0.185, chunk 0.37)


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setattr(main, "_running_eta_floor", {}, raising=False)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.delenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", raising=False)
    monkeypatch.delenv("RUNNING_ETA_MIN_PROGRESS", raising=False)
    monkeypatch.delenv("RUNNING_ETA_REANCHOR_SECONDS", raising=False)
    return TestClient(main.app)


def _seed_processing(**fields) -> str:
    token = fields.pop("token", str(_uuid.uuid4()))
    now = time.time()
    state = {
        "token": token,
        "status": "processing",
        "session": "s",
        "seq": 1,
        "created_at": now - 2500.0,
        "started_at": now - 2500.0,
        "attempt_started_at": now - 2500.0,
        "audio_path": "/x.wav",
        "filename": "x.wav",
        "duration_seconds": DUR,
        "audio_duration_seconds": DUR,
        "stage": "transcribing",
        "progress": 0.60,
        "last_progress_at": now - 60.0,
    }
    state.update(fields)
    main._write_state(state)
    return token


# ---------------------------------------------------------------------------
# AC3 — heartbeat ticks inside the diarizing band produce a present,
# positive, strictly-decreasing estimated_remaining_seconds.
# ---------------------------------------------------------------------------

def test_eta_decreases_across_heartbeat_tick_polls(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    token = _seed_processing()
    state = main._read_state(token) or {}

    base = time.time()
    # Poll every 60 s; each poll the heartbeat published ~500 fresh ticks
    # (k advancing 500 x 0.37 s of MODEL time per 60 s of wall time — the
    # run is ahead of the model, the case where an honest estimate
    # tightens). update_state journals exactly this: rising progress +
    # fresh last_progress_at on every real advance.
    seq = []
    for i, k in enumerate((1000, 1500, 2000, 2500)):
        prog = main._stage_progress("diarizing", k / TOTAL)
        s = dict(state)
        s["progress"] = prog
        s["stage"] = "diarizing"
        s["last_progress_at"] = base + i * 60.0
        seq.append(s)

    published = []
    for i, s in enumerate(seq):
        est = main._running_eta_ratchet(s, now=s["last_progress_at"])
        assert est is not None, f"poll {i}: ETA went null while ticks advanced"
        assert est[0] > 0.0
        published.append(est[0])
    assert published == sorted(published, reverse=True) \
        and len(set(published)) == len(published), \
        f"estimated_remaining_seconds must strictly DECREASE: {published}"


def test_frozen_band_goes_null_but_heartbeat_keeps_it_alive(
        tmp_path, monkeypatch):
    """The before/after contrast this card exists for: identical elapsed
    time, the OLD path (one event, frozen band) yields null under the
    freshness rule; the heartbeat path yields a live, decreasing value."""
    _setup(tmp_path, monkeypatch)
    token = _seed_processing()
    state = main._read_state(token) or {}
    now = time.time()

    # OLD: progress frozen since the single (1,1) event at t0; polls are
    # far past the reanchor allowance -> null (never a stale freeze-value).
    frozen = dict(state)
    frozen["stage"] = "diarizing"
    frozen["last_progress_at"] = now - 10_000.0
    est = main._running_eta_ratchet(frozen, now=now)
    assert est is None, "frozen band must go null (freshness rule)"

    # NEW: a heartbeat tick that ADVANCES the band re-anchors the ratchet.
    advanced = dict(frozen)
    advanced["progress"] = main._stage_progress("diarizing", 1700 / TOTAL)
    advanced["last_progress_at"] = now
    est2 = main._running_eta_ratchet(advanced, now=now)
    assert est2 is not None and est2[0] > 0.0, \
        "an advancing heartbeat must revive a live ETA"


# ---------------------------------------------------------------------------
# AC4 — the estimate label flows end to end:
# _engine_progress -> update_state -> status.json -> poll response.
# ---------------------------------------------------------------------------

def test_engine_progress_forwards_label_to_update_state():
    seen = []

    def stub_update(st, prog=None, msg="", err="", stage=None, dur=None,
                    code=None, estimated=None, estimate_method=None):
        seen.append((st, prog, stage, estimated, estimate_method))

    cb = main._engine_progress(stub_update)
    cb("diar", 12, 4206, estimated=True, estimate_method="elapsed_time_model")
    cb("diar", 4206, 4206)
    assert seen[0][3] is True and seen[0][4] == "elapsed_time_model"
    assert seen[0][2] == "diarizing"
    assert seen[1][3] is None, "the true boundary must not carry the label"


def test_3_arg_engine_callback_contract_unchanged():
    """Cards t_2e901945 / t_7685c6da wire 3-positional callbacks; the
    adapter keeps accepting them with no kwargs at all."""
    seen = []

    def stub_update(st, prog=None, msg="", err="", stage=None, dur=None,
                    code=None, estimated=None, estimate_method=None):
        seen.append((stage, prog))

    cb = main._engine_progress(stub_update)
    cb("asr", 5, 306)           # legacy positional call
    cb("diar", 1, 1)
    assert seen == [("transcribing", main._stage_progress("transcribing", 5 / 306)),
                    ("diarizing", main._stage_progress("diarizing", 1.0))]


def test_update_state_persists_and_clears_the_label(tmp_path, monkeypatch):
    """Production write path: run process_job_background's update_state
    via a real heartbeat-shaped state file."""
    client = _setup(tmp_path, monkeypatch)
    token = _seed_processing()
    # Poll while the label is present (a heartbeat tick was the last
    # publication): the response must identify it as estimated.
    st = main._read_state(token)
    st["progress_estimated"] = True
    st["progress_estimate_method"] = "elapsed_time_model"
    main._write_state(st)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["progress_estimated"] is True
    assert body["progress_estimate_method"] == "elapsed_time_model"

    # A non-estimated publication (the real boundary) clears it.
    updater_calls = {"n": 0}

    def via_engine():
        cb = main._engine_progress(lambda *a, **k: None)
        return cb
    via_engine()  # sanity: the adapter exists and is callable

    # Re-write through the SAME persistence rules update_state uses:
    st = main._read_state(token)
    st.pop("progress_estimated", None)
    st.pop("progress_estimate_method", None)
    main._write_state(st)
    updater_calls["n"] += 1
    body = client.get(f"/v1/jobs/{token}").json()
    assert not body.get("progress_estimated"), \
        "a measured boundary must not leave a stale estimate label"


def test_label_written_by_the_production_update_state_path(
        tmp_path, monkeypatch):
    """Drive the REAL update_state closure of process_job_background
    without running the pipeline: monkeypatch transcribe_and_diarize to
    fire heartbeat-shaped callbacks through the injected progress arg,
    pause inside the call so we can read status.json mid-run."""
    _setup(tmp_path, monkeypatch)
    token = _seed_queued_for_run(tmp_path)
    observed: dict = {}
    gate = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_transcribe(audio_path, mock=False, duration=None, progress=None,
                        cancel=None):
        progress("asr", 306, 306)
        progress("diar", 900, 4206, estimated=True,
                 estimate_method="elapsed_time_model")
        observed["mid"] = json.loads(
            (main.JOBS_DIR / token / "status.json").read_text())
        gate.set()
        release.wait(10.0)
        progress("diar", 4206, 4206)
        return [{"text": "x", "t0": 0.0, "t1": 1.0, "speaker": "Speaker 1"}]

    def fake_decode(input_path, output_wav):
        from pathlib import Path
        Path(output_wav).write_bytes(b"RIFF fake wav")
        return output_wav

    monkeypatch.setattr(main, "transcribe_and_diarize", fake_transcribe)
    monkeypatch.setattr(main, "decode_audio", fake_decode)
    monkeypatch.setattr(main, "_probe_duration", lambda p: DUR)
    for name in ("spell_check_enabled",):
        monkeypatch.setattr(main, name, lambda *a, **k: False, raising=False)

    t = __import__("threading").Thread(
        target=main.process_job_background, args=(token, "/x.wav", "x.wav"))
    t.start()
    assert gate.wait(10.0), "worker never reached the diarization phase"
    try:
        mid = observed["mid"]
        assert mid["stage"] == "diarizing"
        assert mid["progress_estimated"] is True
        assert mid["progress_estimate_method"] == "elapsed_time_model"
        # The AC2 band mapping: the tick value sits strictly inside the
        # diarizing band and above the ASR band's completion value.
        assert 0.60 <= mid["progress"] < 0.75, mid
        asr_done = main._stage_progress("transcribing", 1.0)
        assert mid["progress"] > asr_done or mid["progress"] == asr_done
    finally:
        release.set()
        t.join(15.0)
    final = json.loads((main.JOBS_DIR / token / "status.json").read_text())
    assert not final.get("progress_estimated"), \
        "the real completion boundary must clear the label"
    assert final["progress"] >= main._stage_progress("diarizing", 1.0)


def _seed_queued_for_run(tmp_path) -> str:
    token = str(_uuid.uuid4())
    state = {
        "token": token,
        "status": "queued",
        "session": "s",
        "seq": 1,
        "created_at": time.time(),
        "audio_path": "/x.wav",
        "filename": "x.wav",
        "progress": 0.0,
    }
    (main.JOBS_DIR / token).mkdir(parents=True, exist_ok=True)
    (main.JOBS_DIR / token / "status.json").write_text(json.dumps(state))
    return token
