"""Global concurrency (MAX_PROCESSING_GLOBAL) acceptance tests — card t_dbad2235.

The queue worker loop was serial BY CONSTRUCTION: one task, one awaited job
at a time. V1.5.1 replaces it with N concurrent workers behind a claim-time
global in-flight cap. The default must stay 1 so an existing deployment is
byte-for-byte unchanged until it opts in.

Deterministic strategy (same discipline as tests/api/test_fair_share.py):
`process_job_background` is monkeypatched with a GATE — a call marks the job
in-flight (the claim already wrote `processing` on disk), then blocks until the
test releases it. While every gated job is parked, the observable in-flight
count is exact, not a race. GPU is never required.

AC map:
  1  default unset -> exactly one job at a time
  2  MAX_PROCESSING_GLOBAL=3 -> three jobs simultaneously in processing
  3  cap never exceeded with more queued jobs than slots (claim-time unit
     test + end-to-end peak test)
  4  a raising job is marked failed with the real error text; the worker
     CONTINUES and later queued jobs still complete
  5  3 workers x 3 sessions: no session starved; per-session cap still
     serialises one session's jobs
  6  0 / -1 / 'abc' warn+clamp to 1; ' 3 ' parses to 3
  8  validate_config() reports max_processing_global=N; the resolved worker
     count is logged at startup; GET /v1/jobs reports processing_count
  +  _WakeHub fan-out: N waiters, no lost wake-up (one worker's clear cannot
     strand the others)
"""
import asyncio
import json
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."


def _setup(tmp_path, monkeypatch, worker="1", global_cap=None, per_session_cap=None,
           poll="0.02"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", poll)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 10 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    if global_cap is None:
        monkeypatch.delenv("MAX_PROCESSING_GLOBAL", raising=False)
    else:
        monkeypatch.setenv("MAX_PROCESSING_GLOBAL", str(global_cap))
    if per_session_cap is None:
        monkeypatch.delenv("MAX_PROCESSING_PER_SESSION", raising=False)
    else:
        monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", str(per_session_cap))
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("MAX_JOBS_PER_SESSION", raising=False)
    return root


def _post(client, session, name="a.wav"):
    return client.post(
        "/v1/jobs",
        files={"file": (name, AUDIO, "audio/wav")},
        headers={"X-Session": session},
    )


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _write_state(root, state):
    job_dir = root / state["token"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "audio_source.wav").write_bytes(AUDIO)
    (job_dir / "status.json").write_text(json.dumps(state))


def _mk(token, seq, status_, session):
    return {
        "token": token,
        "status": status_,
        "seq": seq,
        "session": session,
        "session_ident": session,
        "filename": f"{session}-{seq}.wav",
        "queued_at": float(seq),
        "message": "",
        "progress": 0.0,
    }


def _count_inflight(root):
    """On-disk truth: jobs currently in status == 'processing'."""
    n = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        sf = d / "status.json"
        if not sf.exists():
            continue
        try:
            data = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # torn read mid-write: skip, never crash the poller
        if data.get("status") == "processing":
            n += 1
    return n


def _wait_until(pred, why, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for: {why}")


class _Gate:
    """Park process_job_background calls until the test releases them."""

    def __init__(self):
        self.entered = []
        self.active = 0
        self.peak_active = 0
        self.release = threading.Event()
        self.lock = threading.Lock()

    def install(self, monkeypatch):
        real = main.process_job_background

        def gated(token, audio_path, filename):
            with self.lock:
                self.entered.append(token)
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
            assert self.release.wait(30), "test forgot to release the gate"
            with self.lock:
                self.active -= 1
            return real(token, audio_path, filename)

        monkeypatch.setattr(main, "process_job_background", gated)
        return self

    def started(self):
        with self.lock:
            return list(self.entered)

    def open(self):
        self.release.set()


class _Poller:
    """Sample the on-disk in-flight count so peaks are observed, not guessed."""

    def __init__(self, root):
        self.root = root
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, _count_inflight(self.root))
            self.stop.wait(0.01)

    def start(self):
        self.thread.start()
        return self

    def stop_and_join(self):
        self.stop.set()
        self.thread.join(timeout=5)


# --- AC1: unset knob == today's serial behaviour ------------------------------


def test_ac1_default_is_strictly_serial(tmp_path, monkeypatch):
    """MAX_PROCESSING_GLOBAL unset: with 3 queued jobs max in-flight == 1."""
    root = _setup(tmp_path, monkeypatch)  # global cap not set
    assert main._max_processing_global() == 1
    gate = _Gate().install(monkeypatch)
    poller = _Poller(root).start()
    try:
        with TestClient(main.app) as client:
            toks = [_post(client, s).json()["token"] for s in
                    ("alice", "bob", "carol")]
            # The single slot gets taken; the other two jobs stay queued
            # while it is parked in the gate.
            _wait_until(lambda: len(gate.started()) == 1,
                        "first job claimed")
            time.sleep(0.3)  # grace: any leak to a 2nd worker would show up
            assert len(gate.started()) == 1, gate.started()
            assert _count_inflight(root) == 1
            gate.open()
            _wait_until(lambda: all(_state(root, t)["status"] == "done"
                                    for t in toks), "all three complete")
        assert poller.peak == 1, f"serial default observed peak {poller.peak}"
    finally:
        gate.open()
        poller.stop_and_join()


# --- AC2: the knob actually creates parallelism -------------------------------


def test_ac2_three_workers_three_jobs_in_processing_simultaneously(tmp_path,
                                                                   monkeypatch):
    root = _setup(tmp_path, monkeypatch, global_cap=3)
    gate = _Gate().install(monkeypatch)
    poller = _Poller(root).start()
    try:
        with TestClient(main.app) as client:
            toks = [_post(client, s).json()["token"] for s in
                    ("alice", "bob", "carol")]
            _wait_until(lambda: len(gate.started()) == 3,
                        "3 workers claim 3 jobs simultaneously")
            # All three are parked: the on-disk in-flight count is exact.
            assert gate.peak_active == 3
            for t in toks:
                assert _state(root, t)["status"] == "processing", t
            assert poller.peak == 3
            gate.open()
            _wait_until(lambda: all(_state(root, t)["status"] == "done"
                                    for t in toks), "all three complete")
    finally:
        gate.open()
        poller.stop_and_join()


# --- AC3: the cap is never exceeded -------------------------------------------


def test_ac3_claim_time_cap_blocks_when_global_inflight_is_full(tmp_path,
                                                                monkeypatch):
    """Deterministic unit: 3 in flight, cap 3 -> no claim; free a slot -> claim."""
    root = _setup(tmp_path, monkeypatch, global_cap=3)
    with TestClient(main.app):
        pass  # fresh scheduler pointer
    # Isolate the GLOBAL cap: widen the per-session cap to 3 so alice's three
    # processing jobs are legal per-session — the ONLY thing that should block
    # dave's five queued jobs is the service-wide in-flight ceiling.
    monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", "3")
    for i in range(3):
        _write_state(root, _mk(f"a{i}", i + 1, "processing", "alice"))
    for i in range(5):
        _write_state(root, _mk(f"q{i}", 10 + i, "queued", "dave"))
    assert main._claim_next_queued() is None
    # Free one slot: the next claim must succeed immediately.
    st = _state(root, "a1")
    st["status"] = "done"
    _write_state(root, st)
    job = main._claim_next_queued()
    assert job is not None and job["token"] == "q0"


def test_ac3_end_to_end_five_jobs_three_slots_never_exceeds_three(tmp_path,
                                                                  monkeypatch):
    root = _setup(tmp_path, monkeypatch, global_cap=3)
    gate = _Gate().install(monkeypatch)
    poller = _Poller(root).start()
    try:
        with TestClient(main.app) as client:
            toks = [_post(client, f"u{i}").json()["token"] for i in range(5)]
            _wait_until(lambda: len(gate.started()) == 3,
                        "three slots filled")
            time.sleep(0.3)  # grace: an over-claim would flip a 4th to processing
            assert len(gate.started()) == 3, gate.started()
            assert _count_inflight(root) == 3
            gate.open()
            _wait_until(lambda: all(_state(root, t)["status"] == "done"
                                    for t in toks), "all five complete")
        assert poller.peak == 3, f"in-flight peak observed {poller.peak} > cap 3"
    finally:
        gate.open()
        poller.stop_and_join()


# --- AC4: a failing job must not wedge a worker -------------------------------


def test_ac4_raising_job_fails_without_wedging_the_service(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)  # serial default: one worker
    real_transcribe = main.transcribe_and_diarize
    calls = []

    def flaky(wav_path, **kw):
        calls.append(wav_path)
        if len(calls) == 1:
            # Mirror the measured hard engine abort text (card NOTES).
            raise RuntimeError("nemotron: sched alloc encoder graph failed")
        return real_transcribe(wav_path, **kw)

    monkeypatch.setattr(main, "transcribe_and_diarize", flaky)
    with TestClient(main.app) as client:
        t1 = _post(client, "alice").json()["token"]
        t2 = _post(client, "bob").json()["token"]
        _wait_until(lambda: _state(root, t2)["status"] == "done",
                    "the worker continues past the failure")
        st1 = _state(root, t1)
        assert st1["status"] == "failed", st1
        # The real error text lands in the job record (AC4 / API contract).
        assert "sched alloc encoder graph failed" in (st1.get("error") or "")
        assert "sched alloc encoder graph failed" in (st1.get("message") or "")
        assert _state(root, t2)["status"] == "done"


# --- AC5: fairness survives the pool ------------------------------------------


def test_ac5_three_workers_three_sessions_all_claimed(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, global_cap=3)
    gate = _Gate().install(monkeypatch)
    try:
        with TestClient(main.app) as client:
            toks = [_post(client, s).json()["token"] for s in
                    ("alice", "bob", "carol")]
            _wait_until(lambda: len(gate.started()) == 3,
                        "no session starved")
            assert set(gate.started()) == set(toks)
            gate.open()
            _wait_until(lambda: all(_state(root, t)["status"] == "done"
                                    for t in toks), "all complete")
    finally:
        gate.open()


def test_ac5_single_session_still_serialised_by_per_session_cap(tmp_path,
                                                                 monkeypatch):
    """3 workers, one session, MAX_PROCESSING_PER_SESSION=1: never >1 in flight."""
    root = _setup(tmp_path, monkeypatch, global_cap=3, per_session_cap=1)
    gate = _Gate().install(monkeypatch)
    poller = _Poller(root).start()
    try:
        with TestClient(main.app) as client:
            toks = [_post(client, "alice", f"a{i}.wav").json()["token"]
                    for i in range(3)]
            _wait_until(lambda: len(gate.started()) == 1, "first job claimed")
            time.sleep(0.3)
            assert len(gate.started()) == 1, "per-session cap leaked a 2nd job"
            gate.open()
            _wait_until(lambda: all(_state(root, t)["status"] == "done"
                                    for t in toks), "all complete")
        assert poller.peak == 1, f"session ran {poller.peak} jobs at once"
    finally:
        gate.open()
        poller.stop_and_join()


# --- AC6: clamp behaviour, no silent ignore, no freeze ------------------------


@pytest.mark.parametrize("raw,expected", [
    ("0", 1),        # zero would freeze the queue
    ("-1", 1),       # negative likewise
    ("abc", 1),      # unparseable: warn, fall back, never crash
    ("", 1),         # empty means unset
    (" 3 ", 3),      # surrounding whitespace parses cleanly
    ("7", 7),        # plain value is honoured
])
def test_ac6_accessor_warns_and_clamps(tmp_path, monkeypatch, caplog, raw,
                                       expected):
    root = _setup(tmp_path, monkeypatch)
    if raw:
        monkeypatch.setenv("MAX_PROCESSING_GLOBAL", raw)
    else:
        monkeypatch.delenv("MAX_PROCESSING_GLOBAL", raising=False)
    with caplog.at_level(logging.WARNING):
        assert main._max_processing_global() == expected
    if raw and expected == 1 and raw != "1":
        assert any("MAX_PROCESSING_GLOBAL" in r.getMessage()
                   for r in caplog.records), \
            "invalid values must warn — never silently ignore the operator"
    # The claim path must not freeze with the clamped value:
    _write_state(root, _mk("a1", 1, "queued", "alice"))
    job = main._claim_next_queued()
    assert job is not None and job["token"] == "a1"


# --- AC8: the knob is visible at startup and on the API -----------------------


def test_ac8_validate_config_reports_resolved_value(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    summary = main.validate_config()
    assert "max_processing_global=1" in summary
    monkeypatch.setenv("MAX_PROCESSING_GLOBAL", "3")
    summary = main.validate_config()
    assert "max_processing_global=3" in summary


def test_ac8_startup_logs_worker_count(tmp_path, monkeypatch, caplog):
    _setup(tmp_path, monkeypatch, global_cap=3)
    with caplog.at_level(logging.INFO), TestClient(main.app):
        pass
    assert any("MAX_PROCESSING_GLOBAL=3" in r.getMessage()
               for r in caplog.records), "startup must state the worker count"


def test_ac8_inflight_count_readable_on_get_jobs(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, global_cap=3)
    gate = _Gate().install(monkeypatch)
    try:
        with TestClient(main.app) as client:
            t1 = _post(client, "alice").json()["token"]
            _wait_until(lambda: len(gate.started()) == 1, "job claimed")
            body = client.get("/v1/jobs", headers={"X-Session": "alice"}).json()
            assert "processing_count" in body
            assert int(body["processing_count"]) == 1
            gate.open()
            _wait_until(lambda: _state(root, t1)["status"] == "done",
                        "job completes")
            body = client.get("/v1/jobs", headers={"X-Session": "alice"}).json()
            assert int(body["processing_count"]) == 0
    finally:
        gate.open()


# --- wake fan-out under N waiters (NOTES: one clear() must not strand others) --


def test_wake_hub_set_wakes_every_waiter(tmp_path, monkeypatch):
    hub = main._WakeHub()
    ev_a, ev_b = hub.register(), hub.register()

    async def drive():
        hub.set()
        await hub.wait(ev_a, timeout=1.0)
        await hub.wait(ev_b, timeout=1.0)

    asyncio.run(drive())  # both complete: neither clear() can strand the other
    assert not ev_a.is_set() and not ev_b.is_set()


def test_wake_hub_unregister_keeps_remaining_waiters(tmp_path, monkeypatch):
    hub = main._WakeHub()
    ev_a, ev_b = hub.register(), hub.register()

    async def drive():
        hub.unregister(ev_a)
        hub.set()
        await hub.wait(ev_b, timeout=1.0)

    asyncio.run(drive())
