"""V1.5 queue-aware wait estimate acceptance tests (card t_b3ae9366, AC1-AC6).

Honest ETA for queued clients: duration probed cheaply at admission
(ffprobe header read — NEVER a full decode, NEVER re-probed at request
time), a linear model duration_seconds x PROCESSING_SECONDS_PER_AUDIO_SECOND
(hardware-specific constant; the 0.185 default derives from the measured
T400 run: 39-min audio took 423-434 s wall), and a prefix-sum walk of the
FAIR execution order: the currently-processing job's remaining estimate,
plus every queued job ahead, plus the job's own estimate. Every poll
recomputes from live state, so queue movement (new jobs, cancellations,
completions) is reflected immediately.

The estimate is an ESTIMATE, never a guarantee: it ignores diarization
variance and chunk overhead; observed wall time differs.

Deterministic strategy: the model helpers are driven directly against
seeded on-disk state (worker OFF), with the probe monkeypatched so no
ffprobe subprocess is ever spawned by the assertions; one HTTP test pins
the GET /v1/jobs/{token} surface and the 202 admission body. No GPU
required anywhere.
"""
import json
import math
import time
import uuid as _uuid

import pytest
from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."
RATE = 0.185


def _setup(tmp_path, monkeypatch, worker="0", rate=None):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 10 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", raising=False)
    monkeypatch.delenv("AUDIO_DURATION_FALLBACK_SECONDS", raising=False)
    if rate is not None:
        monkeypatch.setenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", str(rate))
    return root


def _mk(token, status_="queued", seq=0, session="s", duration=None, **extra):
    state = {
        "token": token, "status": status_, "seq": seq, "session": session,
        "filename": "a.wav", "duration_seconds": duration,
    }
    state.update(extra)
    return state


def _seed(root, states):
    for st in states:
        d = root / st["token"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(json.dumps(st))


def _sched():
    return {"serve_counter": 0, "last_served": {}}


def _est(root, sched=None, now=None):
    return main._wait_estimates(main._iter_states(),
                                sched if sched is not None else _sched(),
                                now=now)


# --- AC1: duration probe at admission, stored in job state -------------------

def test_admission_stores_probed_duration(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 123.456)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    assert r.status_code == 202
    st = json.loads((root / r.json()["token"] / "status.json").read_text())
    assert st["duration_seconds"] == pytest.approx(123.456)


def test_admission_probe_failure_stores_none(tmp_path, monkeypatch):
    # A failed/absent probe (0.0 sentinel) records duration_seconds=None —
    # "unknown", never a fake 0 that would silently zero the ETA.
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 0.0)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    st = json.loads((root / r.json()["token"] / "status.json").read_text())
    assert st["duration_seconds"] is None


def test_runtime_probe_refreshes_admission_value(tmp_path, monkeypatch):
    # AC1 "recorded at admission, refreshed when the job actually runs":
    # the non-mock decode path re-probes the 16k wav and persists the truth.
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 42.0)
    monkeypatch.setattr(main, "decode_audio", lambda src, dst: dst)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        lambda path, **kw: [{"start": 0, "end": 1, "text": "x", "speaker": "S0"}])
    monkeypatch.setenv("MOCK_PIPELINE", "0")
    tok = "rt1"
    d = root / tok
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps(
        _mk(tok, duration=10.0, status="queued")))
    main.process_job_background(tok, str(d / "audio_source.wav"), "rt.wav")
    st = json.loads((d / "status.json").read_text())
    assert st["duration_seconds"] == pytest.approx(42.0)


# --- AC2: documented, configurable rate constant ------------------------------

def test_default_rate_matches_measured_t400():
    assert main.DEFAULT_PROCESSING_SECONDS_PER_AUDIO_SECOND == pytest.approx(RATE)


def test_rate_env_override():
    import os
    os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"] = "0.5"
    try:
        assert main._processing_rate() == pytest.approx(0.5)
    finally:
        del os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"]


def test_invalid_rate_warns_and_falls_back(caplog):
    import logging
    import os
    os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"] = "abc"
    try:
        with caplog.at_level(logging.WARNING, logger="api.main"):
            assert main._processing_rate() == pytest.approx(RATE)
        assert any("PROCESSING_SECONDS_PER_AUDIO_SECOND" in r.message
                   for r in caplog.records)
    finally:
        del os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"]


def test_negative_rate_warns_and_falls_back(caplog):
    import logging
    import os
    os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"] = "-1"
    try:
        with caplog.at_level(logging.WARNING, logger="api.main"):
            assert main._processing_rate() == pytest.approx(RATE)
        assert any("PROCESSING_SECONDS_PER_AUDIO_SECOND" in r.message
                   for r in caplog.records)
    finally:
        del os.environ["PROCESSING_SECONDS_PER_AUDIO_SECOND"]


def test_zero_rate_disables_eta_emission(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, rate=0)
    _seed(root, [_mk("z1", duration=100.0)])
    assert _est(root) == {}


# --- AC3/AC4: queue-aware prefix-sum estimates --------------------------------

def test_empty_queue_estimate_single_job(tmp_path, monkeypatch):
    # "Estimate for an empty queue": one queued job, nothing ahead, no
    # processing job -> wait == own duration x rate, completion == now + wait.
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("e1", duration=100.0)])
    now = 1_000.0
    out = _est(root, now=now)
    assert out["e1"][0] == pytest.approx(100.0 * RATE, abs=0.05)
    assert out["e1"][1] == pytest.approx(now + 100.0 * RATE, abs=0.05)


def test_estimate_with_N_ahead_is_prefix_sum(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("a", duration=600.0, seq=0),
                 _mk("b", duration=300.0, seq=1),
                 _mk("c", duration=100.0, seq=2)])
    out = _est(root)
    assert out["a"][0] == pytest.approx(600 * RATE, abs=0.05)
    assert out["b"][0] == pytest.approx(900 * RATE, abs=0.05)
    assert out["c"][0] == pytest.approx(1000 * RATE, abs=0.05)
    # AC4: distinct, non-overlapping per position; chain of completion times.
    assert out["b"][1] == pytest.approx(out["a"][1] + 300 * RATE, abs=0.05)
    assert out["c"][1] == pytest.approx(out["b"][1] + 100 * RATE, abs=0.05)


def test_uses_fair_scheduled_order_not_admission_order(tmp_path, monkeypatch):
    # RR across sessions: with the pointer having just served session "a",
    # session "b" (later admission) runs FIRST — the ETA must follow the
    # real execution order, not admission FIFO.
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("a1", session="a", seq=0, duration=100.0),
                 _mk("a2", session="a", seq=2, duration=300.0),
                 _mk("b1", session="b", seq=1, duration=100.0)])
    sched = {"serve_counter": 1, "last_served": {"a": 1, "b": 0}}
    out = _est(root, sched=sched)
    # Fair order: b1, a1, a2.
    assert out["b1"][0] == pytest.approx(100 * RATE, abs=0.05)
    assert out["a1"][0] == pytest.approx(200 * RATE, abs=0.05)
    assert out["a2"][0] == pytest.approx(500 * RATE, abs=0.05)
    # Admission order would put a1 first — it does not get the front ETA.


def test_processing_remaining_counts_for_every_queued_job(tmp_path, monkeypatch):
    # AC4: N steps ahead -> N distinct, non-overlapping estimates; the
    # current processing job's REMAINING estimate is the shared prefix.
    root = _setup(tmp_path, monkeypatch)
    now = 10_000.0
    _seed(root, [_mk("p", status="processing", duration=1000.0,
                     started_at=now - 100.0),
                 _mk("q1", duration=200.0, seq=1),
                 _mk("q2", duration=300.0, seq=2)])
    out = _est(root, now=now)
    # p est = 185, elapsed 100 -> remaining 85. q1: 85 + 37. q2: 85 + 37 + 55.5
    assert out["q1"][0] == pytest.approx(85 + 200 * RATE, abs=0.05)
    assert out["q2"][0] == pytest.approx(85 + 500 * RATE, abs=0.05)


def test_progress_fallback_for_started_at_less_processing(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    # No started_at (seeded/legacy state): remaining = est x (1 - progress).
    _seed(root, [_mk("p", status="processing", duration=1000.0, progress=0.5),
                 _mk("q", duration=100.0, seq=1)])
    out = _est(root)
    assert out["q"][0] == pytest.approx(92.5 + 18.5, abs=0.05)


def test_estimate_decreases_as_ahead_completes(tmp_path, monkeypatch):
    # AC5 "recomputed on every queue movement": finish A -> B's estimate
    # drops by A's full estimate.
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("a", duration=600.0), _mk("b", duration=300.0, seq=1)])
    before = _est(root)["b"][0]
    _seed(root, [_mk("a", status="done", duration=600.0)])
    after = _est(root)["b"][0]
    assert after < before
    assert after == pytest.approx(300 * RATE, abs=0.05)


def test_cancelled_job_leaves_the_prefix(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("a", duration=600.0), _mk("b", duration=300.0, seq=1)])
    before = _est(root)["b"][0]
    (root / "a" / "status.json").write_text(json.dumps(
        _mk("a", status="cancelled", duration=600.0)))
    after = _est(root)["b"][0]
    assert after < before


# --- AC5: negatives clamped, unknown durations, no re-probe ------------------

def test_negative_remaining_clamped_to_zero(tmp_path, monkeypatch):
    # A processing job far past its linear estimate contributes 0, not a
    # negative prefix that would shrink every queued estimate.
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("p", status="processing", duration=100.0, started_at=0.0),
                 _mk("q", duration=100.0, seq=1)])
    out = _est(root, now=10_000.0)
    assert out["q"][0] == pytest.approx(100 * RATE, abs=0.05)
    assert main._processing_remaining_seconds(
        _mk("p", status="processing", duration=100.0, started_at=0.0),
        RATE, now=10_000.0) == 0.0


def test_unknown_own_duration_yields_no_estimate(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("u", duration=None)])
    out = _est(root)
    assert "u" not in out


def test_unknown_ahead_uses_fallback_constant(tmp_path, monkeypatch):
    # Jobs ahead without a probed duration don't poison the tail: the
    # documented fallback constant is substituted in the prefix sum.
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("u", duration=None, seq=0),
                 _mk("v", duration=100.0, seq=1)])
    out = _est(root)
    assert "u" not in out
    assert out["v"][0] == pytest.approx(
        (main.DEFAULT_AUDIO_DURATION_SECONDS + 100.0) * RATE, abs=0.05)


def test_no_reprobe_or_audio_read_at_request_time(tmp_path, monkeypatch):
    # AC5: polls never re-run ffprobe and never open the audio file.
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 300.0)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    token = r.json()["token"]
    (root / token / "audio_source.wav").unlink()  # no audio to read at poll time

    def boom(path):
        raise AssertionError("duration probe re-run at request time")

    monkeypatch.setattr(main, "_probe_duration", boom)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_wait_seconds"] == pytest.approx(300 * RATE, abs=0.05)


def test_estimate_stable_across_polls_with_static_queue(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _seed(root, [_mk("s", duration=500.0)])
    v1 = _est(root)["s"][0]
    v2 = _est(root)["s"][0]
    assert math.isclose(v1, v2, abs_tol=1.0)


# --- AC3/AC5 HTTP surfaces ----------------------------------------------------

def test_202_admission_response_carries_estimate(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    assert r.status_code == 202
    body = r.json()
    assert body["estimated_wait_seconds"] == pytest.approx(200 * RATE, abs=0.05)
    assert body["estimated_completion_at"].endswith("Z")
    time.strptime(body["estimated_completion_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_queued_poll_carries_estimate(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    token = r.json()["token"]
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["status"] == "queued"
    assert body["estimated_wait_seconds"] == pytest.approx(200 * RATE, abs=0.05)
    assert body["estimated_completion_at"].endswith("Z")
    # ...and it is never persisted: raw state carries no ETA fields.
    st = json.loads((root / token / "status.json").read_text())
    assert "estimated_wait_seconds" not in st
    assert "estimated_completion_at" not in st


def test_polling_queued_across_queue_change_is_honored(tmp_path, monkeypatch):
    # AC5: a client's polled estimate stays current without any client
    # hint — a new job admitted ahead (earlier seq, other session RR)
    # changes the fair order and thus the recomputed wait.
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    r1 = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                     headers={"X-Session": "s1"})
    t1 = r1.json()["token"]
    wait1 = client.get(f"/v1/jobs/{t1}").json()["estimated_wait_seconds"]
    assert wait1 == pytest.approx(200 * RATE, abs=0.05)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 800.0)
    r2 = client.post("/v1/jobs", files={"file": ("b.wav", AUDIO, "audio/wav")},
                     headers={"X-Session": "s2"})
    wait1_after = client.get(f"/v1/jobs/{t1}").json()["estimated_wait_seconds"]
    # RR: s2's job is served after s1's (pointer advanced to s2 after t1),
    # so t1's own estimate stands — but the NEW job got an honest estimate.
    body2 = client.get(f"/v1/jobs/{r2.json()['token']}").json()
    assert body2["estimated_wait_seconds"] >= wait1_after


# --- Session job listing surface (card t_f97d12a8, stacked) -------------------

def test_listing_queued_entries_carry_eta_matching_the_poll(tmp_path, monkeypatch):
    # AC3 "in the session job listing": GET /v1/jobs entries for queued
    # jobs carry the SAME live estimated_* values as GET /v1/jobs/{token},
    # chained by fair position (one shared queue snapshot).
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    t1 = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                     headers={"X-Session": "s1"}).json()["token"]
    t2 = client.post("/v1/jobs", files={"file": ("b.wav", AUDIO, "audio/wav")},
                     headers={"X-Session": "s1"}).json()["token"]
    listing = client.get("/v1/jobs", headers={"X-Session": "s1"}).json()
    by_token = {e["token"]: e for e in listing["jobs"]}
    # Within-session FIFO: t1 ahead of t2; estimates chain by own cost.
    w1 = by_token[t1]["estimated_wait_seconds"]
    w2 = by_token[t2]["estimated_wait_seconds"]
    assert w1 == pytest.approx(200 * RATE, abs=0.05)
    assert w2 == pytest.approx(400 * RATE, abs=0.05)
    assert by_token[t1]["queue_position"] == 1
    assert by_token[t2]["queue_position"] == 2
    # Agreement with the poll endpoint (same snapshot, same numbers).
    poll2 = client.get(f"/v1/jobs/{t2}").json()
    assert poll2["estimated_wait_seconds"] == w2
    assert poll2["estimated_completion_at"] == by_token[t2]["estimated_completion_at"]
    assert by_token[t2]["estimated_completion_at"].endswith("Z")


def test_listing_eta_absent_when_rate_zero(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rate=0)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                headers={"X-Session": "s1"})
    listing = client.get("/v1/jobs", headers={"X-Session": "s1"}).json()
    entry = listing["jobs"][0]
    assert entry["status"] == "queued"
    assert "queue_position" in entry  # rank survives; only ETA is disabled
    assert "estimated_wait_seconds" not in entry
    assert "estimated_completion_at" not in entry


def test_listing_non_queued_entries_have_no_eta(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "_probe_duration", lambda p: 200.0)
    client = TestClient(main.app)
    r = client.post("/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")},
                    headers={"X-Session": "s1"})
    token = r.json()["token"]
    client.delete(f"/v1/jobs/{token}")
    listing = client.get("/v1/jobs", headers={"X-Session": "s1"}).json()
    entry = next(e for e in listing["jobs"] if e["token"] == token)
    assert entry["status"] == "cancelled"
    assert "estimated_wait_seconds" not in entry


def test_processing_poll_emits_explicit_nulls(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    tok = str(_uuid.uuid4())
    _seed(root, [_mk(tok, status="processing", duration=1000.0,
                     started_at=time.time())])
    client = TestClient(main.app)
    body = client.get(f"/v1/jobs/{tok}").json()
    assert body["status"] == "processing"
    assert body["estimated_wait_seconds"] is None
    assert body["estimated_completion_at"] is None


def test_finished_statuses_have_no_eta_fields(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    tok = str(_uuid.uuid4())
    _seed(root, [_mk(tok, status="done", duration=1000.0)])
    client = TestClient(main.app)
    body = client.get(f"/v1/jobs/{tok}").json()
    assert "estimated_wait_seconds" not in body
    assert "estimated_completion_at" not in body


def test_rate_zero_poll_emits_no_estimate(tmp_path, monkeypatch):
    # rate=0 disables EMISSION entirely: the fields are absent, not null —
    # a deployment that opted out should not see ETA keys at all.
    root = _setup(tmp_path, monkeypatch, rate=0)
    tok = str(_uuid.uuid4())
    _seed(root, [_mk(tok, duration=100.0)])
    client = TestClient(main.app)
    body = client.get(f"/v1/jobs/{tok}").json()
    assert body["status"] == "queued"
    assert "estimated_wait_seconds" not in body
    assert "estimated_completion_at" not in body


def test_unknown_duration_poll_emits_no_estimate(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    tok = str(_uuid.uuid4())
    _seed(root, [_mk(tok, duration=None)])
    client = TestClient(main.app)
    body = client.get(f"/v1/jobs/{tok}").json()
    assert body["estimated_wait_seconds"] is None
