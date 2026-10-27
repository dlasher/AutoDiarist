"""Queue-hardening acceptance tests (card t_daebf369, AC1-AC6).

Deterministic strategy: tests that need jobs to *stay* queued run with
QUEUE_WORKER_ENABLED=0; restart is simulated by opening a new TestClient
(fresh lifespan) against the same on-disk JOBS_DIR, which is the only
queue state the service is allowed to carry.
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."


def _setup(tmp_path, monkeypatch, worker="0", depth=None, per_session=None, poll="0.02"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", poll)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", str(depth if depth is not None else 50))
    if per_session is not None:
        monkeypatch.setenv("MAX_JOBS_PER_SESSION", str(per_session))
    else:
        monkeypatch.delenv("MAX_JOBS_PER_SESSION", raising=False)
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    return root


def _post(client, session=None, name="a.wav"):
    # V1.5 (card t_920fb7dc): the session identity header defaults to
    # X-Session. Authorization is only honored when SESSION_HEADER names it.
    headers = {"X-Session": session} if session else None
    return client.post(
        "/v1/jobs",
        files={"file": (name, AUDIO, "audio/wav")},
        headers=headers,
    )


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _write_state(root, token, st):
    (root / token / "status.json").write_text(json.dumps(st))


def _queued_tokens(root):
    return sorted(
        st["token"]
        for st in (
            _state(root, d.name) for d in root.iterdir() if d.is_dir()
        )
        if st.get("status") == "queued"
    )


def _wait_status(client, token, wanted=("done", "failed"), timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/v1/jobs/{token}").json()
        if last.get("status") in wanted:
            return last
        time.sleep(0.05)
    pytest.fail(f"job {token} never reached {wanted}, last: {last}")


# AC1 -------------------------------------------------------------------------


def test_ac1_depth_limit_rejects_third_with_503_retry_after(tmp_path, monkeypatch):
    """GIVEN MAX_QUEUE_DEPTH=2 WHEN 3 queued THEN POST 503+Retry-After, no enqueue."""
    root = _setup(tmp_path, monkeypatch, depth=2, per_session=10)
    with TestClient(main.app) as client:
        t1 = _post(client, "alice").json()["token"]
        t2 = _post(client, "bob").json()["token"]

        r3 = _post(client, "carol")
        assert r3.status_code == 503, r3.text
        assert "Retry-After" in r3.headers

        # Third job was never enqueued: exactly 2 queued on disk.
        assert len(_queued_tokens(root)) == 2

        # Previously queued jobs unaffected.
        for tok in (t1, t2):
            body = client.get(f"/v1/jobs/{tok}").json()
            assert body["status"] == "queued"

        # A cancelled slot frees admission (limit is about queued depth).
        assert client.delete(f"/v1/jobs/{t2}").status_code == 202
        assert _post(client, "carol").status_code == 202


# AC2 -------------------------------------------------------------------------


def test_ac2_per_session_admission_multiple_jobs_all_202(tmp_path, monkeypatch):
    """GIVEN MAX_JOBS_PER_SESSION=2 WHEN session POSTs 3 THEN all 3 are 202.

    V1.5 (card t_82bfbc15): the per-session rejection path is gone; the
    deprecated knob must not gate admission in any way.
    """
    _setup(tmp_path, monkeypatch, depth=50, per_session=2)
    with TestClient(main.app) as client:
        assert _post(client, "alice").status_code == 202
        assert _post(client, "alice").status_code == 202
        r3 = _post(client, "alice")
        assert r3.status_code == 202, r3.text

        # Other sessions keep their own identity buckets (never rejected).
        assert _post(client, "bob").status_code == 202

        # Distinct bearer tokens are distinct sessions even with same client IP.
        assert _post(client, "carol").status_code == 202


def test_ac2_no_session_admission_limit_by_default(tmp_path, monkeypatch):
    """MAX_JOBS_PER_SESSION unset -> there is no session cap any more.

    Replaces the retired "default limit is two" contract (card t_82bfbc15):
    five jobs from one session are all admitted and queued.
    """
    _setup(tmp_path, monkeypatch, depth=50, per_session=None)
    with TestClient(main.app) as client:
        for _ in range(5):
            assert _post(client, "alice").status_code == 202


def test_ac2_cancel_never_gates_admission(tmp_path, monkeypatch):
    """Cancel does not "free quota" because there is no quota (t_82bfbc15).

    Regression guard: with the deprecated knob at its most restrictive
    value, a session may submit again immediately — with or without a
    prior cancel.
    """
    _setup(tmp_path, monkeypatch, depth=50, per_session=1)
    with TestClient(main.app) as client:
        t1 = _post(client, "alice").json()["token"]
        assert _post(client, "alice").status_code == 202
        assert _post(client, "alice").status_code == 202
        assert client.delete(f"/v1/jobs/{t1}").status_code == 202
        assert _post(client, "alice").status_code == 202


def test_rejected_by_depth_leaves_no_enqueued_trace(tmp_path, monkeypatch):
    """A 503 from queue depth must not enqueue anything for the caller."""
    root = _setup(tmp_path, monkeypatch, depth=1, per_session=1)
    with TestClient(main.app) as client:
        t1 = _post(client, "alice").json()["token"]
        # bob hits the depth wall (queue full from alice's job)
        assert _post(client, "bob").status_code == 503
        assert len(_queued_tokens(root)) == 1
        assert client.delete(f"/v1/jobs/{t1}").status_code == 202
        # bob's rejected POST left no active job -> he can still enqueue
        assert _post(client, "bob").status_code == 202


def test_delete_unknown_and_malformed_token(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, depth=50, per_session=10)
    with TestClient(main.app) as client:
        assert client.delete(
            "/v1/jobs/00000000-0000-0000-0000-000000000000"
        ).status_code == 404
        assert client.delete("/v1/jobs/not-a-uuid").status_code == 422


# AC3 -------------------------------------------------------------------------


def test_ac3_cancel_queued_job(tmp_path, monkeypatch):
    """GIVEN queued job WHEN DELETE THEN 202 + cancelled + disk cleaned."""
    root = _setup(tmp_path, monkeypatch, depth=50, per_session=10)
    with TestClient(main.app) as client:
        tok = _post(client, "alice").json()["token"]

        r = client.delete(f"/v1/jobs/{tok}")
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "cancelled"

        body = client.get(f"/v1/jobs/{tok}").json()
        assert body["status"] == "cancelled"

        # Disk cleaned: input audio no longer retained for a cancelled job.
        assert not (root / tok / "audio_source.wav").exists()
        # No GPU work scheduled: no pipeline outputs, never left 'queued'.
        assert not (root / tok / "output_result.json").exists()
        assert not (root / tok / "markdown_report.md").exists()


def test_ac3_worker_never_claims_cancelled_job(tmp_path, monkeypatch):
    """A cancelled queued job must not be picked up when the worker starts."""
    root = _setup(tmp_path, monkeypatch, depth=50, per_session=10)
    with TestClient(main.app) as client:
        t_cancel = _post(client, "alice").json()["token"]
        t_keep = _post(client, "bob").json()["token"]
        assert client.delete(f"/v1/jobs/{t_cancel}").status_code == 202

    # Restart with the worker enabled: only t_keep should be processed.
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    with TestClient(main.app) as client:
        done = _wait_status(client, t_keep)
        assert done["status"] == "done"
        # give the worker a chance to (incorrectly) claim the cancelled job
        time.sleep(0.2)
        assert _state(root, t_cancel)["status"] == "cancelled"
        assert not (root / t_cancel / "output_result.json").exists()


# AC4 -------------------------------------------------------------------------


def test_ac4_delete_running_job_is_cancellable(tmp_path, monkeypatch):
    """Contract UPDATED by card t_fef54150: DELETE no longer 409s on
    processing. With the worker OFF this processing state is a dead
    attempt — no live engine to kill — so the handler finalises
    ``cancelled`` itself (202), journals the transition and flags the
    state. Unknown/bad tokens keep the GET-parity contract."""
    root = _setup(tmp_path, monkeypatch, depth=50, per_session=10)
    with TestClient(main.app) as client:
        tok = _post(client, "alice").json()["token"]
        # Simulate the worker having claimed the job (concurrency=1 mid-GPU).
        st = _state(root, tok)
        st.update({"status": "processing", "progress": 0.5})
        _write_state(root, tok, st)

        r = client.delete(f"/v1/jobs/{tok}")
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "cancelled"

        after = _state(root, tok)
        assert after["status"] == "cancelled"
        assert after["cancel_requested"] is True

        # Unknown token -> 404; bad format -> 422 (same contract as GET).
        assert client.delete(
            "/v1/jobs/00000000-0000-0000-0000-000000000000"
        ).status_code == 404
        assert client.delete("/v1/jobs/not-a-uuid").status_code == 422


# AC5 -------------------------------------------------------------------------


def test_ac5_queued_survives_restart_and_limits_revalidated(tmp_path, monkeypatch):
    """GIVEN restart mid-queue THEN queued survive; depth limits revalidated on boot."""
    root = _setup(tmp_path, monkeypatch, depth=2, per_session=10)

    # "Boot #1": enqueue 2 jobs with the worker parked, then shut down.
    with TestClient(main.app) as client:
        t1 = _post(client, "alice").json()["token"]
        t2 = _post(client, "alice").json()["token"]
    assert _state(root, t1)["status"] == "queued"
    assert _state(root, t2)["status"] == "queued"

    # "Boot #2": fresh app lifespan over the same disk; queue must be intact
    # and admission limits revalidated from persisted state (worker still off).
    with TestClient(main.app) as client:
        assert client.get(f"/v1/jobs/{t1}").json()["status"] == "queued"
        assert client.get(f"/v1/jobs/{t2}").json()["status"] == "queued"
        assert len(_queued_tokens(root)) == 2
        # depth=2 with 2 restored queued jobs -> new POST rejected, not enqueued
        assert _post(client, "bob").status_code == 503
        assert len(_queued_tokens(root)) == 2

    # "Boot #3": worker enabled drains the restored FIFO in order.
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    with TestClient(main.app) as client:
        assert _wait_status(client, t1)["status"] == "done"
        assert _wait_status(client, t2)["status"] == "done"
        assert (root / t1 / "output_result.json").exists()
        assert (root / t2 / "output_result.json").exists()


# AC6 -------------------------------------------------------------------------


def test_ac6_queue_position_consistent_with_fifo(tmp_path, monkeypatch):
    """queue_position exposed when queued and strictly consistent with pickup order."""
    _setup(tmp_path, monkeypatch, depth=50, per_session=10)
    with TestClient(main.app) as client:
        t1 = _post(client, "a", "first.wav").json()["token"]
        t2 = _post(client, "b", "second.wav").json()["token"]
        t3 = _post(client, "c", "third.wav").json()["token"]

        assert client.get(f"/v1/jobs/{t1}").json()["queue_position"] == 1
        assert client.get(f"/v1/jobs/{t2}").json()["queue_position"] == 2
        assert client.get(f"/v1/jobs/{t3}").json()["queue_position"] == 3

        # Cancelling position 1 recompresses the queue.
        assert client.delete(f"/v1/jobs/{t1}").status_code == 202
        assert client.get(f"/v1/jobs/{t2}").json()["queue_position"] == 1
        assert client.get(f"/v1/jobs/{t3}").json()["queue_position"] == 2

    # The worker must claim in ascending queue_position order.
    claimed = []
    real = main.process_job_background

    def spy(token, audio_path, filename):
        claimed.append(token)
        return real(token, audio_path, filename)

    monkeypatch.setattr(main, "process_job_background", spy)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    with TestClient(main.app) as client:
        assert _wait_status(client, t2)["status"] == "done"
        assert _wait_status(client, t3)["status"] == "done"
        # non-queued states must not carry a stale queue_position
        body = client.get(f"/v1/jobs/{t2}").json()
        assert body.get("queue_position") in (None, "missing")

    assert claimed == [t2, t3]
