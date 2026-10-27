"""Cancel a RUNNING job through the HTTP surface (card t_fef54150).

End-to-end over the real queue worker with the mock pipeline parked in the
transcribing stage (MOCK_TRANSCRIBE_SECONDS): the DELETE endpoint must
accept a ``processing`` job (202, never the old 409), the kill/cancel
signal must reach the live attempt through the process-local registry,
and the WORKER — not the handler — must finalize the terminal ``cancelled``
state with a journal that shows the cancel_request -> cancelled ordering.

Pins:
* AC1: DELETE on processing -> 202 {"status": "cancelling"}; the engine
      handle attached to the registry is terminated.
* AC2: the journal records the cancellation request before the terminal
      cancelled publish, and the terminal state is real (reached by
      stopping the attempt, results never written after the kill).
* AC3: no 409 remains for processing; done/failed keep theirs.
* AC4: cancel_requested is observable from the poll endpoint and the
      session listing while the kill is in flight.
* AC5: double-cancel is idempotent, the handler never races the worker's
      writes (all state writes under _queue_lock), and the queue keeps
      working afterwards (no lock leak).
* AC6: a boot requeue can never resurrect a cancel-requested job.
"""

import json
import os
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MOCK_PIPELINE", "1")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Lifespan-managed client: mock pipeline, worker on, transcribing
    stage parked for 3s so ``processing`` is observably cancellable."""
    from api import main

    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MOCK_TRANSCRIBE_SECONDS", "3")
    with TestClient(main.app) as c:
        yield c


def _submit(client) -> str:
    r = client.post("/v1/jobs", files={
        "file": ("cancel-me.wav", b"RIFF....WAVEfmt ....data....", "audio/wav")})
    assert r.status_code == 202
    return r.json()["token"]


def _poll_until(client, token, predicate, timeout=15.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{token}").json()
        if predicate(body):
            return body
        time.sleep(0.05)
    pytest.fail(f"job {token} never met {what} within {timeout}s (last: {body})")


def _wait_terminal(client, token, timeout=15.0):
    body = _poll_until(client, token,
                       lambda b: b["status"] in ("done", "failed", "cancelled"),
                       timeout, "a terminal status")
    assert body["status"] != "failed", \
        f"cancelled job was finalized as failed: {body}"
    return body


def _journal(token):
    from api import main
    path = main.JOBS_DIR / token / "status_journal.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- AC1/AC3: processing cancel is accepted and finalizes cancelled ----------


def test_delete_processing_returns_202_and_worker_finalises_cancelled(client):
    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "processing",
                what="processing stage")

    r = client.delete(f"/v1/jobs/{token}")
    assert r.status_code == 202, "processing jobs are cancellable now (no 409)"
    assert r.json()["status"] == "cancelling"

    body = _wait_terminal(client, token)
    assert body["status"] == "cancelled"
    assert "cancel" in (body.get("message") or "").lower()
    # Terminal state is real: the pipeline stopped mid-attempt — a cancelled
    # job never produced its result files.
    from api import main
    assert not (main.JOBS_DIR / token / "output_result.json").exists()

    # AC2: journal shows cancel_request -> cancelled, in that order, and
    # nothing ever moves backwards to a live/failed state after the kill.
    journal = _journal(token)
    messages = [str(j.get("message") or "") for j in journal]
    req_at = next((i for i, m in enumerate(messages)
                   if "cancellation requested" in m.lower()), None)
    assert req_at is not None, f"no cancel-request entry in journal: {journal}"
    cancelled_at = next(i for i, j in enumerate(journal)
                        if j["status"] == "cancelled")
    assert cancelled_at > req_at
    assert all(j["status"] != "failed" for j in journal[cancelled_at:])
    assert journal[-1]["status"] == "cancelled"


def test_cancel_via_registry_terminates_live_engine_handle(client):
    """The request handler reaches the ACTUAL live subprocess through the
    per-job registry and kills its whole process group — here simulated by
    attaching a real `sleep` as the job's engine handle."""
    from api import main

    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "processing",
                what="processing stage")

    proc = subprocess.Popen(["/bin/sleep", "120"], start_new_session=True)
    with main._cancel_registry_lock:
        ctrl = main._cancel_registry[token]
    assert ctrl is not None, "running job has no cancel control registered"
    ctrl.attach(proc)
    assert proc.poll() is None

    r = client.delete(f"/v1/jobs/{token}")
    assert r.status_code == 202
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None, "DELETE did not terminate the live engine"

    body = _wait_terminal(client, token)
    assert body["status"] == "cancelled"


# --- AC4: cancellation is observable while in flight -------------------------


def test_cancel_requested_visible_on_poll_and_listing(client):
    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "processing",
                what="processing stage")
    r = client.delete(f"/v1/jobs/{token}")
    assert r.status_code == 202 and r.json()["status"] == "cancelling"

    # Before the worker finalizes, the poll endpoint must show the request.
    body = _poll_until(client, token, lambda b: b.get("cancel_requested") is True,
                       what="cancel_requested on GET /v1/jobs/{token}")
    assert body["status"] in ("processing", "cancelled")

    _wait_terminal(client, token)
    listed = client.get("/v1/jobs").json()
    entry = next(j for j in listed["jobs"] if j["token"] == token)
    assert entry["status"] == "cancelled"
    assert entry.get("cancel_requested") is True


# --- orphan handling: processing without a live attempt -----------------------


def test_orphan_processing_finalises_cancelled_directly(tmp_path, monkeypatch):
    """A ``processing`` state file with no live worker in this process must
    not 409 forever and must not fake a kill: the handler finalizes
    ``cancelled`` itself, under the same _queue_lock the worker uses."""
    from api import main

    root = tmp_path / "jobs"
    root.mkdir()
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    with TestClient(main.app) as c:
        token = "11111111-1111-1111-1111-111111111111"
        (root / token).mkdir()
        state = {"token": token, "status": "processing", "stage": "transcribing",
                 "progress": 0.5, "filename": "orphan.wav", "seq": 1,
                 "created_at": time.time(), "started_at": time.time(),
                 "session_ident": "x", "cancel_requested": False}
        (root / token / "status.json").write_text(json.dumps(state))

        r = c.delete(f"/v1/jobs/{token}")
        assert r.status_code == 202
        assert r.json()["status"] == "cancelled"
        final = json.loads((root / token / "status.json").read_text())
        assert final["status"] == "cancelled"
        assert final["cancel_requested"] is True
        journal = _journal(token)
        assert journal[-1]["status"] == "cancelled"


# --- AC5: idempotent, race-free, no lock leak --------------------------------


def test_double_cancel_is_idempotent_and_queue_survives(client):
    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "processing",
                what="processing stage")
    r1 = client.delete(f"/v1/jobs/{token}")
    r2 = client.delete(f"/v1/jobs/{token}")
    assert r1.status_code == 202 and r2.status_code == 202
    assert r2.json()["status"] in ("cancelling", "cancelled")

    _wait_terminal(client, token)
    # Third delete lands on the terminal state — still consistent.
    r3 = client.delete(f"/v1/jobs/{token}")
    assert r3.status_code == 202
    assert r3.json()["status"] == "cancelled"

    # No lock leak: the queue serves the next job to done afterwards.
    t2 = _submit(client)
    _poll_until(client, t2, lambda b: b["status"] == "done", timeout=20,
                what="the follow-up job to finish")


def test_concurrent_cancels_never_corrupt_state(client):
    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "processing",
                what="processing stage")
    results = []

    def hammer():
        results.append(client.delete(f"/v1/jobs/{token}").status_code)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(code == 202 for code in results), results

    body = _wait_terminal(client, token)
    assert body["status"] == "cancelled"
    journal = _journal(token)
    assert sum(1 for j in journal if j["status"] == "cancelled") == 1


# --- 409 contract preserved for the other non-cancellable states -------------


def test_done_and_failed_still_409(client):
    token = _submit(client)
    _poll_until(client, token, lambda b: b["status"] == "done", timeout=20,
                what="done")
    r = client.delete(f"/v1/jobs/{token}")
    assert r.status_code == 409


# --- AC6: requeue/retry can never resurrect a cancel-requested job -----------


def test_worker_never_resurrects_orphan_finalised_cancel(tmp_path, monkeypatch):
    """The claim->register window: a DELETE that landed before the worker
    thread existed finalises the job as an orphan (cancelled + flag on
    disk). When the attempt then starts, the durable flag must short-
    circuit it before ANY pipeline work — no output files, no state
    resurrection, no second terminal journal line."""
    from api import main

    root = tmp_path / "jobs"
    (root / "44444444-4444-4444-4444-444444444444").mkdir(parents=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    token = "44444444-4444-4444-4444-444444444444"
    jd = root / token
    (jd / "audio_source.wav").write_bytes(b"RIFF....WAVEfmt ....data....")
    state = {"token": token, "status": "cancelled", "stage": "transcribing",
             "progress": 0.2, "filename": "x.wav", "seq": 1,
             "created_at": time.time(), "started_at": time.time(),
             "session_ident": "z", "cancel_requested": True,
             "message": "Cancelled by client request"}
    (jd / "status.json").write_text(json.dumps(state))
    journal_before = (jd / "status_journal.jsonl").read_text().splitlines() \
        if (jd / "status_journal.jsonl").exists() else []

    main.process_job_background(token, str(jd / "audio_source.wav"), "x.wav")

    final = json.loads((jd / "status.json").read_text())
    assert final["status"] == "cancelled"
    assert not (jd / "output_result.json").exists()
    assert not (jd / "markdown_report.md").exists()
    journal_after = (jd / "status_journal.jsonl").read_text().splitlines() \
        if (jd / "status_journal.jsonl").exists() else []
    assert journal_after == journal_before, \
        "straggler attempt re-published state on a terminal cancelled job"
    # Registry hygiene: the attempt registered and deregistered cleanly.
    with main._cancel_registry_lock:
        assert token not in main._cancel_registry


def test_boot_requeue_finalises_cancel_requested_instead_of_requeuing(tmp_path, monkeypatch):
    from api import main

    root = tmp_path / "jobs"
    root.mkdir()
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    with TestClient(main.app):  # lifespan runs the boot reclaim
        pass

    def write_state(token, **over):
        d = root / token
        d.mkdir()
        state = {"token": token, "status": "processing", "stage": "transcribing",
                 "progress": 0.4, "filename": "x.wav", "seq": 1,
                 "created_at": time.time(), "started_at": time.time(),
                 "session_ident": "y"}
        state.update(over)
        (d / "status.json").write_text(json.dumps(state))

    write_state("22222222-2222-2222-2222-222222222222")
    write_state("33333333-3333-3333-3333-333333333333",
                cancel_requested=True,
                message="Cancellation requested by client")
    # Reclaim on the NEXT boot: plain processing requeues (survives, as
    # before), the cancel-requested one must go terminal cancelled.
    c2 = TestClient(main.app)
    with c2:
        pass
    a = json.loads((root / "22222222-2222-2222-2222-222222222222" / "status.json").read_text())
    b = json.loads((root / "33333333-3333-3333-3333-333333333333" / "status.json").read_text())
    assert a["status"] == "queued"
    assert b["status"] == "cancelled", \
        f"boot requeue resurrected a cancel-requested job: {b}"
    journal_b = [json.loads(l) for l in
                 (root / "33333333-3333-3333-3333-333333333333" / "status_journal.jsonl").read_text().splitlines()]
    assert journal_b[-1]["status"] == "cancelled"
