"""V1.5 admission acceptance tests (card t_82bfbc15, AC1-AC6).

Admission changed from "reject a second session job with 429" to "accept
and queue": every submission that fits under MAX_QUEUE_DEPTH answers 202
with the token, the session id and the queue position. 503 (queue full)
is the ONLY rejection path; MAX_JOBS_PER_SESSION is a deprecated,
accepted-but-ignored knob with a startup warning.
"""
import json
import logging
import pathlib

from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."


def _setup(tmp_path, monkeypatch, depth=50, per_session=None, poll="0.02"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", poll)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", str(depth))
    if per_session is not None:
        monkeypatch.setenv("MAX_JOBS_PER_SESSION", str(per_session))
    else:
        monkeypatch.delenv("MAX_JOBS_PER_SESSION", raising=False)
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    return root


def _post(client, session, name="a.wav"):
    return client.post(
        "/v1/jobs",
        files={"file": (name, AUDIO, "audio/wav")},
        headers={"X-Session": session},
    )


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


# --- AC1: multiple active jobs per session, no 429 anywhere -----------------


def test_ac1_three_active_jobs_same_session_all_202(tmp_path, monkeypatch):
    # MAX_JOBS_PER_SESSION=1 is the worst-case deprecated knob setting:
    # even it must no longer reject a second or third active job.
    root = _setup(tmp_path, monkeypatch, per_session=1)
    with TestClient(main.app) as client:
        responses = [_post(client, "alice") for _ in range(3)]
    assert [r.status_code for r in responses] == [202, 202, 202]
    assert not any(r.status_code == 429 for r in responses)
    # all three really are active (queued) jobs of one session on disk
    states = [_state(root, r.json()["token"]) for r in responses]
    assert [s["status"] for s in states] == ["queued"] * 3
    assert len({s["session"] for s in states}) == 1


# --- AC2: the only rejection is depth; 503 + Retry-After + depth in detail ---


def test_ac2_depth_full_answers_503_with_retry_after_and_depth(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, depth=2)
    with TestClient(main.app) as client:
        assert _post(client, "alice").status_code == 202
        assert _post(client, "bob").status_code == 202
        third = _post(client, "carol")
        assert third.status_code == 503
        retry = third.headers.get("retry-after")
        assert retry is not None and int(retry) >= 1
        detail = third.json()["detail"]
        # detail names both the queue depth and the Retry-After value (AC: scope 5)
        assert "/2" in detail, f"depth not in detail: {detail!r}"
        assert retry in detail, f"Retry-After value not in detail: {detail!r}"


# --- AC3: deprecated knob warns at startup and never rejects -----------------


def test_ac3_deprecated_knob_warns_at_startup_and_never_rejects(
    tmp_path, monkeypatch, caplog
):
    _setup(tmp_path, monkeypatch, per_session=1)
    with (
        caplog.at_level(logging.WARNING, logger="api.main"),
        TestClient(main.app) as client,
    ):
        first = _post(client, "alice")
        second = _post(client, "alice")  # over the old limit: must still pass
    assert first.status_code == 202 and second.status_code == 202
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "MAX_JOBS_PER_SESSION" in m and "ignor" in m.lower() for m in warnings
    ), f"no startup warning that the knob is ignored: {warnings!r}"


# --- AC4: 429 appears nowhere in the service source --------------------------


def test_ac4_no_429_anywhere_in_service_source():
    api_dir = pathlib.Path(main.__file__).resolve().parent
    offenders = [
        str(p) for p in sorted(api_dir.rglob("*.py")) if "429" in p.read_text()
    ]
    assert not offenders, f"429 found in service source: {offenders}"


# --- AC5: every accepted POST carries token + session (and queue position) ---


def test_ac5_accepted_post_always_returns_token_and_session(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        for i in range(3):
            body = _post(client, "alice")
            assert body.status_code == 202
            payload = body.json()
            assert payload["token"], "accepted response missing token"
            assert payload["session"], "accepted response missing session"
            # admission answers 202 + queue position (card title / docs scope 6)
            assert payload["queue_position"] == i + 1


# --- AC6: the queued job's status keeps reporting its rank -------------------


def test_ac6_queued_job_status_reports_pollable_queue_position(
    tmp_path, monkeypatch
):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tokens = [_post(client, f"session-{i}").json()["token"] for i in range(3)]
        for i, token in enumerate(tokens, 1):
            body = client.get(f"/v1/jobs/{token}").json()
            assert body["status"] == "queued"
            assert body["queue_position"] == i
        # rank moves up as the head job leaves the queue
        client.delete(f"/v1/jobs/{tokens[0]}")
        assert client.get(f"/v1/jobs/{tokens[1]}").json()["queue_position"] == 1
