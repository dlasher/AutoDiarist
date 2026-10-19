import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ["MOCK_PIPELINE"] = "1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Lifespan-managed client over an isolated jobs dir with the queue worker on."""
    from api import main

    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    with TestClient(main.app) as c:
        yield c


def _wait_done(client, token, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{token}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail(f"job {token} did not finish within {timeout}s")


def test_create_job_wav(client):
    response = client.post(
        "/v1/jobs",
        files={"file": ("test.wav", b"RIFF....WAVEfmt ....data....", "audio/wav")}
    )
    assert response.status_code == 202
    data = response.json()
    assert "token" in data
    assert len(data["token"]) == 36


def test_create_job_unsupported_type(client):
    response = client.post(
        "/v1/jobs",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 415


def test_invalid_token_format(client):
    response = client.get("/v1/jobs/not-a-uuid")
    assert response.status_code == 422


def test_unknown_token(client):
    response = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_mock_job_lifecycle(client):
    # Create
    res = client.post("/v1/jobs", files={"file": ("interview.mp3", b"ID3v2....", "audio/mpeg")})
    assert res.status_code == 202
    token = res.json()["token"]

    # Status (queue worker completes the job asynchronously)
    status_res = client.get(f"/v1/jobs/{token}")
    assert status_res.status_code == 200
    _wait_done(client, token)

    # Download
    dl_res = client.get(f"/v1/jobs/{token}/download")
    assert dl_res.status_code == 200
    # Matches pipeline.transcribe._mock_segments() output (the old "Mock
    # transcript text" fixture string no longer exists in the pipeline).
    assert "NVIDIA ASR pipeline demo" in dl_res.text
    md_res = client.get(f"/v1/jobs/{token}/download?format=md")
    assert md_res.status_code == 200
