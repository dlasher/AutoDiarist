"""Error codes on the job surface (card t_1799d07f) — API acceptance.

AC1: every FAILED job gets a non-empty error_code in status.json and in
both GET responses. AC4: the existing error/message fields are unchanged
(additive only). AC7: cancelled jobs carry NO code. AC3's classes are
exercised end to end through the real worker path, using the measured
raw strings.
"""
import json
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import main
from pipeline.transcribe import CrispasrError, TranscodeError

AUDIO = b"RIFF....WAVEfmt ....data...."

MEASURED_VRAM_CRASH = (
    "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of "
    "647530880 bytes\nnemotron: sched alloc encoder graph failed"
)


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    return root


def _post(client, session="alice", name="a.wav", payload=AUDIO):
    return client.post("/v1/jobs", files={"file": (name, payload, "audio/wav")},
                       headers={"X-Session": session})


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _wait_terminal(client, token, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = client.get(f"/v1/jobs/{token}").json().get("status")
        if st in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"job {token} never reached a terminal state")


def _boom(exc_factory):
    """Replace the transcribe call with one that raises the given error."""
    def boom(wav_path, **kw):
        raise exc_factory()
    return boom


# --- AC1 + AC3: every failure class, end to end through the real path -------


def test_measured_vram_crash_gets_out_of_memory_code(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: RuntimeError(MEASURED_VRAM_CRASH)))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        st = _state(root, token)
        assert st["status"] == "failed"
        # AC1: the code lands on the job record itself.
        assert st["error_code"] == "OUT_OF_MEMORY"
        # AC4/additive: the raw human string is UNCHANGED in error + message.
        assert "ggml_gallocr" in st["error"]
        assert "sched alloc encoder graph failed" in st["message"]
        # AC1: both GET responses carry the same code.
        body = client.get(f"/v1/jobs/{token}").json()
        assert body["error_code"] == "OUT_OF_MEMORY"
        listing = client.get("/v1/jobs").json()
        entry = next(j for j in listing["jobs"] if j["token"] == token)
        assert entry["error_code"] == "OUT_OF_MEMORY"


def test_timeout_gets_timeout_code(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: CrispasrError(
                            "crispasr timed out after 1800s: ['/usr/local/bin/crispasr']")))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        assert _state(root, token)["error_code"] == "TIMEOUT"


def test_engine_crash_gets_engine_crashed_code(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: CrispasrError(
                            "crispasr exited with code 139: segmentation fault")))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        assert _state(root, token)["error_code"] == "ENGINE_CRASHED"


def test_bad_upload_gets_unsupported_audio_code(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("MOCK_PIPELINE", "0")  # the mock path skips decode
    monkeypatch.setattr(main, "decode_audio",
                        lambda *a, **k: (_ for _ in ()).throw(TranscodeError(
                            "ffmpeg failed: Invalid data found when processing input")))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        assert _state(root, token)["error_code"] == "UNSUPPORTED_AUDIO"


def test_internal_error_gets_catch_all_not_nothing(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: ValueError("totally novel failure")))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        st = _state(root, token)
        assert st["error_code"] == "INTERNAL"
        assert "totally novel failure" in st["error"]


def test_empty_transcript_failure_also_carries_a_code(tmp_path, monkeypatch):
    # The direct update_state("failed", ...) site (no exception object)
    # must not be able to write a failed record without a code (AC1:
    # EVERY failed job).
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize", lambda *a, **k: [])
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        st = _state(root, token)
        assert st["error_code"] == "INTERNAL"
        assert "Empty transcript" in st["error"]  # semantics unchanged


def test_413_body_inside_a_job_classifies_input_too_large(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum size of {10 * 1024 * 1024} bytes")))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        assert _state(root, token)["error_code"] == "INPUT_TOO_LARGE"


# --- AC7: cancelled is NOT a failure — no code ------------------------------


def test_cancelled_job_carries_no_error_code(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert client.delete(f"/v1/jobs/{token}").status_code == 202
        st = _state(root, token)
        assert st["status"] == "cancelled"
        # The field may be present (never breaks pollers) but must carry
        # no code — a cancelled job did not fail.
        assert st.get("error_code") is None
        assert client.get(f"/v1/jobs/{token}").json().get("error_code") is None
        entry = next(j for j in client.get("/v1/jobs").json()["jobs"]
                     if j["token"] == token)
        assert entry.get("error_code") is None


# --- AC1: an unclassified pre-card record still surfaces a code on read -----


def test_pre_card_failed_record_is_classified_on_read(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        # Rewrite the record the way the PRE-card writer would have:
        # failed + raw string, no error_code on disk.
        st = _state(root, token)
        st["status"] = "failed"
        st["error"] = MEASURED_VRAM_CRASH
        st.pop("error_code", None)
        (root / token / "status.json").write_text(json.dumps(st))
        assert client.get(f"/v1/jobs/{token}").json()["error_code"] == "OUT_OF_MEMORY"


# --- AC4: additive-only; old fields and statuses unchanged ------------------


def test_in_flight_jobs_report_null_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")  # keep it queued
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        body = client.get(f"/v1/jobs/{token}").json()
        assert body["status"] == "queued"
        assert "error_code" in body and body["error_code"] is None
        entry = next(j for j in client.get("/v1/jobs").json()["jobs"]
                     if j["token"] == token)
        assert "error_code" in entry and entry["error_code"] is None


def test_additive_only_old_fields_and_statuses_unchanged(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "done"
        st = _state(root, token)
        assert st["status"] == "done"
        assert st.get("error_code") is None


def test_absent_job_404_body_unchanged(tmp_path, monkeypatch):
    # 404 semantics are frozen for existing clients: detail string, no
    # invented code (the card scopes codes to FAILED jobs only).
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        resp = client.get("/v1/jobs/11111111-1111-1111-1111-111111111111")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Job not found"


def test_failed_job_retains_its_input_audio(tmp_path, monkeypatch):
    """Card follow-on (measured, not assumed): a failed job keeps the
    uploaded bytes for debugging/reprocessing until ordinary retention
    sweeps it — the failure path must never delete the evidence."""
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_and_diarize",
                        _boom(lambda: RuntimeError(MEASURED_VRAM_CRASH)))
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        assert _wait_terminal(client, token) == "failed"
        assert (root / token / "audio_source.wav").exists()
