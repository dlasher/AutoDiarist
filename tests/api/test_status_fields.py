"""Status-payload field gaps (card t_73254ea6).

AC1  done jobs carry completed_at >= started_at, near the real completion.
AC2  failed and cancelled jobs carry completed_at.
AC3  queued/processing jobs do NOT (absent or null — this card chose null,
     i.e. the key may be present with value None or missing; either passes).
AC4  invariant across every state: completed_at present <=> status terminal.
AC5  audio_duration_seconds is exposed and equals the probed audio length.
AC6  a boot-reclaimed (interrupted) job is distinguishable from the status
     payload alone: explicit marker fields + explicit message, status stays
     'queued' (no new scheduler-visible state — AC8 then only has to assert
     the reclaimed job is claimable, which AC7's run proves).
AC7  the reclaimed job is still claimable, still runs to done, progress
     never regresses.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

_SEGMENT = {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 2.0,
    "text": "hello world",
}


def _make_client(tmp_path, monkeypatch, worker: str, root=None):
    """Fresh TestClient over an isolated jobs dir; `worker` toggles the queue
    worker so the same dir can be re-opened to simulate a service restart
    (pattern from tests/api/test_queue_hardening.py)."""
    from api import main

    if root is None:
        root = tmp_path / "jobs"
        root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    return TestClient(main.app), main, root


def _upload(client, name="clip.wav"):
    resp = client.post("/v1/jobs", files={"file": (name, b"RIFF fake wav bytes", "audio/wav")})
    assert resp.status_code == 202
    return resp.json()["token"]


def _wait_final(client, token, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{token}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {token} never reached a terminal state")


def _instrument(main, monkeypatch, boom=None, probe=42.5, seen=None):
    """Real-branch fakes: decode writes a file, _probe_duration is pinned,
    the engine snapshots status.json at the transcribe boundary (so the test
    observes a PROCESSING state deterministically) or explodes (failure path)."""
    monkeypatch.setattr(main, "_probe_duration", lambda path: probe)

    def fake_decode(input_path, output_wav):
        from pathlib import Path

        Path(output_wav).write_bytes(b"RIFF fake wav")
        return output_wav

    def fake_transcribe(audio_path, mock=False, duration=None, progress=None,
                      cancel=None):
        if seen is not None:
            from pathlib import Path

            seen.append(json.loads((Path(main.JOBS_DIR) / seen[0] / "status.json").read_text()))
        if boom == "transcribe":
            raise RuntimeError("engine exploded")
        return [dict(_SEGMENT)]

    def fake_spellcheck(segments):
        from types import SimpleNamespace

        return SimpleNamespace(segments=segments)

    monkeypatch.setattr(main, "decode_audio", fake_decode)
    monkeypatch.setattr(main, "transcribe_and_diarize", fake_transcribe)
    monkeypatch.setattr(main, "spell_check_segments", fake_spellcheck)
    monkeypatch.setattr(main, "suggest_speaker_names", lambda segs, extra_evidence=None: segs)
    monkeypatch.setattr(main, "render_markdown", lambda segs, meta: "# report\n")


# ---------------------------------------------------------------- AC1

def test_done_job_has_completed_at(tmp_path, monkeypatch):
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="1")
    with client:
        token = _upload(client)
        body = _wait_final(client, token)
        assert body["status"] == "done"
        # AC1: completed_at present, >= started_at, sane window vs. now.
        assert isinstance(body.get("completed_at"), (int, float)), body
        assert body["completed_at"] >= body["started_at"]
        assert abs(time.time() - body["completed_at"]) < 15.0


# ---------------------------------------------------------------- AC2

def test_failed_job_has_completed_at(tmp_path, monkeypatch):
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="1")
    _instrument(main, monkeypatch, boom="transcribe")
    with client:
        token = _upload(client)
        body = _wait_final(client, token)
        assert body["status"] == "failed"
        assert isinstance(body.get("completed_at"), (int, float)), body
        assert body["completed_at"] >= body["started_at"]


def test_cancelled_job_has_completed_at(tmp_path, monkeypatch):
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="0")
    with client:
        token = _upload(client)
        resp = client.delete(f"/v1/jobs/{token}")
        assert resp.status_code == 202
        body = client.get(f"/v1/jobs/{token}").json()
        assert body["status"] == "cancelled"
        assert isinstance(body.get("completed_at"), (int, float)), body


# ---------------------------------------------------------------- AC3 + AC4

def test_completed_at_invariant_across_all_states(tmp_path, monkeypatch):
    """AC4 as a table: queued and processing -> no completed_at; done,
    failed, cancelled -> completed_at. The processing observation is
    captured deterministically from inside the engine fake, not by racing
    a poller."""
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="0")
    with client:
        # queued
        token = _upload(client)
        queued = client.get(f"/v1/jobs/{token}").json()
        assert queued["status"] == "queued"
        assert queued.get("completed_at") is None

        # processing (snapshot taken by the engine fake mid-run; the fake
        # appends its observation to the [token] box it was handed)
        box = [token]
        _instrument(main, monkeypatch, seen=box)
        main.process_job_background(token, str(main.JOBS_DIR / token / "audio_source.wav"), "clip.wav")
        assert len(box) == 2, "engine fake never ran"
        processing = box[1]
        assert processing["status"] == "processing"
        assert processing.get("completed_at") is None

        # done
        done = client.get(f"/v1/jobs/{token}").json()
        assert done["status"] == "done"
        assert isinstance(done.get("completed_at"), (int, float)), done

        # failed
        token2 = _upload(client, name="boom.wav")
        _instrument(main, monkeypatch, boom="transcribe")
        main.process_job_background(token2, str(main.JOBS_DIR / token2 / "audio_source.wav"), "boom.wav")
        failed = client.get(f"/v1/jobs/{token2}").json()
        assert failed["status"] == "failed"
        assert isinstance(failed.get("completed_at"), (int, float)), failed

        # cancelled (fresh client for a fresh queued job)
        token3 = _upload(client, name="gone.wav")
        client.delete(f"/v1/jobs/{token3}")
        cancelled = client.get(f"/v1/jobs/{token3}").json()
        assert cancelled["status"] == "cancelled"
        assert isinstance(cancelled.get("completed_at"), (int, float)), cancelled


# ---------------------------------------------------------------- AC5

def test_audio_duration_seconds_exposed_and_equals_probe(tmp_path, monkeypatch):
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="0")
    monkeypatch.setattr(main, "_probe_duration", lambda path: 42.5)
    with client:
        token = _upload(client)
        body = client.get(f"/v1/jobs/{token}").json()
        assert body["duration_seconds"] == pytest.approx(42.5)
        assert body.get("audio_duration_seconds") == pytest.approx(42.5)

        # the decode-time refresh keeps the two fields equal (real branch)
        _instrument(main, monkeypatch, probe=42.5)
        main.process_job_background(token, str(main.JOBS_DIR / token / "audio_source.wav"), "clip.wav")
        done = client.get(f"/v1/jobs/{token}").json()
        assert done["status"] == "done"
        assert done["audio_duration_seconds"] == pytest.approx(done["duration_seconds"])
        assert done["audio_duration_seconds"] == pytest.approx(42.5)


def test_audio_duration_seconds_null_when_probe_fails(tmp_path, monkeypatch):
    """Unknown duration stays unknown: consistent with duration_seconds."""
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="0")
    monkeypatch.setattr(main, "_probe_duration", lambda path: 0.0)
    with client:
        token = _upload(client)
        body = client.get(f"/v1/jobs/{token}").json()
        assert body["duration_seconds"] is None
        assert body.get("audio_duration_seconds") is None


# ---------------------------------------------------------------- AC6 + AC7 + AC8

def test_reclaim_marks_interrupted_job_and_it_still_completes(tmp_path, monkeypatch):
    """Simulated crash/restart: a 'processing' job is reclaimed by the boot
    pass. Client-visible marker (status payload only): explicit
    interrupted_at/recovered_reason fields + explicit message, status stays
    'queued' (so no new scheduler-visible state exists to corrupt — AC8).
    The job must then be claimable and reach done with progress preserved."""
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)

    client_a, main, _ = _make_client(tmp_path, monkeypatch, worker="0", root=root)
    with client_a:
        token = _upload(client_a, name="long.wav")
        # Crash mid-run: stamp the persisted state as processing with real
        # progress, exactly what a killed worker leaves behind.
        state_file = root / token / "status.json"
        with main._queue_lock:
            st = json.loads(state_file.read_text())
            st.update(status="processing", stage="transcribing", progress=0.35,
                      started_at=time.time() - 30.0)
            state_file.write_text(json.dumps(st))

    # Restart 1: fresh lifespan over the same jobs dir with the worker held
    # off, so the reclaimed payload is read deterministically (no claim race).
    client_b, main, _ = _make_client(tmp_path, monkeypatch, worker="0", root=root)
    with client_b:
        body = client_b.get(f"/v1/jobs/{token}").json()
        # AC6: distinguishable from the payload alone.
        assert body["status"] == "queued"
        assert body["message"] == "Requeued after service restart"
        assert isinstance(body.get("interrupted_at"), (int, float)), body
        assert body.get("recovered_reason") == "service_restart"
        # progress must not have regressed at reclaim time
        assert body["progress"] == pytest.approx(0.35)

    # Restart 2: same dir again, worker on — the reclaimed job must be
    # claimable and run to completion.
    client_c, main, _ = _make_client(tmp_path, monkeypatch, worker="1", root=root)
    with client_c:
        # AC7: reclaimed job is still claimable and still completes.
        done = _wait_final(client_c, token)
        assert done["status"] == "done"
        assert done["progress"] >= 0.35
        assert isinstance(done.get("completed_at"), (int, float)), done
        # marker survives completion (it is history, not current state)
        assert done["interrupted_at"] == body["interrupted_at"]

        # AC8 (field-model): no distinct status was introduced, so the only
        # claimability assertion is that the reclaimed job was claimed — done
        # above. A job left in the marker state must be claimable too; that
        # is the same _claim_next_queued path with status 'queued'.
        assert done["status"] not in ("interrupted", "requeued")


def test_unreclaimed_marker_never_created_without_restart(tmp_path, monkeypatch):
    """interrupted_at/recovered_reason appear ONLY via the boot reclaim."""
    client, main, _ = _make_client(tmp_path, monkeypatch, worker="0")
    with client:
        token = _upload(client)
        body = client.get(f"/v1/jobs/{token}").json()
        assert body.get("interrupted_at") is None
        assert body.get("recovered_reason") is None
