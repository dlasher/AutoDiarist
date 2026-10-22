"""V1.5 fair-share scheduling acceptance tests (card t_4b40acfd, AC1-AC5).

The worker claims the next job round-robin across sessions (each session's
own jobs keep FIFO order by seq), sessions at MAX_PROCESSING_PER_SESSION
in-flight jobs are skipped, and queue_position reports the rank in that
fair execution order rather than raw admission order. The round-robin
pointer lives in JOBS_DIR/_scheduler.json so it survives a restart.

Deterministic strategy: most tests drive main.claim_next_job() directly
against seeded on-disk state (worker OFF); one gated-pipeline test drives
the real worker loop end-to-end. No GPU is required anywhere.
"""
import json
import time

from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."


def _setup(tmp_path, monkeypatch, worker="0", cap=None, poll="0.02"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", poll)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 10 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    if cap is not None:
        monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", str(cap))
    else:
        monkeypatch.delenv("MAX_PROCESSING_PER_SESSION", raising=False)
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


def _write_state(root, state):
    job_dir = root / state["token"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(state))
    # Seed a real (tiny) audio file so a claim that reaches the mock
    # pipeline can actually complete.
    (job_dir / "audio_source.wav").write_bytes(AUDIO)


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


def _claim_token(root):
    """Run one scheduler claim cycle; returns the claimed token or None."""
    job = main._claim_next_queued()
    return job["token"] if job else None


# --- AC1: round-robin across sessions, FIFO within a session -----------------


def test_ac1_round_robin_interleaves_sessions(tmp_path, monkeypatch):
    """alice(3 jobs), bob(1), carol(1): claim order is A B C A A, not A A A B C."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a1 = _post(client, "alice", "a1.wav").json()["token"]
        a2 = _post(client, "alice", "a2.wav").json()["token"]
        a3 = _post(client, "alice", "a3.wav").json()["token"]
        b1 = _post(client, "bob", "b1.wav").json()["token"]
        c1 = _post(client, "carol", "c1.wav").json()["token"]

    # Drive the scheduler directly (worker off): all jobs queued, fresh
    # pointer. Complete each claim before the next so the per-session cap
    # (AC2) doesn't block — cap enforcement is covered in its own tests.
    order = []
    for _ in range(5):
        tok = _claim_token(root)
        order.append(tok)
        st = _state(root, tok)
        st["status"] = "done"
        _write_state(root, st)
    assert order == [a1, b1, c1, a2, a3]
    # Nothing left queued: the claim cycle yields None, not a job.
    assert _claim_token(root) is None


def test_ac1_fifo_within_session_preserved(tmp_path, monkeypatch):
    """One session's own jobs are claimed in admission (seq) order."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    _write_state(root, _mk("a1", 1, "queued", "alice"))
    _write_state(root, _mk("a2", 2, "queued", "alice"))
    _write_state(root, _mk("a3", 3, "queued", "alice"))
    # Completing each claim between picks keeps the single session eligible
    # (the cap blocks a second concurrent in-flight job of the same session).
    claimed = []
    for _ in range(3):
        tok = _claim_token(root)
        claimed.append(tok)
        st = _state(root, tok)
        st["status"] = "done"
        _write_state(root, st)
    assert claimed == ["a1", "a2", "a3"]


# --- AC2: MAX_PROCESSING_PER_SESSION enforced by the scheduler ---------------


def test_ac2_cap_blocks_second_inflight_of_same_session(tmp_path, monkeypatch):
    """a1 PROCESSING, a2/b1 QUEUED, cap=1: next claim is b1, not a2."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    _write_state(root, _mk("a1", 1, "processing", "alice"))
    _write_state(root, _mk("a2", 2, "queued", "alice"))
    _write_state(root, _mk("b1", 3, "queued", "bob"))

    assert _claim_token(root) == "b1"
    # b1 now in flight too; both sessions are at cap -> nothing is claimable
    # even though a2 is queued. This is the proof the cap is enforced by the
    # scheduler, not by the single-worker accident.
    assert _claim_token(root) is None


def test_ac2_cap_is_configurable(tmp_path, monkeypatch):
    """With cap=2 a session may run a second job while one is in flight."""
    root = _setup(tmp_path, monkeypatch, cap=2)
    with TestClient(main.app):
        pass
    _write_state(root, _mk("a1", 1, "processing", "alice"))
    _write_state(root, _mk("a2", 2, "queued", "alice"))
    _write_state(root, _mk("a3", 3, "queued", "alice"))
    _write_state(root, _mk("b1", 4, "queued", "bob"))

    # alice has 1 in flight < cap 2 and the FIFO head, so a2 starts.
    assert _claim_token(root) == "a2"
    # alice now at cap (2 processing): b1 must be picked over a3.
    assert _claim_token(root) == "b1"
    assert _claim_token(root) is None


def test_ac2_cap_invalid_values(tmp_path, monkeypatch):
    """Cap <= 0 or unparseable: warn and clamp to the default 1, no crash."""
    for bad in ("0", "-3", "not-a-number", ""):
        monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", bad)
        assert main._max_processing_per_session() == 1, bad
    monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", "2")
    assert main._max_processing_per_session() == 2


# --- AC3: queue_position = rank in the actual fair execution order -----------


def test_ac3_position_is_fair_rank_not_admission_order(tmp_path, monkeypatch):
    """alice a1,a2 admitted first, then bob b1, carol c1: fair order is
    a1 b1 c1 a2, so a2 sits at position 4 despite being the 2nd arrival."""
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a1 = _post(client, "alice", "a1.wav")
        a2 = _post(client, "alice", "a2.wav")
        b1 = _post(client, "bob", "b1.wav")
        c1 = _post(client, "carol", "c1.wav")
        t_a1, t_a2 = a1.json()["token"], a2.json()["token"]
        t_b1, t_c1 = b1.json()["token"], c1.json()["token"]
        # Admission-time positions (snapshot at each POST).
        assert a1.json()["queue_position"] == 1
        assert a2.json()["queue_position"] == 2
        # After all four are queued, polling reflects the fair order.
        assert client.get(f"/v1/jobs/{t_a1}").json()["queue_position"] == 1
        assert client.get(f"/v1/jobs/{t_b1}").json()["queue_position"] == 2
        assert client.get(f"/v1/jobs/{t_c1}").json()["queue_position"] == 3
        assert client.get(f"/v1/jobs/{t_a2}").json()["queue_position"] == 4

        # The claim order must equal the rank order the poll reported.
        # Worker is disabled, so direct claims drive the scheduler here.
        assert main._claim_next_queued()["token"] == t_a1
        assert main._claim_next_queued()["token"] == t_b1

        # Polling recomputes positions from the REMAINING queue: c1 (never
        # served) now outranks alice's a2 (already served once).
        assert client.get(f"/v1/jobs/{t_c1}").json()["queue_position"] == 1
        assert client.get(f"/v1/jobs/{t_a2}").json()["queue_position"] == 2
        # processing jobs carry no queue position
        assert client.get(f"/v1/jobs/{t_a1}").json()["queue_position"] is None


def test_ac3_single_session_stays_plain_fifo(tmp_path, monkeypatch):
    """One session with N jobs: fairness degenerates to that session's FIFO."""
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        toks = [_post(client, "solo", f"s{i}.wav").json()["token"] for i in range(4)]
        for rank, tok in enumerate(toks, start=1):
            assert client.get(f"/v1/jobs/{tok}").json()["queue_position"] == rank
        # Cancelling the head recompresses positions.
        assert client.delete(f"/v1/jobs/{toks[0]}").status_code == 202
        for rank, tok in enumerate(toks[1:], start=1):
            assert client.get(f"/v1/jobs/{tok}").json()["queue_position"] == rank


# --- AC4: the round-robin pointer survives a restart --------------------------


def test_ac4_persisted_pointer_decides_after_restart(tmp_path, monkeypatch):
    """alice served 3 times, bob once before 'restart': b2 next, not a3."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    # alice a1 a2 a3 (seq 1..3), bob b1 b2 (seq 4..5). Serve a1, b1, a2 and
    # complete them; counters become alice=3, bob=2.
    _write_state(root, _mk("a1", 1, "queued", "alice"))
    _write_state(root, _mk("a2", 2, "queued", "alice"))
    _write_state(root, _mk("a3", 3, "queued", "alice"))
    _write_state(root, _mk("b1", 4, "queued", "bob"))
    _write_state(root, _mk("b2", 5, "queued", "bob"))
    for expected in ("a1", "b1", "a2"):
        tok = _claim_token(root)
        assert tok == expected
        st = _state(root, tok)
        st["status"] = "done"
        _write_state(root, st)

    sched = json.loads((root / "_scheduler.json").read_text())
    assert sched["last_served"]["alice"] == 3
    assert sched["last_served"]["bob"] == 2

    # 'Restart' = read the scheduler state from disk again (exactly what a
    # fresh process does; claim_next_job never trusts an in-memory copy).
    # With the pointer: bob was served at 2 < alice at 3, so b2 runs before
    # a3. With a lost pointer both tie at 0 and the seq tie-break would
    # wrongly pick a3 (seq 3 < 5) -- starving bob behind alice's backlog.
    assert _claim_token(root) == "b2"
    assert _claim_token(root) == "a3"


def test_ac4_restart_then_new_worker_claims_fairly(tmp_path, monkeypatch):
    """End-to-end: claims from a fresh lifespan honour the on-disk pointer."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    _write_state(root, _mk("a3", 3, "queued", "alice"))
    _write_state(root, _mk("b2", 5, "queued", "bob"))
    _save = main._load_scheduler_state()
    _save["last_served"] = {"alice": 3, "bob": 2}
    _save["serve_counter"] = 3
    main._save_scheduler_state(_save)

    claimed = []
    real = main.process_job_background

    def spy(token, audio_path, filename):
        claimed.append(token)
        return real(token, audio_path, filename)

    monkeypatch.setattr(main, "process_job_background", spy)
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    with TestClient(main.app):
        deadline = time.time() + 30
        while time.time() < deadline and len(claimed) < 2:
            time.sleep(0.05)
        # (tokens here are short labels, not UUIDs, so they bypass GET
        # validation intentionally — assert straight from the disk state)
        for tok in ("a3", "b2"):
            st = _state(root, tok)
            assert st["status"] in ("processing", "done"), st
    assert claimed == ["b2", "a3"]


def test_ac4_corrupt_scheduler_file_is_healed(tmp_path, monkeypatch, caplog):
    root = _setup(tmp_path, monkeypatch)
    (root / "_scheduler.json").write_text("{ this is not json")
    with caplog.at_level("WARNING"):
        sched = main._load_scheduler_state()
    assert sched == {"serve_counter": 0, "last_served": {}}
    assert any("scheduler state" in r.message.lower() for r in caplog.records)
    # A claim still works on the healed state.
    _write_state(root, _mk("a1", 1, "queued", "alice"))
    assert _claim_token(root) == "a1"


# --- AC5: preserved guarantees -------------------------------------------------


def test_ac5_boot_requeues_orphans_and_fairness_survives(tmp_path, monkeypatch):
    """A crash orphan (processing) is requeued at boot and keeps its rank."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    _write_state(root, _mk("a1", 1, "queued", "alice"))
    _write_state(root, _mk("a2", 2, "processing", "alice"))
    _write_state(root, _mk("b1", 3, "queued", "bob"))

    with TestClient(main.app):
        pass  # fresh lifespan: boot reclaim runs
    assert _state(root, "a2")["status"] == "queued"

    # a2 re-enters its session's FIFO at the head; with cap=1 the served
    # round-robin order (completing each claim so the cap stays clear) is
    # a1, b1, a2: first-round tie breaks by head seq, alice (seq 1) before
    # bob (seq 3); after serving alice, bob wins; a2 is alice's turn again.
    order = []
    for _ in range(3):
        tok = _claim_token(root)
        order.append(tok)
        st = _state(root, tok)
        st["status"] = "done"
        _write_state(root, st)
    assert order == ["a1", "b1", "a2"]


def test_ac5_never_two_processing_same_session(tmp_path, monkeypatch):
    """cap=1: claiming can never put two jobs of one session in flight."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app):
        pass
    for i in range(1, 6):
        _write_state(root, _mk(f"a{i}", i, "queued", "alice"))
    _write_state(root, _mk("b1", 6, "queued", "bob"))
    _write_state(root, _mk("c1", 7, "queued", "carol"))

    for _ in range(8):
        tok = _claim_token(root)
        processing = [
            s for s in main._iter_states() if s.get("status") == "processing"
        ]
        sessions = [s.get("session_ident") for s in processing]
        assert len(sessions) == len(set(sessions)), sessions
        if tok is None:
            break
        if processing:  # complete every in-flight job to keep claiming moving
            for s in processing:
                s["status"] = "done"
                _write_state(root, s)


def test_ac5_seq_monotonic_across_sessions(tmp_path, monkeypatch):
    """The global admission seq keeps rising whatever session submits."""
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        seqs = []
        for session in ("x", "y", "x", "z", "x"):
            tok = _post(client, session).json()["token"]
            seqs.append(_state(main.JOBS_DIR, tok)["seq"])
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_ac5_cancelled_and_done_jobs_leave_the_order(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a1 = _post(client, "alice", "a1.wav").json()["token"]
        a2 = _post(client, "alice", "a2.wav").json()["token"]
        b1 = _post(client, "bob", "b1.wav").json()["token"]
        assert client.delete(f"/v1/jobs/{a1}").status_code == 202
        assert client.get(f"/v1/jobs/{a1}").json().get("queue_position") is None
        # a1 leaves the queue; alice's a2 still leads bob's b1 (never served,
        # smaller head seq), positions compress accordingly.
        assert client.get(f"/v1/jobs/{a2}").json()["queue_position"] == 1
        assert client.get(f"/v1/jobs/{b1}").json()["queue_position"] == 2
        assert main._claim_next_queued()["token"] == a2
        assert main._claim_next_queued()["token"] == b1
