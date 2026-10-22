"""V1.5 job-listing acceptance tests (card t_f97d12a8, AC1-AC6).

GET /v1/jobs gives a client the session-scoped view of what it submitted:
the caller's own jobs only (filtered by the same hashed session_ident used
for admission), newest-first (created_at desc, seq tiebreak), capped by a
?limit= clamped to [1, MAX_LIST_LIMIT]. Terminal jobs appear (cancelled is
a state users want to see), failed jobs carry their error string, queued
jobs carry the fair-scheduled queue_position the worker actually follows.

Harness note (same as card t_4b40acfd): synthetic bytes + MOCK_PIPELINE +
QUEUE_WORKER_ENABLED=0 — admission is pure bookkeeping, the fake pipeline
only ever runs on the worker path we keep disabled. Jobs therefore stay
'queued' unless a test rewrites status.json to simulate terminal states.
"""
import json

from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."


def _setup(tmp_path, monkeypatch, depth=50):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", str(depth))
    monkeypatch.delenv("MAX_JOBS_PER_SESSION", raising=False)
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("MAX_PROCESSING_PER_SESSION", raising=False)
    return root


def _post(client, session, name="a.wav"):
    r = client.post(
        "/v1/jobs",
        files={"file": (name, AUDIO, "audio/wav")},
        headers={"X-Session": session},
    )
    assert r.status_code == 202, r.text
    return r.json()["token"]


def _list(client, session, params=None):
    headers = {"X-Session": session} if session is not None else {}
    r = client.get("/v1/jobs", headers=headers, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _set_state(root, token, **updates):
    st = _state(root, token)
    st.update(updates)
    (root / token / "status.json").write_text(json.dumps(st))
    return st


def _tokens(body):
    return [j["token"] for j in body["jobs"]]


# --- AC1: session A sees exactly its jobs, session B exactly its own --------


def test_ac1_listing_is_scoped_to_the_callers_session(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a_tokens = [_post(client, "user-a", f"a{i}.wav") for i in range(3)]
        b_tokens = [_post(client, "user-b", f"b{i}.wav") for i in range(2)]

        body_a = _list(client, "user-a")
        assert sorted(_tokens(body_a)) == sorted(a_tokens)
        assert body_a["count"] == 3 and body_a["total"] == 3

        body_b = _list(client, "user-b")
        assert sorted(_tokens(body_b)) == sorted(b_tokens)
        assert body_b["count"] == 2

        # A third session with no jobs gets an empty list, not an error.
        body_c = _list(client, "user-c")
        assert body_c["jobs"] == [] and body_c["count"] == 0


# --- AC2: each entry carries token/status/progress/stage/source/created_at --


def test_ac2_entries_carry_the_required_fields(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        token = _post(client, "user-a", "team-sync.wav")
        body = _list(client, "user-a")
        (job,) = body["jobs"]
        for key in ("token", "status", "progress", "stage", "source", "created_at"):
            assert key in job, f"{key} missing from listing entry"
        assert job["token"] == token
        assert job["status"] == "queued"
        assert job["progress"] == 0.0
        assert job["stage"] == "queued"
        assert job["source"] == "team-sync.wav"
        assert isinstance(job["created_at"], (int, float))


# --- AC3: a cancelled job appears with status == 'cancelled' ----------------


def test_ac3_cancelled_job_appears_in_the_listing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        keep = _post(client, "user-a", "keep.wav")
        gone = _post(client, "user-a", "gone.wav")
        r = client.delete(f"/v1/jobs/{gone}", headers={"X-Session": "user-a"})
        assert r.status_code == 202

        body = _list(client, "user-a")
        by_token = {j["token"]: j for j in body["jobs"]}
        assert set(by_token) == {keep, gone}
        assert by_token[gone]["status"] == "cancelled"


# --- AC4: ?limit= clamps to [1, 200]; default is 50 --------------------------


def test_ac4_limit_clamps_and_defaults(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, depth=300)
    with TestClient(main.app) as client:
        for i in range(55):
            _post(client, "user-a", f"a{i}.wav")

        # Default limit: 50 newest of 55.
        body = _list(client, "user-a")
        assert body["limit"] == 50
        assert body["count"] == 50 and body["total"] == 55

        # Out-of-range values clamp to the bounds, they never error.
        assert _list(client, "user-a", {"limit": 1000})["limit"] == main.MAX_LIST_LIMIT
        assert _list(client, "user-a", {"limit": 1000})["count"] <= main.MAX_LIST_LIMIT
        assert _list(client, "user-a", {"limit": 0})["limit"] == 1
        assert _list(client, "user-a", {"limit": -5})["limit"] == 1

        # In-range limits truncate newest-first.
        small = _list(client, "user-a", {"limit": 2})
        assert small["count"] == 2 and small["total"] == 55
        newest = _list(client, "user-a")
        assert _tokens(small) == _tokens(newest)[:2]


def test_ac4_non_integer_limit_is_rejected(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        _post(client, "user-a")
        r = client.get("/v1/jobs", headers={"X-Session": "user-a"}, params={"limit": "abc"})
        assert r.status_code == 422


# --- AC5: mixed terminal states, exact key set, newest-first ----------------


def test_ac5_mixed_states_newest_first_with_exact_keys(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        # Five jobs for A across terminal states + one queued; three for B.
        a = [_post(client, "user-a", f"a{i}.wav") for i in range(6)]
        b = [_post(client, "user-b", f"b{i}.wav") for i in range(3)]
        for i, token in enumerate(a):
            _set_state(
                root, token,
                status="done", progress=1.0, stage="done",
                created_at=1000.0 + i,
            )
        _set_state(root, a[4], status="failed", error="decode failed", created_at=1004.0)
        _set_state(root, a[5], status="cancelled", error=None, created_at=1005.0)
        for i, token in enumerate(b):
            _set_state(root, token, status="done", progress=1.0, stage="done",
                       created_at=2000.0 + i)

        body = _list(client, "user-a")
        assert all(t not in _tokens(body) for t in b)  # AC1 isolation again
        # Newest-first by created_at desc: a[5]=1005, a[4]=1004, ..., a[0]=1000.
        assert _tokens(body) == [a[5], a[4], a[3], a[2], a[1], a[0]]

        # A done entry carries EXACTLY the six base keys.
        entry = body["jobs"][2]
        assert entry["status"] == "done"
        assert set(entry) == {"token", "status", "progress", "stage", "source", "created_at"}

        # The failed entry adds the error string; cancelled stays in view.
        failed = next(j for j in body["jobs"] if j["status"] == "failed")
        assert failed["error"] == "decode failed"
        cancelled = next(j for j in body["jobs"] if j["status"] == "cancelled")
        assert "error" not in cancelled or cancelled.get("error") is None


def test_ac5_same_created_at_tiebreaks_by_seq_desc(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        early = _post(client, "user-a", "early.wav")   # seq 1
        late = _post(client, "user-a", "late.wav")     # seq 2
        _set_state(root, early, status="done", created_at=1234.5)
        _set_state(root, late, status="done", created_at=1234.5)

        assert _tokens(_list(client, "user-a")) == [late, early]  # seq desc wins


# --- AC6: queued jobs carry queue_position consistent with the scheduler ----


def test_ac6_queued_entries_expose_fair_schedule_position(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a1 = _post(client, "user-a", "a1.wav")
        _post(client, "user-b", "b1.wav")  # interleaved session for RR order
        a2 = _post(client, "user-a", "a2.wav")

        # Ground truth: the worker's simulated pickup order
        # (round-robin across sessions, FIFO inside one session).
        order = main._fair_queued_order(main._iter_states(), main._load_scheduler_state())

        body = _list(client, "user-a")
        by_token = {j["token"]: j for j in body["jobs"]}
        assert sorted(by_token) == sorted([a1, a2])
        for token in (a1, a2):
            assert by_token[token]["status"] == "queued"
            assert by_token[token]["queue_position"] == order.index(token) + 1

        # Done jobs never carry a queue position.
        root = main.JOBS_DIR
        _set_state(root, a2, status="done", progress=1.0, stage="done")
        after = {j["token"]: j for j in _list(client, "user-a")["jobs"]}
        assert "queue_position" not in after[a2]
        assert after[a1]["queue_position"] == 1  # a2 left the queue; rank holds

        # The single-job endpoint keeps its token-based behavior (AC: scope 6).
        r = client.get(f"/v1/jobs/{a1}")
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["queue_position"] == 1


# --- Session plumbing: cookie / body-id reuse lists the same session --------


def test_listing_follows_the_session_precedence_chain(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        # No header at all: server derives the session from the client IP and
        # returns the issued id in the body. Replaying that id via X-Session
        # must list the job (header > cookie > IP precedence, card t_920fb7dc).
        r = client.post("/v1/jobs", files={"file": ("x.wav", AUDIO, "audio/wav")})
        assert r.status_code == 202
        issued = r.json()["session"]
        assert issued

        body = _list(client, issued)
        assert _tokens(body) == [r.json()["token"]]

        # Without the id, a brand-new anonymous/IP session sees nothing.
        assert _list(client, "someone-else")["jobs"] == []
