"""V1.5 server-issued session identity (card t_920fb7dc, AC1-AC7).

The server issues a random high-entropy session id on first contact, returns
it both as the `autodiarist_session` cookie and in the POST body ("session"),
and resolves identity with precedence: SESSION_HEADER header > session cookie
> client IP > "anonymous". Job state stores HASHED values only (domain-
separated digests, same convention as V1.0: header identities hash
"header:{name}:{value}", cookie identities "cookie:{value}", IPs "ip:{host}",
fallback "anonymous"); the session id is additionally persisted hashed under
the dedicated key `session_ident` ("session:{id}"). No server-side session
table: identity derives purely from what the client presents.
"""
import hashlib
import json
import re
import uuid

from fastapi.testclient import TestClient
from starlette.requests import Request

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."
COOKIE_NAME = "autodiarist_session"


def _setup(tmp_path, monkeypatch, per_session=None):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    monkeypatch.setenv("MAX_JOBS_PER_SESSION", str(per_session or 50))
    monkeypatch.delenv("SESSION_HEADER", raising=False)
    return root


def _post(client, headers=None):
    return client.post(
        "/v1/jobs", files={"file": ("a.wav", AUDIO, "audio/wav")}, headers=headers or {}
    )


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _h(raw):
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _request(headers=None, client_addr="10.1.2.3"):
    """Minimal ASGI Request for unit-level fallback-chain checks."""
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "headers": raw,
        "client": (client_addr, 5555) if client_addr else None,
    }
    return Request(scope)


# --- AC1: fresh high-entropy id in Set-Cookie AND the POST body ----------- #


def test_ac1_first_contact_issues_cookie_and_body_id(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        r = _post(client)
        assert r.status_code == 202
        body_id = r.json()["session"]
        assert body_id, "POST body must carry the issued session id"
        assert client.cookies.get(COOKIE_NAME) == body_id
        raw_set = r.headers.get("set-cookie", "")
        assert f"{COOKIE_NAME}={body_id}" in raw_set


def test_ac1_issued_id_is_high_entropy_and_unique(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as c1, TestClient(main.app) as c2:
        id1 = _post(c1).json()["session"]
        id2 = _post(c2).json()["session"]
        assert re.fullmatch(r"[0-9a-f]{32}", id1), "session id must be 32 hex chars"
        uuid.UUID(id1)
        assert id1 != id2


# --- AC2: same cookie back => stable identity ----------------------------- #


def test_ac2_cookie_reuse_gives_stable_identity(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        t1 = _post(client).json()["token"]  # first contact: IP identity
        sid = client.cookies.get(COOKIE_NAME)
        t2 = _post(client).json()["token"]  # cookie comes back: cookie identity
        r3 = _post(client)
        t3 = r3.json()["token"]
        # Stable id: the server never re-rolls a presented session.
        assert r3.json()["session"] == sid
        s1, s2, s3 = _state(root, t1), _state(root, t2), _state(root, t3)
        assert s2["session"] == s3["session"] == _h(f"cookie:{sid}")
        # The dedicated session key groups ALL jobs of the issued session,
        # including the first-contact job.
        assert s1["session_ident"] == s2["session_ident"] == s3["session_ident"] == _h(
            f"session:{sid}"
        )


# --- AC3: header beats cookie; configured header name honored ------------- #


def test_ac3_default_header_x_session_beats_cookie(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        _post(client)
        sid = client.cookies.get(COOKIE_NAME)
        token = _post(client, headers={"X-Session": "manual-42"}).json()["token"]
        st = _state(root, token)
        assert st["session"] == _h("header:X-Session:manual-42")
        assert st["session_ident"] == _h("session:manual-42")
        assert st["session"] != _h(f"cookie:{sid}")


def test_ac3_authorization_honored_when_configured(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("SESSION_HEADER", "Authorization")
    with TestClient(main.app) as client:
        token = _post(client, headers={"Authorization": "Bearer alice"}).json()["token"]
        assert _state(root, token)["session"] == _h("header:Authorization:Bearer alice")


def test_ac3_authorization_ignored_by_default(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        token = _post(client, headers={"Authorization": "Bearer alice"}).json()["token"]
        st = _state(root, token)
        # Without SESSION_HEADER=Authorization, the bearer token is not an
        # identity source: identity falls to the client IP bucket.
        assert st["session"] == _h("ip:testclient")
        assert st["session"] != _h("header:Authorization:Bearer alice")


# --- AC4: IP fallback, then anonymous -------------------------------------- #


def test_ac4_ip_fallback_when_no_cookie_or_header(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        # TestClient presents client host "testclient".
        assert _state(root, token)["session"] == _h("ip:testclient")


def test_ac4_unit_fallback_chain():
    # cookie beats IP
    r = _request(headers={"cookie": f"{COOKIE_NAME}=cookie-7"})
    identity, session_raw, source = main._resolve_session(r)
    assert (identity, source) == (_h("cookie:cookie-7"), "cookie")
    assert session_raw == "cookie-7"
    # header beats cookie
    r = _request(headers={"x-session": "hdr-1", "cookie": f"{COOKIE_NAME}=cookie-7"})
    identity, session_raw, source = main._resolve_session(r)
    assert (identity, source) == (_h("header:X-Session:hdr-1"), "header")
    assert session_raw == "hdr-1"
    # no header/cookie -> IP bucket + fresh issuance
    r = _request(client_addr="203.0.113.9")
    identity, session_raw, source = main._resolve_session(r)
    assert (identity, source) == (_h("ip:203.0.113.9"), "ip")
    assert re.fullmatch(r"[0-9a-f]{32}", session_raw)
    # no client address at all -> anonymous bucket + fresh issuance
    r = _request(client_addr=None)
    identity, _, source = main._resolve_session(r)
    assert (identity, source) == (_h("anonymous"), "anonymous")


# --- AC5: hashed identity only in job state -------------------------------- #


def test_ac5_raw_session_id_never_lands_in_job_files(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        token = _post(client).json()["token"]
        sid = client.cookies.get(COOKIE_NAME)
        st = _state(root, token)
        assert st["session"] == _h("ip:testclient")
        assert st["session_ident"] == _h(f"session:{sid}")
        for path in (root / token).rglob("*"):
            if path.is_file():
                blob = path.read_text(errors="ignore")
                assert sid not in blob, f"raw session id leaked into {path}"


# --- AC6: two sessions distinguishable, quota per-person ------------------- #


def test_ac6_two_sessions_distinguishable_and_quota_independent(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, per_session=1)
    with TestClient(main.app) as c1, TestClient(main.app) as c2:
        r1 = _post(c1, headers={"X-Session": "user-a"})
        r2 = _post(c2, headers={"X-Session": "user-b"})
        assert r1.status_code == 202 and r2.status_code == 202
        s1, s2 = _state(root, r1.json()["token"]), _state(root, r2.json()["token"])
        assert s1["session"] != s2["session"]
        assert s1["session_ident"] != s2["session_ident"]
        # Quota is per session, not per IP: user-a's second job is rejected
        # while user-b sits in its own bucket.
        r3 = _post(c1, headers={"X-Session": "user-a"})
        assert r3.status_code == 429


# --- AC7: stateless: no server-side session store -------------------------- #


def test_ac7_no_session_store_beyond_job_dirs(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as c1, TestClient(main.app) as c2:
        t1 = _post(c1).json()["token"]
        t2 = _post(c2).json()["token"]
        # Nothing but per-token job dirs exists under JOBS_DIR.
        assert sorted(p.name for p in root.iterdir()) == sorted([t1, t2])
        # Identity is derived purely from what the client presents: a brand
        # new client gets the IP bucket, never a server-side lookup.
        fresh = TestClient(main.app)
        t3 = _post(fresh).json()["token"]
        assert _state(root, t3)["session"] == _h("ip:testclient")
