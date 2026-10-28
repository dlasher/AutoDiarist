"""Session-scoped HISTORY view + CSV/JSON export (card t_a8a71898, AC B1-B5).

GET /v1/history is a DERIVED view (decision B1): it reads the same on-disk
per-job status.json files and the same session filter that back GET /v1/jobs
— no parallel history store. The window defaults to HISTORY_DEFAULT_LIMIT (30,
decision B2) and ?limit= clamps into [1, MAX_LIST_LIMIT] like the listing
precedent. The column set is pinned (decision B3): token, filename,
audio_sha256, status, created_at, completed_at, duration_seconds,
download_json, download_md, cache_hit — identical in CSV and JSON, with
cache_hit reserved for card C (B always emits the key, False pre-cache).

Harness note (same as test_jobs_listing.py): synthetic bytes + MOCK_PIPELINE +
QUEUE_WORKER_ENABLED=0, and terminal states are simulated by rewriting
status.json — the _state() helper here is that sibling's helper extended with
duration_seconds and audio_sha256 (AC B5: suite count only goes UP).
"""
import csv
import io
import json
import time

from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."

HISTORY_COLUMNS = [
    "token", "filename", "audio_sha256", "status", "created_at",
    "completed_at", "duration_seconds", "download_json", "download_md",
    "cache_hit",
]


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("SESSION_HEADER", "X-Session")
    return root


def _post(client, session, filename="clip.wav"):
    r = client.post("/v1/jobs", headers={"X-Session": session},
                    files={"file": (filename, AUDIO, "audio/wav")})
    assert r.status_code == 202, r.text
    return r.json()["token"]


def _state(root, token, session, filename="clip.wav", job_status="done",
           seq=1, created_at=None, duration_seconds=12.5,
           audio_sha256="a" * 64):
    """Simulate a finished job by writing the terminal status.json directly,
    the way the sibling listing tests do — extended with the card-A
    fingerprint and the duration column so every pinned field is real.
    audio_sha256=None simulates a pre-card-A legacy state."""
    job_dir = root / token
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "audio_source.wav").write_bytes(AUDIO)
    if job_status == "done":
        (job_dir / "output_result.json").write_text(json.dumps({"segments": []}))
        (job_dir / "markdown_report.md").write_text("# report\n")
    state = {
        "token": token,
        "status": job_status,
        "seq": seq,
        "filename": filename,
        "session_ident": main._hash_ident(f"session:{session}"),
        "created_at": created_at if created_at is not None else time.time(),
        "queued_at": time.time(),
        "duration_seconds": duration_seconds,
        "progress": 1.0,
        "message": "simulated",
    }
    if audio_sha256 is not None:
        state["audio_sha256"] = audio_sha256
    (job_dir / "status.json").write_text(json.dumps(state))
    return token


def _history(client, session, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = "/v1/history" + (f"?{query}" if query else "")
    r = client.get(url, headers={"X-Session": session})
    assert r.status_code == 200, r.text
    return r


# --- AC B4: one tenant's history is never visible to another ----------------

def test_history_is_session_scoped(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tok_a = _post(client, "user-a", "a.wav")
        _state(root, tok_a, "user-a")
        tok_b = _post(client, "user-b", "b.wav")
        _state(root, tok_b, "user-b", job_status="cancelled")

        rows_a = _history(client, "user-a").json()["history"]
        rows_b = _history(client, "user-b").json()["history"]
        assert [r["token"] for r in rows_a] == [tok_a]
        assert [r["token"] for r in rows_b] == [tok_b]


# --- AC B2: newest-first order + default window of 30 ------------------------

def test_order_newest_first_and_default_window_30(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        now = time.time()
        tokens = []
        for i in range(35):
            tok = _post(client, "user-a", f"c{i}.wav")
            _state(root, tok, "user-a", seq=i, created_at=now + i)
            tokens.append(tok)

        body = _history(client, "user-a").json()
        got = [r["token"] for r in body["history"]]
        assert len(got) == main.HISTORY_DEFAULT_LIMIT == 30
        # newest-first, identical to the listing (created_at desc)
        assert got == list(reversed(tokens))[:30]


# --- AC B2: ?limit= clamps into [1, MAX_LIST_LIMIT], never 4xx ---------------

def test_limit_clamps_without_error(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        for i in range(3):
            _state(root, _post(client, "user-a", f"c{i}.wav"), "user-a", seq=i)
        for raw in (0, -5, main.MAX_LIST_LIMIT + 50):
            r = client.get(f"/v1/history?limit={raw}",
                           headers={"X-Session": "user-a"})
            assert r.status_code == 200, f"limit={raw} -> {r.status_code}"
        body = _history(client, "user-a", limit=2).json()
        assert len(body["history"]) == 2


# --- AC B1/B3: documented column set, JSON view ------------------------------

def test_json_rows_carry_the_pinned_column_set(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tok = _post(client, "user-a", "meeting.wav")
        _state(root, tok, "user-a", filename="meeting.wav")
        (root / tok / "status.json").touch()  # terminal clock = file mtime
        row = _history(client, "user-a").json()["history"][0]
        assert set(row) == set(HISTORY_COLUMNS)
        assert row["token"] == tok
        assert row["filename"] == "meeting.wav"
        assert row["status"] == "done"
        assert row["audio_sha256"] == "a" * 64
        assert row["duration_seconds"] == 12.5
        # ISO-8601 UTC stamps on both clocks (B3)
        assert row["created_at"].endswith("Z")
        assert row["completed_at"].endswith("Z")
        # download links are the existing per-job endpoints, relative URLs
        assert row["download_json"] == f"/v1/jobs/{tok}/download?format=json"
        assert row["download_md"] == f"/v1/jobs/{tok}/download?format=md"
        # non-finished jobs carry a null completion time, never a fake one
        tok2 = _post(client, "user-a", "pending.wav")
        _state(root, tok2, "user-a", job_status="queued")
        r2 = _history(client, "user-a").json()["history"][0]
        assert r2["token"] == tok2 and r2["completed_at"] is None


# --- AC B5/B1: CSV + JSON carry identical rows; pinned filenames -------------

def test_csv_and_json_agree_row_for_row(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tok = _post(client, "user-a", "sync.wav")
        _state(root, tok, "user-a")
        jrows = _history(client, "user-a", format="json").json()["history"]
        r = _history(client, "user-a", format="csv")
        assert r.headers["content-type"].startswith("text/csv")
        assert r.headers["content-disposition"] == \
            'attachment; filename="_session_history.csv"'
        assert client.get("/v1/history", headers={"X-Session": "user-a"}
                          ).headers["content-disposition"] == \
            'attachment; filename="_session_history.json"'
        records = list(csv.DictReader(io.StringIO(r.text)))
        assert len(records) == len(jrows) == 1
        for col in HISTORY_COLUMNS:
            # CSV cells are strings; compare against the JSON cell rendered
            # the same way csv writes it (booleans True/False, numbers, null
            # -> empty string).
            expected = jrows[0][col]
            got = records[0][col]
            if expected is None:
                assert got == "", col
            elif isinstance(expected, bool):
                assert got == str(expected), col
            else:
                assert got == str(expected), col


# --- AC B5: adversarial filenames survive the CSV round-trip -----------------

def test_csv_escaping_for_adversarial_filenames(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    nasty = 'we"ird, na\nme.wav'
    with TestClient(main.app) as client:
        tok = _post(client, "user-a", nasty)
        _state(root, tok, "user-a", filename=nasty)
        text = _history(client, "user-a", format="csv").text
        (row,) = list(csv.DictReader(io.StringIO(text)))
        assert row["filename"] == nasty
        assert row["token"] == tok


# --- AC B1 / decision B7: legacy jobs without the fingerprint render null ----

def test_legacy_job_missing_hash_yields_null_cell(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tok = _post(client, "user-a", "legacy.wav")
        _state(root, tok, "user-a", audio_sha256=None)
        row = _history(client, "user-a").json()["history"][0]
        assert row["audio_sha256"] is None


# --- AC B3: cache_hit key is reserved, False until card C stamps it ----------

def test_cache_hit_key_present_and_false_pre_cache(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        tok = _post(client, "user-a", "plain.wav")
        _state(root, tok, "user-a")
        row = _history(client, "user-a").json()["history"][0]
        assert row["cache_hit"] is False
        # card C may stamp the state later; the view reads it through.
        st = json.loads((root / tok / "status.json").read_text())
        st["cache_hit"] = True
        (root / tok / "status.json").write_text(json.dumps(st))
        row = _history(client, "user-a").json()["history"][0]
        assert row["cache_hit"] is True


# --- empty-session behavior ----------------------------------------------------

def test_history_empty_session_is_200(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        body = _history(client, "user-a").json()
        assert body["history"] == []
        r = _history(client, "user-a", format="csv")
        assert r.text.strip() == ",".join(HISTORY_COLUMNS)


# --- unknown format is the documented 422 (mirrors the download endpoint) ------

def test_unknown_format_rejected_422(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        r = client.get("/v1/history?format=xml", headers={"X-Session": "user-a"})
        assert r.status_code == 422
        assert "format=json or format=csv" in r.json()["detail"]
