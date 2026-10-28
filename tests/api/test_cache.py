"""Transcript-cache acceptance tests (card t_ac748379, AC C1-C8 + C-EN1..4).

Model recap (pinned by the card, supersedes the original deployment-knob
plan): the cache is PER-REQUEST OPT-IN via the X-Cache header on
POST /v1/jobs — one flag, BOTH directions (lookup-if-present AND
store-on-completion). CACHE_DISABLED is a pure kill switch: when truthy the
header is obeyed-as-absent service-wide, never an error. A malformed header
value is rejected loudly with 400 (never silently treated as off or on).

The key is the TRIPLE (audio_sha256, config_fp, session_ident): the same
audio with any output-affecting config change must MISS, and another tenant
with byte-identical audio must never see the first tenant's transcript.

Harness notes:
  - HIT tests use synthetic RIFF bytes and MAX_QUEUE_DEPTH=0 as proof that
    the cached path runs BEFORE the queue-depth check (AC C5/C3: a hit never
    enqueues, never occupies a worker, answers essentially immediately).
  - Fresh-run tests use real decodable WAV bytes (ffmpeg present, same
    skipif gate as test_audio_hash.py) and drive the pipeline through the
    existing MOCK_PIPELINE seam (process_job_background directly, like
    tests/api/test_llm_cleanup_seam.py) — no GPU work, no live worker loop.
"""

import io
import json
import os
import shutil
import sqlite3
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main

os.environ.setdefault("MOCK_PIPELINE", "1")

# Undecodable-but-valid-container junk: the admission hash degrades to null
# (card A behaviour) — never a cache key, never a crash (legacy-null arm).
AUDIO = b"RIFF....WAVEfmt ....data...."

# A real decodable clip: 16 kHz mono s16, so the pipeline's decode domain
# (ffmpeg -ar 16000 -ac 1 -f s16le) is an identity resample and the golden
# fingerprint is pure-Python sha256 over the frames — same construction as
# test_audio_hash.py's _golden/_wav_bytes pair.
PCM = bytes(bytearray((i * 37) % 256 for i in range(16000)))  # 1 s @ 16 kHz

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="cache fixtures need the decodable-audio admission hash",
)

CACHE_DB = "cache_index.sqlite"


def _wav_bytes(pcm, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


WAV = _wav_bytes(PCM)


def _setup(tmp_path, monkeypatch, **env):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("SESSION_HEADER", "X-Session")
    monkeypatch.delenv("CACHE_DISABLED", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return root


def _post(client, session, x_cache=None, filename="clip.wav", audio=None):
    headers = {"X-Session": session}
    if isinstance(x_cache, list):
        headers = [("X-Session", session)] + [
            ("X-Cache", v) for v in x_cache]
    elif x_cache is not None:
        headers["X-Cache"] = x_cache
    r = client.post("/v1/jobs", headers=headers,
                    files={"file": (filename, audio if audio is not None else AUDIO,
                                    "audio/wav")})
    return r


def _rows(root):
    db = Path(root) / CACHE_DB
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT sha, config_fp, session_ident, source_token, hit_count"
            " FROM cache_entries ORDER BY sha, config_fp, session_ident,"
            " source_token").fetchall()
    finally:
        conn.close()


def _state(root, token):
    return json.loads((root / token / "status.json").read_text(encoding="utf-8"))


def _seed_done_job(root, session, audio=WAV, x_cache="1", filename="clip.wav"):
    """Run the REAL admission + worker path for one job so the cache row is
    stored exactly the way production stores it (worker's done transition).
    Returns (token, sha). MOCK_PIPELINE is already set — no GPU work."""
    with TestClient(main.app) as client:
        r = _post(client, session, x_cache=x_cache, audio=audio,
                  filename=filename)
        assert r.status_code == 202, r.text
        token = r.json()["token"]
        sha = r.json()["audio_sha256"]
        assert sha, "seed fixture needs a decodable upload (non-null hash)"
    main.process_job_background(token, str(root / token / "audio_source.wav"),
                                filename)
    assert _state(root, token)["status"] == "done"
    return token, sha


# --- C-EN1: header parsing — normalise, accept the pinned set, reject loudly --

def test_malformed_x_cache_is_400_and_leaves_no_trace(tmp_path, monkeypatch):
    """AC C4d: ANY other value is rejected loudly with 400 naming the
    accepted list; never silently treated as off or on, never half-admitted."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        for bad in ("yes", "maybe", "on", "enabled", "0", "false", "2",
                    "no", "allow1", "true1"):
            r = _post(client, "u", x_cache=bad)
            assert r.status_code == 400, f"{bad!r} -> {r.status_code}"
            detail = str(r.json()["detail"]).lower()
            assert "1" in detail and "true" in detail and "allow" in detail
        # malformed requests never reach the queue: no job dirs left behind
        assert not [p for p in root.iterdir() if p.is_dir()]


def test_duplicate_header_lines_are_all_validated(tmp_path, monkeypatch):
    """C4d definite behaviour extends to repeated X-Cache lines: a junk
    instance among accepted ones is rejected (never first-wins on/off),
    and two accepted spellings simply mean ON."""
    _setup(tmp_path, monkeypatch, MAX_QUEUE_DEPTH="0")
    with TestClient(main.app) as client:
        r = _post(client, "u", x_cache=["1", "junk"])
        assert r.status_code == 400
        r = _post(client, "u", x_cache=["nope", "1"])
        assert r.status_code == 400
        # accepted spellings only: not a 400 — the request reaches the
        # queue path (503 here; no row exists either way)
        r = _post(client, "u", x_cache=["1", "ALLOW"])
        assert r.status_code == 503


def test_empty_header_is_off_like_absent(tmp_path, monkeypatch):
    """C-EN1: absent OR empty -> cache OFF, request path byte-for-byte today's."""
    _setup(tmp_path, monkeypatch, MAX_QUEUE_DEPTH="0")  # queue path 503s
    with TestClient(main.app) as client:
        r_absent = _post(client, "u")
        r_empty = _post(client, "u", x_cache="")
        r_ws = _post(client, "u", x_cache="   ")
        for r in (r_absent, r_empty, r_ws):
            assert r.status_code == 503  # today's queue path, untouched by cache
            assert r.json()["detail"].startswith("Queue is full")


def test_header_normalises_case_and_space(tmp_path, monkeypatch):
    """C-EN1 documented normalisation: case-insensitive, whitespace-trimmed.
    With MAX_QUEUE_DEPTH=0 (read at use time) the queue path 503s, so the
    only path that can answer 202 for the same audio is the cache lookup —
    each accepted spelling must therefore reach a HIT."""
    root = _setup(tmp_path, monkeypatch)
    _seed_done_job(root, "s1")
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")
    with TestClient(main.app) as client:
        for spelling in ("1", "true", "TRUE", " Allow ", "allow", "\tTRUE\t"):
            r = _post(client, "s1", x_cache=spelling, audio=WAV)
            assert r.status_code == 202, f"{spelling!r}: {r.status_code} {r.text}"
            assert r.json()["cache_hit"] is True


def test_kill_switch_makes_header_obeyed_as_absent(tmp_path, monkeypatch):
    """AC C4b: CACHE_DISABLED truthy forces off service-wide; the header is
    obeyed-as-absent, not an error. Nothing stored, nothing served. Read at
    USE time (per request) — same process, env flipped, no restart."""
    root = _setup(tmp_path, monkeypatch)
    _seed_done_job(root, "s1")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.status_code == 202 and r.json()["cache_hit"] is True
        # flip the kill switch in-process...
        monkeypatch.setenv("CACHE_DISABLED", "1")
        r2 = _post(client, "s1", x_cache="1", audio=WAV)
        assert r2.status_code == 202
        body = r2.json()
        assert body["cache_hit"] is False
        assert body["status"] == "queued"  # today's full path, not served
        assert _state(root, body["token"]).get("cache_opt_in") in (None, False)
        rows = _rows(root)
        assert len(rows) == 1 and rows[0][4] == 1  # nothing stored, no extra hit
    # switch off again (same process): the cached path is back — use-time read
    monkeypatch.delenv("CACHE_DISABLED", raising=False)
    with TestClient(main.app) as client:
        r3 = _post(client, "s1", x_cache="1", audio=WAV)
        assert r3.status_code == 202
        assert r3.json()["cache_hit"] is True


def test_kill_switch_makes_malformed_header_not_an_error(tmp_path, monkeypatch):
    """The 2x2 matrix cell (malformed header x kill switch ON): with the
    switch on the header is obeyed-as-absent (C-EN2), so a junk value must
    NOT 400 — validation is skipped along with the whole cache path."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("CACHE_DISABLED", "TRUE")
    with TestClient(main.app) as client:
        r = _post(client, "u", x_cache="yes")
        assert r.status_code == 202   # not 400
        assert r.json()["cache_hit"] is False


# --- C-EN2/C-EN4 + storage: the index, the stamp, and the store rule ----------

def test_absent_header_stores_nothing_and_uses_nothing(tmp_path, monkeypatch):
    """AC C4a: without the header the request path is exactly today's —
    the job runs normally, nothing is WRITTEN to the index, and the same
    audio re-posted with the flag still MISSes (no retroactive store)."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        r = _post(client, "s1", audio=WAV)  # no X-Cache
        assert r.status_code == 202 and r.json()["cache_hit"] is False
        tok = r.json()["token"]
    main.process_job_background(tok, str(root / tok / "audio_source.wav"), "clip.wav")
    assert _rows(root) == []                      # never stored
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")     # only a hit escapes 503
    with TestClient(main.app) as client:
        r2 = _post(client, "s1", x_cache="1", audio=WAV)
        assert r2.status_code == 503               # MISS -> today's queue path


def test_opt_in_stamps_state_and_row_shape(tmp_path, monkeypatch):
    """C-EN4: cache_opt_in is stamped at admission so the worker's done
    transition knows to store. The row is the pinned (sha, config_fp,
    session_ident, source_token) triple, inserted at COMPLETION only."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        tok = r.json()["token"]
        sha = r.json()["audio_sha256"]
        st = _state(root, tok)
        assert st["status"] == "queued" and st["cache_opt_in"] is True
        assert _rows(root) == []                   # never insert at admission
    main.process_job_background(tok, str(root / tok / "audio_source.wav"), "clip.wav")
    rows = _rows(root)
    assert len(rows) == 1
    got_sha, fp, sess, src, hits = rows[0]
    assert got_sha == sha and src == tok and hits == 0
    assert sess == main._hash_ident(f"session:{'s1'}")
    assert isinstance(fp, str) and len(fp) == 16 and \
        all(c in "0123456789abcdef" for c in fp)


def test_kill_switch_blocks_store_but_still_runs(tmp_path, monkeypatch):
    """C-EN2 store arm: a job admitted with the flag while the kill switch
    is on runs normally (no lookup, no error) but stores nothing."""
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("CACHE_DISABLED", "1")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.status_code == 202 and r.json()["cache_hit"] is False
        tok = r.json()["token"]
        assert _state(root, tok).get("cache_opt_in") in (None, False)
    main.process_job_background(tok, str(root / tok / "audio_source.wav"), "clip.wav")
    assert _rows(root) == []


# --- C1/C5/C6/C2: hit path, fingerprint change, visibility ---------------------

def test_hit_is_immediate_done_never_queued_never_worker(tmp_path, monkeypatch):
    """AC C3/C5/C6: the HIT answers before the queue check (MAX_QUEUE_DEPTH=0
    still 202s), lands in done state immediately with the explicit signal on
    the 202, the poll, and the job's own state, and carries the source."""
    root = _setup(tmp_path, monkeypatch)
    src, sha = _seed_done_job(root, "s1")
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["cache_hit"] is True
        assert body["status"] == "done"
        assert body["audio_sha256"] == sha
        hit = body["token"]
        assert hit != src
        p = client.get(f"/v1/jobs/{hit}")
        assert p.status_code == 200
        poll = p.json()
        assert poll["status"] == "done" and poll["cache_hit"] is True
        assert poll["cache_source_token"] == src
        st = _state(root, hit)
        assert st["cache_hit"] is True and st["cache_source_token"] == src
        # hit_count / last_hit_at advanced on the row
        row = next(x for x in _rows(root) if x[3] == src)
        assert row[4] == 1
        # explicit signal on the session listing (card A/B precedent entries)
        # and the history view's reserved cache_hit column (card B, C.2)
        lst = client.get("/v1/jobs", headers={"X-Session": "s1"}).json()
        entry = next(j for j in lst["jobs"] if j["token"] == hit)
        assert entry["cache_hit"] is True
        hist = client.get("/v1/history", headers={"X-Session": "s1"}).json()
        hrow = next(h for h in hist["history"] if h["token"] == hit)
        assert hrow["cache_hit"] is True


def test_config_input_change_misses(tmp_path, monkeypatch):
    """AC C1: changing ONE output-affecting config input changes config_fp,
    so the same audio + same session MISSes (a stale transcript can never
    be silently served after a config change)."""
    root = _setup(tmp_path, monkeypatch)
    _seed_done_job(root, "s1")
    before = main._cache_config_fp()
    monkeypatch.setenv("DIAR_METHOD", "none")
    after = main._cache_config_fp()
    assert before != after
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.status_code == 503  # MISS -> falls through to normal path
    # vocab file BYTES are a fingerprint input too: same DIAR_METHOD as the
    # stored row, different vocab content -> different config_fp.
    monkeypatch.delenv("DIAR_METHOD", raising=False)
    vocab = tmp_path / "other_vocab.txt"
    vocab.write_text("zzzplugh\n")
    monkeypatch.setenv("VOCAB_PATH", str(vocab))
    assert main._cache_config_fp() != before


def test_fingerprint_is_stable_and_restart_safe(tmp_path, monkeypatch):
    """C-EN/plan: the fingerprint derives from env + file content at use
    time, so a fresh process computes the SAME value (nothing memory-only
    dies on restart)."""
    _setup(tmp_path, monkeypatch)
    fp1 = main._cache_config_fp()
    fp2 = main._cache_config_fp()
    assert fp1 == fp2 and len(fp1) == 16


# --- C7: tenant isolation is structural (session_ident in the key) ------------

def test_cross_session_never_sees_other_tenants_result(tmp_path, monkeypatch):
    """AC C7: byte-identical audio from session B must MISS and can never
    observe A's token, state, or artifacts — even when A cached it."""
    root = _setup(tmp_path, monkeypatch)
    _seed_done_job(root, "s1")
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")
    with TestClient(main.app) as client:
        r = _post(client, "s2", x_cache="1", audio=WAV)
        assert r.status_code == 503          # MISS for the other tenant
    # B with no opt-in: today's path, no store under B either
    monkeypatch.delenv("MAX_QUEUE_DEPTH", raising=False)
    with TestClient(main.app) as client:
        r = _post(client, "s2", audio=WAV)
        assert r.status_code == 202 and r.json()["cache_hit"] is False
    assert {x[2] for x in _rows(root)} == {main._hash_ident("session:s1")}


# --- C8: served content is byte-identical; hits survive source eviction ------

def test_hit_artifacts_are_byte_identical_and_self_contained(tmp_path, monkeypatch):
    """AC C8 / plan C.8: the cached download is BYTE-IDENTICAL to the source
    job's output_result.json and markdown_report.md (the copies are real
    copies, not symlinks), and the hit record keeps working after the
    SOURCE job dir is deleted."""
    root = _setup(tmp_path, monkeypatch)
    src, _ = _seed_done_job(root, "s1")
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "0")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.json()["cache_hit"] is True
        hit = r.json()["token"]
    for fname in ("output_result.json", "markdown_report.md"):
        a, b = root / src / fname, root / hit / fname
        assert not b.is_symlink()
        assert a.read_bytes() == b.read_bytes(), fname
    # evict the source dir entirely (what the retention sweep does to it);
    # the hit job stands on its own copied artifacts.
    shutil.rmtree(root / src)
    with TestClient(main.app) as client:
        d = client.get(f"/v1/jobs/{hit}/download?format=json")
        assert d.status_code == 200
        md = client.get(f"/v1/jobs/{hit}/download?format=md")
        assert md.status_code == 200


def test_opt_in_vs_off_produce_identical_transcripts(tmp_path, monkeypatch):
    """AC C8 second arm: the flag selects the DELIVERY PATH, never the
    result (C4f) — same audio with and without X-Cache ends with identical
    transcript artifacts."""
    root = _setup(tmp_path, monkeypatch)
    tok_on, _ = _seed_done_job(root, "s1", x_cache="1")
    with TestClient(main.app) as client:
        r = _post(client, "s1", audio=WAV)  # no header: today's path
        tok_off = r.json()["token"]
        assert r.json()["cache_hit"] is False
    main.process_job_background(tok_off, str(root / tok_off / "audio_source.wav"), "clip.wav")
    assert _state(root, tok_off)["status"] == "done"
    for fname in ("output_result.json", "markdown_report.md"):
        a = (root / tok_on / fname).read_bytes()
        b = (root / tok_off / fname).read_bytes()
        if fname.endswith(".json"):
            # AC C8: content is identical EXCEPT the per-job token field,
            # which legitimately differs between two runs.
            j, k = json.loads(a), json.loads(b)
            assert j.pop("token") == tok_on and k.pop("token") == tok_off
            assert j == k, fname
        else:
            # markdown: byte-identical except the renderer's `| Processed |`
            # timestamp line (AC C8 excepts timestamp fields).
            norm = lambda bs: [l for l in bs.decode().splitlines()
                               if not l.startswith("| Processed |")]
            assert norm(a) == norm(b), fname


# --- self-healing and legacy jobs --------------------------------------------

def test_stale_row_is_deleted_and_job_reruns(tmp_path, monkeypatch):
    """Storage rule: a row whose source is gone / whose artifacts vanished is
    deleted at lookup time and the request falls through to a FRESH run
    (self-healing), not a 500 and not a silently-empty 202."""
    root = _setup(tmp_path, monkeypatch)
    src, _ = _seed_done_job(root, "s1")
    shutil.rmtree(root / src)              # evict underneath the row
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=WAV)
        assert r.status_code == 202
        body = r.json()
        assert body["cache_hit"] is False
        assert body["status"] == "queued"  # fresh run took over
    assert all(x[3] != src for x in _rows(root))


def test_legacy_null_hash_never_keys_a_cache(tmp_path, monkeypatch):
    """Plan: legacy jobs with null audio_sha256 -> MISS, fresh run, no crash;
    and such a job stores no row even if it somehow completes opted-in."""
    root = _setup(tmp_path, monkeypatch)
    legacy = root / "legacyjob"
    legacy.mkdir()
    (legacy / "status.json").write_text(json.dumps({
        "token": "legacyjob", "status": "done", "seq": 1,
        "session_ident": main._hash_ident("session:s1"),
        "created_at": 1.0, "queued_at": 1.0, "progress": 1.0,
        "filename": "old.wav",
    }))
    (legacy / "output_result.json").write_text("{}")
    (legacy / "markdown_report.md").write_text("# old\n")
    with TestClient(main.app) as client:
        r = _post(client, "s1", x_cache="1", audio=AUDIO)  # undecodable -> null hash
        assert r.status_code == 202
        body = r.json()
        assert body["cache_hit"] is False and body["audio_sha256"] is None
        tok = body["token"]
    main.process_job_background(tok, str(root / tok / "audio_source.wav"), "clip.wav")
    assert _rows(root) == []               # null-hash opt-in job: no row


# --- sweep wiring ---------------------------------------------------------------

def test_age_sweep_purges_rows_with_the_source(tmp_path, monkeypatch):
    """Storage rule: cache-row removal is wired into the EXISTING sweeps —
    a row whose source_token is deleted by the age sweep dies in the SAME
    pass. The cache must never be a reason to delete (or to keep) a job."""
    root = _setup(tmp_path, monkeypatch)
    src, _ = _seed_done_job(root, "s1")
    assert len(_rows(root)) == 1
    removed = main.sweep_expired_jobs(root, retention_days=0)
    assert removed == []                                    # 0 disables (AC2)
    past = (root / src / "status.json").stat().st_mtime - 200000
    os.utime(root / src / "status.json", (past, past))      # age it past 1 day
    removed = main.sweep_expired_jobs(root, retention_days=1)
    assert src in removed
    assert all(x[3] != src for x in _rows(root))


def test_disk_pressure_eviction_purges_rows(tmp_path, monkeypatch):
    """Same wiring for enforce_disk_pressure: evicted sources lose their rows."""
    root = _setup(tmp_path, monkeypatch)
    src, _ = _seed_done_job(root, "s1")
    old = (root / src / "status.json").stat().st_mtime - 400000
    os.utime(root / src / "status.json", (old, old))
    removed = main.enforce_disk_pressure(root, 1, free_bytes_fn=lambda *_: 0)
    assert src in removed
    assert all(x[3] != src for x in _rows(root))
