"""Stage-based progress reporting (card t_14ac6b49).

status.json must expose a machine-readable ``stage`` field, publish
stage-appropriate messages, keep progress monotonic non-decreasing, reach
exactly 1.0 at done, and on failure report the stage where it failed instead
of a stale success-ish progress.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ["MOCK_PIPELINE"] = "1"

# Real engine segment shape; the fakes below stay in-process so it never
# reaches a real renderer.
_SEGMENTS = [
    {"text": "hello", "t0": 0.0, "t1": 1.0, "speaker": "Speaker 1"},
    {"text": "world", "t0": 1.0, "t1": 2.0, "speaker": "Speaker 2"},
]


@pytest.fixture()
def jobs_root(tmp_path, monkeypatch):
    from api import main

    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    return root, main


@pytest.fixture()
def client(jobs_root, monkeypatch):
    _, main = jobs_root
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "1")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    with TestClient(main.app) as c:
        yield c


def _observe(main, root, box, label):
    """Snapshot the live status.json mid-run — what a polling client sees."""
    for status_file in sorted(root.glob("*/status.json")):
        st = json.loads(status_file.read_text())
        box.append(
            {
                "at": label,
                "status": st.get("status"),
                "stage": st.get("stage"),
                "progress": st.get("progress"),
                "message": st.get("message"),
            }
        )


def _instrument(main, root, box, monkeypatch, asr_passes=2, boom=None):
    """Replace the pipeline entry points with fakes that snapshot status.json
    at every real stage boundary."""

    def fake_decode(input_path, output_wav):
        _observe(main, root, box, "decode")
        from pathlib import Path

        Path(output_wav).write_bytes(b"RIFF fake wav")
        return output_wav

    def fake_transcribe(audio_path, mock=False, duration=None, progress=None):
        if boom == "transcribe":
            raise RuntimeError("engine exploded")
        if progress:
            for done in range(1, asr_passes + 1):
                progress("asr", done, asr_passes + 1)
                _observe(main, root, box, f"asr {done}/{asr_passes + 1}")
            progress("diar", asr_passes + 1, asr_passes + 1)
            _observe(main, root, box, "diar")
        return [dict(s) for s in _SEGMENTS]

    def fake_spellcheck(segments):
        _observe(main, root, box, "spellcheck")
        from types import SimpleNamespace

        return SimpleNamespace(segments=segments)

    def fake_self_id(segments):
        _observe(main, root, box, "self_id")
        return segments

    def fake_render(segments, meta):
        _observe(main, root, box, "render")
        return "# report\n"

    monkeypatch.setattr(main, "decode_audio", fake_decode)
    monkeypatch.setattr(main, "transcribe_and_diarize", fake_transcribe)
    monkeypatch.setattr(main, "spell_check_segments", fake_spellcheck)
    monkeypatch.setattr(main, "suggest_speaker_names", fake_self_id)
    monkeypatch.setattr(main, "render_markdown", fake_render)


def _wait_final(client, token, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{token}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail(f"job {token} did not finish within {timeout}s")


def _run_pipeline(client, jobs_root, monkeypatch, **kw):
    root, main = jobs_root
    box = []
    # Exercise the REAL branch structure: non-mock so decode + spellcheck
    # stages are published (the pipeline itself is fully faked below).
    monkeypatch.setenv("MOCK_PIPELINE", "0")
    monkeypatch.setenv("SPELL_CHECK", "1")
    _instrument(main, root, box, monkeypatch, **kw)
    response = client.post(
        "/v1/jobs",
        files={"file": ("test.wav", b"RIFF....WAVEfmt ....data....", "audio/wav")},
    )
    assert response.status_code == 202
    token = response.json()["token"]
    body = _wait_final(client, token)
    return token, body, box


# --- AC1: >= 4 distinct stage messages, in canonical order, ending at done ---

def test_real_run_publishes_stage_sequence_in_order(client, jobs_root, monkeypatch):
    _, main = jobs_root
    token, body, box = _run_pipeline(client, jobs_root, monkeypatch)

    stages_seen = [st["stage"] for st in box if st["stage"]]
    assert None not in stages_seen, f"stage field missing mid-run: {box}"

    distinct = []
    for stage in stages_seen:
        if stage not in distinct:
            distinct.append(stage)
    # Real boundaries crossed inside the worker run, observed live.
    assert {"decoding", "transcribing", "diarizing", "rendering"} <= set(distinct)
    assert len(distinct) >= 4

    order = main.STAGE_ORDER
    idx = [order.index(s) for s in distinct]
    assert idx == sorted(idx), f"stages out of order: {distinct}"

    assert body["status"] == "done"
    assert body["stage"] == "done"


def test_stage_messages_are_stage_appropriate(client, jobs_root, monkeypatch):
    _, _, box = _run_pipeline(client, jobs_root, monkeypatch)
    by_stage = {}
    for st in box:
        by_stage.setdefault(st["stage"], st["message"])
    assert by_stage["decoding"] == "Decoding audio..."
    assert by_stage["transcribing"].startswith("Transcribing")
    assert by_stage["diarizing"].startswith("Diarizing")
    assert by_stage["rendering"] == "Rendering markdown report..."
    assert len(set(by_stage.values())) >= 4


# --- AC2: progress monotonic non-decreasing, final == 1.0 ---

def test_progress_monotonic_and_reaches_one(client, jobs_root, monkeypatch):
    _, _, box = _run_pipeline(client, jobs_root, monkeypatch)
    progresses = [st["progress"] for st in box if st["progress"] is not None]
    assert progresses == sorted(progresses), f"progress regressed: {progresses}"
    assert all(0.0 <= p <= 1.0 for p in progresses)


def test_done_progress_exactly_one(client, jobs_root, monkeypatch):
    _, body, _ = _run_pipeline(client, jobs_root, monkeypatch)
    assert body["progress"] == 1.0


# --- AC3: failure mid-pipeline reports failed + failing stage ---

def test_failed_job_reports_failing_stage(client, jobs_root, monkeypatch):
    _, main = jobs_root
    token, body, box = _run_pipeline(client, jobs_root, monkeypatch, boom="transcribe")

    assert body["status"] == "failed"
    assert body["stage"] == "transcribing"
    assert body["progress"] < 1.0
    assert body["progress"] == json.loads(
        (main.JOBS_DIR / token / "status.json").read_text()
    )["progress"]
    assert "Failed during transcribing" in body["message"]


def test_queued_job_exposes_queued_stage(jobs_root):
    root, main = jobs_root
    monkey_off = os.environ.get("QUEUE_WORKER_ENABLED")
    os.environ["QUEUE_WORKER_ENABLED"] = "0"
    try:
        with TestClient(main.app) as c:
            response = c.post(
                "/v1/jobs",
                files={"file": ("t.wav", b"RIFF....WAVEfmt ....data....", "audio/wav")},
            )
            assert response.status_code == 202
            token = response.json()["token"]
            body = c.get(f"/v1/jobs/{token}").json()
            assert body["status"] == "queued"
            assert body["stage"] == "queued"
            assert body["progress"] == 0.0
            assert body["message"] == "Queued"
    finally:
        if monkey_off is None:
            os.environ.pop("QUEUE_WORKER_ENABLED", None)
        else:
            os.environ["QUEUE_WORKER_ENABLED"] = monkey_off


# --- AC4: stable 'stage' field with documented values ---

def test_stage_field_values_are_from_documented_enum(client, jobs_root, monkeypatch):
    _, main = jobs_root
    token, body, box = _run_pipeline(client, jobs_root, monkeypatch)
    documented = set(main.STAGE_ORDER)
    for st in box:
        assert st["stage"] in documented, f"undocumented stage {st['stage']!r}"
    assert body["stage"] in documented


# --- pass fraction: multi-pass/chunked engines refine within the band ---

def test_per_pass_progress_advances_within_transcribing(client, jobs_root, monkeypatch):
    _, _, box = _run_pipeline(client, jobs_root, monkeypatch, asr_passes=4)
    asr_obs = [st for st in box if st["at"].startswith("asr")]
    assert len(asr_obs) == 4
    passes = [st["progress"] for st in asr_obs]
    assert passes == sorted(passes) and len(set(passes)) == 4, (
        f"per-pass progress did not advance: {passes}"
    )
    assert all(st["stage"] == "transcribing" for st in asr_obs)
    assert "pass 3/5" in asr_obs[2]["message"]


# --- AC5/6: README documents the stage field ---

def _read_journal(main, token):
    p = main.JOBS_DIR / token / "status_journal.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_journal_records_full_stage_sequence(client, jobs_root, monkeypatch):
    """AC1 on the authoritative on-disk journal: every publish, in order."""
    _, main = jobs_root
    token, body, _ = _run_pipeline(client, jobs_root, monkeypatch)
    journal = _read_journal(main, token)
    stages = [e["stage"] for e in journal if e.get("stage")]
    distinct = list(dict.fromkeys(stages))
    assert "queued" in distinct
    assert len(distinct) >= 4
    assert distinct == [s for s in main.STAGE_ORDER if s in distinct], (
        f"stage transitions out of order: {distinct}"
    )
    progresses = [e["progress"] for e in journal if e.get("progress") is not None]
    assert progresses == sorted(progresses), f"journal progress regressed: {progresses}"
    assert journal[-1]["status"] == "done" and journal[-1]["progress"] == 1.0


def test_journal_failure_ends_at_failing_stage(client, jobs_root, monkeypatch):
    _, main = jobs_root
    token, body, _ = _run_pipeline(client, jobs_root, monkeypatch, boom="transcribe")
    journal = _read_journal(main, token)
    assert journal[-1]["status"] == "failed"
    assert journal[-1]["stage"] == "transcribing"
    assert journal[-1]["progress"] < 1.0


def test_readme_documents_stage_field():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text()
    assert "stage" in text
    from api import main

    for stage in main.STAGE_ORDER:
        assert stage in text, f"README does not document stage {stage!r}"


# --- reclaim -> claim regression (review fix): claim must not rewind progress
#
# The lifespan reclaim (api/main.py, boot revalidation) deliberately keeps the
# dead attempt's ``progress`` so FIFO/poll clients never see a rewind at boot.
# The claim publish is the next write a client observes, so it must clamp to
# the preserved value instead of resetting to 0.05 (AC2: monotonic across the
# WHOLE job, restart included).


def _seed_reclaimed_job(root, main, token, progress):
    """status.json exactly as the lifespan reclaim path leaves it."""
    jdir = root / token
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "status.json").write_text(
        json.dumps(
            {
                "token": token,
                "status": "queued",
                "stage": "queued",
                "progress": progress,
                "seq": 1,
                "message": "Requeued after service restart",
                "filename": "a.wav",
                "audio_path": str(jdir / "audio_source.wav"),
                "queued_at": 1.0,
            }
        )
    )
    return jdir


def test_claim_after_reclaim_does_not_regress_progress(jobs_root):
    root, main = jobs_root
    jdir = _seed_reclaimed_job(root, main, "tok_reclaim", 0.45)

    job = main._claim_next_queued()
    assert job is not None and job["status"] == "processing"
    published = json.loads((jdir / "status.json").read_text())
    # No observed value below the preserved 0.45 within the new attempt.
    assert published["progress"] >= 0.45, (
        f"claim regressed progress {0.45} -> {published['progress']}"
    )
    assert published["message"] == "Claimed by worker"
    # Every later band publish goes through update_state()'s max(prev, prog)
    # clamp, so from here on monotonicity already holds; the claim write was
    # the only rewind.
    assert main.STAGE_ORDER.index(published["stage"]) <= main.STAGE_ORDER.index("decoding")


def test_claim_of_fresh_queued_job_still_advances(jobs_root):
    # Clamp must not freeze fresh jobs either: 0.0 -> 0.05 at claim.
    root, main = jobs_root
    jdir = _seed_reclaimed_job(root, main, "tok_fresh", 0.0)

    assert main._claim_next_queued() is not None
    published = json.loads((jdir / "status.json").read_text())
    assert published["progress"] == 0.05
    assert published["message"] == "Claimed by worker"
