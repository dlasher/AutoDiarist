"""RUNNING-PHASE ETA acceptance tests (card t_d64bdf29, AC1-AC6).

A job in `processing` gets an honest completion estimate on its status
payload, derived ONLY from its own observed progress: elapsed wall time
since the attempt start against the published monotonic `progress`
fraction (the stage bands the pipeline already advances on real work
boundaries). The fixed linear rate constant NEVER touches the emitted
value (AC1) — it gates emission only — and queued-side estimates are
byte-identical to card t_b3ae9366 (AC3 regression guard).

Honesty rules under test (README "Wait estimates"):
  * emission starts only once the job has crossed
    RUNNING_ETA_MIN_PROGRESS (default 0.30 — the transcribing band opens,
    i.e. decode + startup cost is fully behind it) and, when the duration
    is known, a startup-cost floor (AC4: a fresh job is null, not a
    fabricated value);
  * the published value ratchets down (min with the last valid value),
    never inflates, and goes null once progress has frozen beyond
    RUNNING_ETA_REANCHOR_SECONDS or a held value has passed its own
    completion time (freshness rule: fresh or null);
  * missing started_at/attempt_started_at or no usable progress -> null (AC4);
  * PROCESSING_SECONDS_PER_AUDIO_SECOND=0 disables the whole feature
    exactly like the queue side (AC4).

Deterministic strategy: pure-function tests pin the math; HTTP tests seed
on-disk state (worker OFF, no pipeline ever runs); one end-to-end test
drives the real mock pipeline through the production claim path with a
pause injected at a stage seam and polls the live API while it processes
(AC6). No GPU required anywhere.
"""
import calendar
import threading
import time
import uuid as _uuid
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from api import main

AUDIO = b"RIFF....WAVEfmt ....data...."
RATE = 0.185


def _setup(tmp_path, monkeypatch, rate=None, min_progress=None,
           reanchor=None, worker="0"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    # Fresh ratchet memory per test (the module dict is replaced, not
    # mutated, so monkeypatch restores the original afterwards).
    monkeypatch.setattr(main, "_running_eta_floor", {}, raising=False)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", worker)
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 10 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "50")
    monkeypatch.setenv("MAX_PROCESSING_PER_SESSION", "1")
    monkeypatch.setenv("MAX_PROCESSING_GLOBAL", "1")
    monkeypatch.delenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", raising=False)
    monkeypatch.delenv("RUNNING_ETA_MIN_PROGRESS", raising=False)
    monkeypatch.delenv("RUNNING_ETA_REANCHOR_SECONDS", raising=False)
    if rate is not None:
        monkeypatch.setenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", str(rate))
    if min_progress is not None:
        monkeypatch.setenv("RUNNING_ETA_MIN_PROGRESS", str(min_progress))
    if reanchor is not None:
        monkeypatch.setenv("RUNNING_ETA_REANCHOR_SECONDS", str(reanchor))
    return TestClient(main.app)


def _seed(**fields):
    """Persist a job state directly (worker OFF; nothing ever processes it)."""
    token = fields.pop("token", str(_uuid.uuid4()))
    state = {
        "token": token,
        "status": "processing",
        "session": "s",
        "seq": 1,
        "created_at": time.time(),
        "audio_path": "/x.wav",
        "filename": "x.wav",
    }
    state.update(fields)
    main._write_state(state)
    return token


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


# ---------------------------------------------------------------------------
# AC1 — a 60/120-minute RUNNING job gets a non-null completion estimate
# derived from its own observed progress, NOT from the fixed rate model.
# ---------------------------------------------------------------------------

def test_running_job_with_observed_progress_gets_completion_estimate(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    started = time.time() - 100.0
    token = _seed(stage="diarizing", progress=0.5, started_at=started,
                  last_progress_at=time.time() - 5.0, duration_seconds=None)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["status"] == "processing"
    assert body["estimated_remaining_seconds"] is not None
    assert body["estimated_completion_at"] is not None
    # 50% done in 100 s observed -> ~100 s remaining, and the completion
    # stamp must agree with the seconds within ISO second-truncation.
    assert abs(body["estimated_remaining_seconds"] - 100.0) <= 3.0
    assert abs(_epoch(body["estimated_completion_at"])
               - (time.time() + body["estimated_remaining_seconds"])) <= 1.5


def test_running_estimate_uses_observed_rate_not_fixed_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # Same observed progress + elapsed with two very different durations
    # (both past the startup-cost emission gate): a duration-derived value
    # would differ by minutes; the observed-rate estimate must be
    # identical — and must NOT equal the linear-model residual
    # (600 x 0.185 - 100 = 11 s).
    now = time.time()
    a = main._running_estimate({"status": "processing", "progress": 0.5,
                                "started_at": now - 100.0,
                                "duration_seconds": 600.0}, now)
    b = main._running_estimate({"status": "processing", "progress": 0.5,
                                "started_at": now - 100.0,
                                "duration_seconds": None}, now)
    assert a is not None and b is not None
    assert abs(a[0] - 100.0) < 0.01 and abs(b[0] - 100.0) < 0.01
    assert abs(a[0] - (600 * RATE - 100.0)) > 30.0


def test_running_estimate_unit_guard_prefers_attempt_clock(tmp_path, monkeypatch):
    """UNIT GUARD for clock precedence inside `_running_estimate` (AC4:
    denominator = attempt clock, so pre-crash elapsed never poisons the
    restarted rate). This calls the helper directly with DIFFERENT stamps
    to pin the lookup order only — in production every claim stamps
    `started_at` and `attempt_started_at` with the same value (they are
    never different), and the restart integrity chain is pinned by
    test_reclaimed_attempt_stays_null_until_new_attempt_earns_the_estimate.
    """
    _setup(tmp_path, monkeypatch)
    now = time.time()
    est = main._running_estimate({
        "status": "processing", "progress": 0.5,
        "started_at": now - 10000.0,        # the original (dead) attempt start
        "attempt_started_at": now - 100.0,  # the live attempt's start
    }, now)
    assert est is not None and abs(est[0] - 100.0) < 0.01


# ---------------------------------------------------------------------------
# AC2 — while progress advances, estimated_remaining_seconds decreases and
# NEVER moves up; the published value ratchets down and goes null instead
# of inflating or surviving stale (freshness rule).
# ---------------------------------------------------------------------------

def test_pure_math_strictly_decreases_across_advancing_fractions():
    now0 = 1_000_000.0
    state = {"status": "processing", "started_at": now0 - 1000.0}
    seen = []
    for frac in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99):
        state["progress"] = frac
        est = main._running_estimate(state, now0)
        assert est is not None
        seen.append(est[0])
    assert all(b < a for a, b in pairwise(seen)), seen


def test_published_value_never_inflates_when_the_tail_slows(tmp_path, monkeypatch):
    # Job at 50% after 100 s -> publishes 100 s. Next poll: the fraction
    # crawls to 0.51 at 300 s total elapsed — the raw observed-rate value
    # would INFLATE to ~288 s; the ratchet must hold at or below 100 s.
    client = _setup(tmp_path, monkeypatch)
    t0 = time.time()
    token = _seed(progress=0.5, started_at=t0 - 100.0,
                  last_progress_at=t0 - 2.0)
    first = client.get(f"/v1/jobs/{token}").json()["estimated_remaining_seconds"]
    assert abs(first - 100.0) <= 2.0
    st = main._read_state(token)
    assert st is not None
    st.update({"progress": 0.51, "started_at": t0 - 300.0,
               "last_progress_at": time.time()})
    main._write_state(st)
    second = client.get(f"/v1/jobs/{token}").json()["estimated_remaining_seconds"]
    assert second is not None and second <= first + 0.1
    assert second < 280.0  # specifically: NOT the inflated raw recomputation


def test_frozen_estimate_goes_null_then_returns_on_real_advance(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch, reanchor=2.0)
    t0 = time.time()
    # Progress last advanced 60 s ago, past the 2 s freshness allowance.
    token = _seed(progress=0.5, started_at=t0 - 70.0, last_progress_at=t0 - 60.0)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is None
    assert body["estimated_completion_at"] is None
    # A real stage advance re-derives from the live observation.
    st = main._read_state(token)
    assert st is not None
    st.update({"progress": 0.6, "last_progress_at": time.time()})
    main._write_state(st)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is not None


def test_held_value_past_its_own_completion_goes_null(tmp_path, monkeypatch):
    # Publish a short remaining, then freeze progress: once the promised
    # completion time passes, the payload must go null — never re-inflate.
    client = _setup(tmp_path, monkeypatch, reanchor=3600.0)
    t0 = time.time()
    token = _seed(progress=0.9, started_at=t0 - 9.0, last_progress_at=t0 - 1.0)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is not None
    assert body["estimated_remaining_seconds"] <= 2.0  # ~1 s at frac 0.9
    time.sleep(1.6)
    body2 = client.get(f"/v1/jobs/{token}").json()
    assert body2["estimated_remaining_seconds"] is None
    assert body2["estimated_completion_at"] is None


# ---------------------------------------------------------------------------
# AC3 — queued jobs' estimates are untouched (regression guard on
# card t_b3ae9366's arithmetic and surfaces).
# ---------------------------------------------------------------------------

def test_queued_estimates_unchanged_by_running_feature(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    _seed(status="processing", stage="transcribing",
          progress=0.35, started_at=time.time() - 10.0,
          last_progress_at=time.time() - 1.0, duration_seconds=1000.0)
    token_q = _seed(status="queued", stage="queued", progress=0.0,
                    duration_seconds=1000.0, created_at=time.time())
    body = client.get(f"/v1/jobs/{token_q}").json()
    # t_b3ae9366 arithmetic: own estimate 185 s + the processing job's
    # linear-model residual clamped to [0, est] (est 185, elapsed 10 -> 175).
    assert abs(body["estimated_wait_seconds"] - 360.0) <= 2.0
    assert body["estimated_completion_at"] is not None
    # The queued payload never grows running-phase fields.
    assert body.get("estimated_remaining_seconds") is None


# ---------------------------------------------------------------------------
# AC4 — no usable signal means null, never a fabricated value.
# ---------------------------------------------------------------------------

def test_fresh_processing_job_is_null(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    token = _seed(stage="decoding", progress=0.05, started_at=time.time(),
                  last_progress_at=time.time(), duration_seconds=3600.0)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is None
    assert body["estimated_completion_at"] is None


def test_missing_start_times_is_null(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    token = _seed(progress=0.5, last_progress_at=time.time())  # no start stamps
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is None
    assert body["estimated_completion_at"] is None


def test_below_min_progress_gate_is_null(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    token = _seed(progress=0.2, started_at=time.time() - 100.0,
                  last_progress_at=time.time() - 1.0)
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["estimated_remaining_seconds"] is None
    # The gate is operator-tunable, not hardcoded: widening it emits.
    monkeypatch.setenv("RUNNING_ETA_MIN_PROGRESS", "0.10")
    token2 = _seed(progress=0.2, started_at=time.time() - 100.0,
                   last_progress_at=time.time() - 1.0)
    body2 = client.get(f"/v1/jobs/{token2}").json()
    assert body2["estimated_remaining_seconds"] is not None


def test_rate_disabled_omits_running_eta(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch, rate="0")
    token = _seed(progress=0.5, started_at=time.time() - 100.0,
                  last_progress_at=time.time() - 1.0)
    body = client.get(f"/v1/jobs/{token}").json()
    # Disabled emission omits the fields entirely — same contract the
    # queue side has (t_b3ae9366), now applied to the running-phase trio.
    assert body.get("estimated_remaining_seconds") is None
    assert body.get("estimated_completion_at") is None
    assert body.get("estimated_wait_seconds") is None


# ---------------------------------------------------------------------------
# AC5 — consistent across surfaces: the poll and the session listing
# agree; the POST admission response is untouched.
# ---------------------------------------------------------------------------

def test_listing_processing_entry_carries_same_estimate(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    header = main._session_header_name()
    token = _seed(progress=0.5, started_at=time.time() - 100.0,
                  last_progress_at=time.time() - 1.0,
                  session_ident=main._hash_ident("session:sessA"))
    poll = client.get(f"/v1/jobs/{token}").json()
    listed = client.get("/v1/jobs", headers={header: "sessA"}).json()
    entry = next(j for j in listed["jobs"] if j["token"] == token)
    assert entry["status"] == "processing"
    assert entry["estimated_remaining_seconds"] is not None
    assert entry["estimated_completion_at"] == poll["estimated_completion_at"]
    assert abs(entry["estimated_remaining_seconds"]
               - poll["estimated_remaining_seconds"]) <= 0.15
    assert abs(entry["estimated_wait_seconds"]
               - entry["estimated_remaining_seconds"]) <= 0.15


def test_post_admission_shape_unchanged(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    # Stub the admission probe exactly like the t_b3ae9366 suite: no
    # ffprobe subprocess. The 202 body keeps its queue-side values and
    # grows NO running-phase field (the job is not running yet).
    monkeypatch.setattr(main, "_probe_duration", lambda path: 1234.0)
    resp = client.post("/v1/jobs", headers={main._session_header_name(): "sessB"},
                       files={"file": ("a.wav", AUDIO, "audio/wav")})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert "estimated_remaining_seconds" not in body
    assert body["estimated_completion_at"] is not None  # queue-side value, as t_b3ae9366


# ---------------------------------------------------------------------------
# AC6 — progress_percent: integer 0..100 published alongside the banded
# float `progress`, consistent with the stage bands, monotonic.
# ---------------------------------------------------------------------------

def test_progress_percent_exposed_on_poll_and_listing(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    token = _seed(stage="transcribing", progress=0.42, started_at=time.time(),
                  last_progress_at=time.time(),
                  session_ident=main._hash_ident("session:sessC"))
    body = client.get(f"/v1/jobs/{token}").json()
    assert body["progress"] == 0.42
    assert body["progress_percent"] == 42
    assert isinstance(body["progress_percent"], int)
    entry = next(j for j in client.get("/v1/jobs", headers={
        main._session_header_name(): "sessC"}).json()["jobs"] if j["token"] == token)
    assert entry["progress_percent"] == 42


def test_progress_percent_matches_band_floor_and_done(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    fresh = _seed(stage="decoding", progress=main.STAGE_PROGRESS["decoding"])
    done = _seed(status="done", stage="done", progress=1.0)
    assert client.get(f"/v1/jobs/{fresh}").json()["progress_percent"] == 5
    d = client.get(f"/v1/jobs/{done}").json()
    assert d["progress_percent"] == 100
    # AC4: a finished job reports NO remaining estimate.
    assert "estimated_remaining_seconds" not in d
    assert d.get("estimated_completion_at") is None


def test_progress_percent_monotonic_and_band_consistent_over_stage_sweep():
    last = -1
    for stage in main.STAGE_ORDER:
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            if stage == "done":
                frac = 0.0
            prog = main._stage_progress(stage, frac)
            pct = main._progress_percent({"progress": prog})
            # Consistent with the banded float: the int is the same value
            # rounded to whole percent, and never rewinds across the sweep.
            assert pct == round(prog * 100)
            assert 0 <= pct <= 100
            assert pct >= last, (stage, frac, pct, last)
            last = pct


def test_estimate_converges_on_a_steady_simulated_series(tmp_path, monkeypatch):
    # AC2 wording: ~50% in, remaining never exceeds elapsed; with a steady
    # observed rate the completion estimate converges to the real finish
    # time as progress advances (simulated series — no GPU, no sleeps).
    _setup(tmp_path, monkeypatch)
    total = 600.0
    for frac in (0.5, 0.7, 0.9, 0.95, 0.99):
        now = total * frac
        est = main._running_estimate({
            "status": "processing", "progress": frac,
            "started_at": 0.0}, now)
        assert est is not None
        elapsed = now
        assert est[0] <= elapsed + 1e-9          # ~50%: remaining <= elapsed
        assert abs(est[0] - (total - now)) < 1e-6  # exact convergence
        assert abs(est[1] - total) < 1e-6


# ---------------------------------------------------------------------------
# Scope 5 — a `cancelling`/cancelled path must not break or fake the ETA.
# ---------------------------------------------------------------------------

def test_cancelling_and_cancelled_jobs_carry_no_completion_estimate(tmp_path, monkeypatch):
    client = _setup(tmp_path, monkeypatch)
    t = time.time()
    cancelling = _seed(status="cancelling", stage="spellcheck", progress=0.75,
                       started_at=t - 100.0, last_progress_at=t - 1.0,
                       message="Cancellation requested")
    body = client.get(f"/v1/jobs/{cancelling}").json()
    assert body["status"] == "cancelling"
    assert "estimated_remaining_seconds" not in body
    assert body.get("estimated_completion_at") is None
    assert body["progress_percent"] == 75
    gone = _seed(status="cancelled", stage="queued", progress=0.0,
                 cancelled_at=t)
    b2 = client.get(f"/v1/jobs/{gone}").json()
    assert b2["progress_percent"] == 0
    assert "estimated_remaining_seconds" not in b2


# ---------------------------------------------------------------------------
# Plumbing guards.
# ---------------------------------------------------------------------------

def test_ratchet_memory_pruned_by_claim_and_terminal_update(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    t0 = time.time()
    token = _seed(progress=0.5, started_at=t0 - 100.0,
                  last_progress_at=t0 - 1.0)
    client = TestClient(main.app)
    assert client.get(f"/v1/jobs/{token}").json()["estimated_remaining_seconds"] is not None
    assert token in main._running_eta_floor
    # A reclaim drops the dead attempt's memory (restart-staleness trap).
    st = main._read_state(token)
    assert st is not None
    st.update({"status": "queued", "progress": 0.5})
    main._write_state(st)
    claimed = main._claim_next_queued()
    assert claimed is not None and claimed["token"] == token
    assert token not in main._running_eta_floor
    assert claimed["attempt_started_at"] >= t0 - 1.0  # fresh attempt stamp
    # Terminal publishes through update_state prune it at the source; the
    # end-to-end test pins that half against the real pipeline.


def test_reclaimed_attempt_stays_null_until_new_attempt_earns_the_estimate(tmp_path, monkeypatch):
    """Restart honesty chain at the production seams (review round 1).

    Encodes the PRODUCTION-VERIFIED sequence: boot requeue (lifespan
    touches status/stage only; `progress` + `last_progress_at` are
    preserved) -> `_claim_next_queued` re-stamps started_at AND
    attempt_started_at — they are always equal — and pops the ratchet
    memory -> polls on the new attempt.

    Verified outcomes (worker OFF, patched clock, no sleeps):
      * polls inside the startup-cost floor AND while progress is frozen
        at the preserved value -> NULL: the optimistic preserved-frac /
        new-clock number never reaches a client — "no false convergence
        after restart" holds via the EMISSION GATES, not via any
        started_at-preservation mechanism;
      * only a genuine advance above the preserved value (real
        new-attempt work) makes an estimate return, consistent with the
        new clock.
    """
    client = _setup(tmp_path, monkeypatch, rate=RATE)
    monkeypatch.setattr(main, "_running_eta_floor", {})  # fresh process

    t_crash = 10_000.0
    token = str(_uuid.uuid4())
    # Dead attempt: 3600 s audio, ran to the preserved band value 0.60.
    state = {
        "token": token, "status": "processing", "session": "s", "seq": 1,
        "created_at": t_crash - 310.0,
        "started_at": t_crash - 300.0,
        "attempt_started_at": t_crash - 300.0,
        "audio_path": "/x.wav", "filename": "x.wav",
        "duration_seconds": 3600.0, "duration_source": "ffprobe",
        "stage": "transcribing", "progress": 0.60,
        "last_progress_at": t_crash - 1.0,
    }

    def patch_clock(now):
        monkeypatch.setattr(time, "time", lambda: now)

    # 1. boot requeue exactly as lifespan does it (status/stage only).
    patch_clock(t_crash)
    state.update(status="queued", stage="queued")
    main._write_state(state)

    # 2. fresh claim: both clocks stamped equal, ratchet memory empty,
    #    progress preserved at 0.60 (the clamp, not a reset).
    claimed = main._claim_next_queued()
    assert claimed is not None and claimed["token"] == token
    st = main._read_state(token)
    assert st is not None
    assert st["attempt_started_at"] == st["started_at"], (
        "claim stamps one per-attempt clock; docs must not promise otherwise")
    assert st["progress"] == 0.60  # preserved (clamp), the trap this guards

    def poll(elapsed):
        now = st["started_at"] + elapsed
        patch_clock(now)
        return client.get(f"/v1/jobs/{token}").json()

    # 3. while the NEW attempt has not yet genuinely advanced past the
    #    preserved value (frozen progress; fast restart < reanchor
    #    window), no optimistic estimate is published. (Startup floor for
    #    this job is 3600*0.185*0.30 = 200 s; the freeze + gates are what
    #    make restart honest.)
    assert poll(11.0)["estimated_remaining_seconds"] is None
    assert poll(201.0)["estimated_remaining_seconds"] is None, (
        "preserved-numerator / new-denominator estimate must not be emitted")

    # 4. only after real new-attempt work advances the fraction does an
    #    estimate return — and it is consistent with the new clock.
    now = st["started_at"] + 520.0
    patch_clock(now)
    adv = main._read_state(token)
    assert adv is not None
    adv.update(progress=0.72, stage="transcribing", last_progress_at=now)
    main._write_state(adv)
    rem = client.get(f"/v1/jobs/{token}").json()["estimated_remaining_seconds"]
    assert rem is not None
    # 666 s = full re-run at the pre-crash observed rate (3600*0.185);
    # the ratchet-capped honest observed-rate region only (preserved-frac
    # arithmetic against stale clocks must never reach the client).
    assert rem <= 666.0 - 520.0 + 60.0, f"estimate regressed vs true finish: {rem}"


# ---------------------------------------------------------------------------
# End to end against the real mock pipeline (the pinned analogue of the
# card's live test-host run): claimed through the
# production path (attempt_started_at written by code, not seeded), with a
# pause injected at a stage seam so the API can be polled WHILE processing.
# The estimate appears only once progress is meaningful, never inflates
# across consecutive polls, and points near the actual finish.
# ---------------------------------------------------------------------------

def test_end_to_end_running_estimate_tracks_the_mock_run(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir(parents=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setattr(main, "_running_eta_floor", {}, raising=False)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", str(RATE))
    monkeypatch.delenv("RUNNING_ETA_MIN_PROGRESS", raising=False)
    monkeypatch.delenv("RUNNING_ETA_REANCHOR_SECONDS", raising=False)

    token = str(_uuid.uuid4())
    (root / token).mkdir()
    (root / token / "audio_source.wav").write_bytes(AUDIO)
    main._write_state({
        "token": token, "status": "queued", "session": "s", "seq": 1,
        "created_at": time.time(),
        "audio_path": str(root / token / "audio_source.wav"),
        "filename": "e2e.wav", "session_ident": "s",
        "duration_seconds": 60.0, "stage": "queued", "progress": 0.0,
    })

    # Pause the run inside suggest_speaker_names: by then the production
    # pipeline has published the transcribing/diarizing bands (progress >=
    # 0.30) and the job stays visibly `processing` for a poll window.
    real_suggest = main.suggest_speaker_names
    entered = threading.Event()

    def paused_suggest(*a, **k):
        entered.set()
        time.sleep(5.0)
        return real_suggest(*a, **k)

    monkeypatch.setattr(main, "suggest_speaker_names", paused_suggest)

    client = TestClient(main.app)
    # Production claim path: writes started_at/attempt_started_at itself.
    claimed = main._claim_next_queued()
    assert claimed is not None and claimed["token"] == token
    assert "attempt_started_at" in claimed, \
        "claim must record the attempt start (AC4 signal source)"

    runner = threading.Thread(
        target=main.process_job_background,
        args=(token, claimed["audio_path"], claimed["filename"]),
        daemon=True)
    runner.start()

    seen = []  # (progress, remaining, completion_epoch) while processing
    finished_at = None
    deadline = time.time() + 30.0
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{token}").json()
        if body["status"] == "done":
            finished_at = time.time()
            break
        if body["status"] == "processing":
            rem = body.get("estimated_remaining_seconds")
            if rem is not None:
                seen.append((float(body.get("progress") or 0.0),
                             rem, _epoch(body["estimated_completion_at"])))
        time.sleep(0.2)
    else:
        pytest.fail("mock pipeline did not finish")
    assert finished_at is not None
    runner.join(timeout=10)
    # Terminal update_state pruned the attempt's estimate memory.
    assert token not in main._running_eta_floor

    assert seen, "no running-phase estimate was ever emitted"
    # Only meaningful-progress emissions (gate respected).
    assert all(p >= 0.30 for p, _, _ in seen)
    # Never inflates across consecutive polls (ratchet contract).
    assert all(b <= a + 0.15 for (_, a, _), (_, b, _) in pairwise(seen)), seen
    # The last published completion stamp lands near the real finish —
    # never a wildly-late stale value while the job runs past it.
    last_completion = seen[-1][2]
    assert last_completion <= finished_at + 3.0
    assert last_completion >= finished_at - 12.0


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_gate_bounds_fall_back_to_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("RUNNING_ETA_MIN_PROGRESS", "9")
    assert main._running_eta_min_progress() == main.DEFAULT_RUNNING_ETA_MIN_PROGRESS
    monkeypatch.setenv("RUNNING_ETA_MIN_PROGRESS", "-1")
    assert main._running_eta_min_progress() == main.DEFAULT_RUNNING_ETA_MIN_PROGRESS
    monkeypatch.setenv("RUNNING_ETA_REANCHOR_SECONDS", "nonsense")
    assert main._running_eta_reanchor_seconds() == main.DEFAULT_RUNNING_ETA_REANCHOR_SECONDS
    monkeypatch.delenv("RUNNING_ETA_MIN_PROGRESS")
    assert main._running_eta_min_progress() == 0.30
