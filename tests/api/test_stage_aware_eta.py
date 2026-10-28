"""Stage-aware running-phase ETA (card t_79f9cdc6).

One rate constant cannot serve both stages. Measured on the NV (T400)
host: the 8411 s file transcribed at 0.1016 s of wall per second of
audio and diarized at 0.3661 (3.60x); the 249 s panel at 0.1245 vs
0.4781 (3.84x) — the ratio is stable across a 34x spread of audio
length. The merged build priced the diarization heartbeat tick plan at
the ASR constant, so at the stage's true end the band fraction was only
done/total = 286/779 — the heartbeat's own tick model over-predicted the
tick count by 2.7x. With the fraction creeping, the linear extrapolation
in _running_estimate — remaining = elapsed x (1 - frac) / frac — GREW
while the stage was genuinely progressing, the ratchet floor was that
growing value, and the estimate re-anchored upward at the end of every
stage (panel: 29.8 -> 52.0 -> 73.4 s across a 119 s stage).

The contract these tests pin (planner instrument + card ACs):
  1. the estimate must not GROW while a stage is progressing — simulated
     through the API's OWN publication path (engine callback ->
     update_state -> poll -> ratchet), and relabelled 'stage_rate_model';
  2. the tick total lands near the stage's REAL tick count (286 on the
     8411 s file, not 779), so done/total reaches 1.0 when the stage
     does;
  3. the shipped defaults are the measured per-stage rates and are in
     the config catalogue (AC4);
  4. the ratchet contract is untouched: never inflated, null when there
     is no honest estimate, expires when a frozen stage passes the
     re-anchor allowance (AC3 — the existing ratchet tests remain the
     real gate; asserted directly here as a safety net).

Deterministic: pure arithmetic on the API's own functions plus one
simulation of the production publication path. No GPU, no model, no
fixture, no wall-clock sleeping.
"""
import math
import time

import pytest

from api import main
from pipeline import config_vars, transcribe as tr

# MEASURED on the NV host (card t_79f9cdc6 body, run on merged 393ec8d):
ASR_RATE = 0.1016            # s wall per s audio, transcribing stage
DIAR_RATE = 0.3661           # s wall per s audio, diarizing stage

DUR = 8411.072               # the measured SEWP file
REAL_TICKS = 286             # ticks the real 3077 s stage published
ASR_TOTAL = 779              # ceil(8411.072 x 0.185 / 2.0) — the bug
DEPLOYED_CHUNK = 2.0         # default DIAR_CHUNK_ESTIMATE_SECONDS
DEPLOYED_PACE = 2.0 / 0.185  # ~10.81 s per tick as deployed
STAGE_WALL = DUR * DIAR_RATE  # 3077 s of measured stage cost

# the 249 s panel fixture (the live poll in the card body):
PANEL_DUR = 249.0
PANEL_ASR_RATE = 0.1245
PANEL_DIAR_RATE = 0.4781
PANEL_STAGE_WALL = PANEL_DUR * PANEL_DIAR_RATE     # ~119 s


def test_diar_tick_total_is_not_priced_at_the_asr_rate():
    """Planner test 2 — the tick total comes from the stage's own rate.

    The deployed model priced the stage at the ASR rate: 779 predicted,
    286 published (2.7x over). Priced at the diarization rate the total
    must land near the real count, at the DEPLOYED tick pace (the tick
    interval is out of scope). Red today: diar_tick_plan's only rate is
    the ASR constant and no knob re-prices the stage.
    """
    total, _pace = tr.diar_tick_plan(
        duration=DUR, rate=tr.processing_rate_per_audio_second(),
        chunk_seconds=DEPLOYED_CHUNK)
    assert total == ASR_TOTAL                     # pins the legacy call
    # the stage-aware spelling: the stage's own rate must drive a total
    # that lands at the real published tick count for the measured wall.
    stage_total, pace = tr.diar_tick_plan(
        duration=DUR, rate=tr.diarization_rate_per_audio_second(),
        chunk_seconds=DEPLOYED_CHUNK)
    assert abs(pace - DEPLOYED_PACE) < 0.5, pace  # pace is not this fix
    predicted_ticks = STAGE_WALL / pace
    assert abs(stage_total - predicted_ticks) <= 40, (
        stage_total, predicted_ticks)
    assert abs(stage_total - REAL_TICKS) <= 40, stage_total


def test_shipped_defaults_are_the_measured_rates(monkeypatch):
    """AC4 — separate per-stage knobs, shipped defaults = measured."""
    monkeypatch.delenv("DIAR_SECONDS_PER_AUDIO_SECOND", raising=False)
    monkeypatch.delenv("PROCESSING_SECONDS_PER_AUDIO_SECOND", raising=False)
    rate = tr.diarization_rate_per_audio_second()
    assert abs(rate - DIAR_RATE) < 0.01, rate
    assert rate > tr.processing_rate_per_audio_second() * 2, \
        "the diarization stage must be priced materially above the ASR one"
    names = {v.name for v in config_vars.CATALOG}
    assert "DIAR_SECONDS_PER_AUDIO_SECOND" in names, sorted(names)[:80]
    doc = " ".join(v.effect or "" for v in config_vars.CATALOG
                   if v.name == "DIAR_SECONDS_PER_AUDIO_SECOND")
    assert "measured" in doc.lower() and "0.3661" in doc, doc


def _clean_eta_env(monkeypatch, tmp_path):
    for var in ("PROCESSING_SECONDS_PER_AUDIO_SECOND",
                "DIAR_SECONDS_PER_AUDIO_SECOND",
                "DIAR_CHUNK_ESTIMATE_SECONDS",
                "DIAR_HEARTBEAT_INTERVAL_SECONDS",
                "RUNNING_ETA_MIN_PROGRESS",
                "RUNNING_ETA_REANCHOR_SECONDS",
                "AUDIO_DURATION_FALLBACK_SECONDS",
                "MOCK_PIPELINE"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setattr(main, "_running_eta_floor", {}, raising=False)


def _heartbeat_step(now, started, stage_t, total, pace, frac_seen):
    """Mirror _DiarHeartbeat.publish_now(): done = elapsed/pace, capped
    at total-1, monotonic; returns None when nothing moves."""
    done = min(int(stage_t / pace), total - 1)
    if done <= frac_seen["done"]:
        return None
    frac_seen["done"] = done
    return done


def test_eta_does_not_grow_while_a_stage_is_progressing(tmp_path, monkeypatch):
    """AC2 — the planner's instrument, through the API's OWN path.

    The panel's measured run shape (the card's live poll): an attempt
    starts, transcribes at 0.1245 s/s-audio, the diarization stage then
    runs 119 s at 0.4781 with heartbeat ticks priced at the ASR-shaped
    total (the deployed model: 5 ticks, so at the stage's true end the
    count is capped and the fraction sits at 4/5 = 0.8, not 1.0). Each
    simulation step publishes through the production engine-callback
    chain (main._engine_progress -> update_state -> _write_state) and
    polls through the production ratchet (_running_eta_ratchet), exactly
    like GET /v1/jobs does. The published remaining seconds must NOT
    increase across the polls; no re-anchor may fire while the stage is
    genuinely progressing; and the number must be labelled
    'stage_rate_model' so a client can tell which model produced it.

    Red today: the fraction creeps against a 2.7x-too-large total,
    remaining = elapsed*(1-frac)/frac grows, the ratchet floors the
    growth for a while and then LEGITIMATELY re-anchors upward when the
    held value misses its own ETA — the panel's 29.8 -> 52.0 -> 73.4.
    """
    _clean_eta_env(monkeypatch, tmp_path)
    token = "sim-panel-growth"
    now0 = time.time()
    started = now0                      # attempt clock == simulation clock

    state_holder = {"stage": "transcribing", "progress": 0.0,
                    "est": None, "method": None}

    def update_state(st, prog=None, msg="", err="", stage=None, dur=None,
                     code=None, estimated=None, estimate_method=None):
        st_data = {
            "token": token, "status": st, "session": "s", "seq": 1,
            "created_at": started, "started_at": started,
            "attempt_started_at": started,
            "audio_path": "/x.wav", "filename": "x.wav",
            "duration_seconds": PANEL_DUR,
            "audio_duration_seconds": PANEL_DUR,
            "stage": stage or state_holder["stage"],
            "progress": float(prog or 0.0),
            "last_progress_at": state_holder["poll_now"],
        }
        if estimated:
            st_data["progress_estimated"] = True
            if estimate_method:
                st_data["progress_estimate_method"] = estimate_method
        else:
            st_data.pop("progress_estimated", None)
            st_data.pop("progress_estimate_method", None)
        main._write_state(st_data)
        state_holder["stage"] = st_data["stage"]
        state_holder["progress"] = st_data["progress"]
        state_holder["est"] = bool(estimated)
        state_holder["method"] = st_data.get("progress_estimate_method")

    on_progress = main._engine_progress(update_state)

    # The production tick plan for the stage as the ENGINE builds it
    # (whatever rates the shipped code picks — the simulation must not
    # re-price the model itself or it stops testing the product):
    interval, chunk = tr.diar_heartbeat_settings()
    total, pace = tr.diar_tick_plan(
        duration=PANEL_DUR,
        rate=tr.diarization_rate_per_audio_second(),
        chunk_seconds=chunk)
    assert total > 0
    state_holder["poll_now"] = now0

    values: list[float] = []
    seen: dict[str, object] = {"done": -1}
    labels: list[str] = []
    t = 0.0
    while t < PANEL_STAGE_WALL:
        t += interval
        now = now0 + t
        state_holder["poll_now"] = now
        done = _heartbeat_step(now, started, t, total, pace, seen)
        if done is None:
            continue
        on_progress("diar", done, total, estimated=True,
                    estimate_method="elapsed_time_model")
        assert state_holder["est"] is True and state_holder["method"]
        state = main._read_state(token)
        est = main._running_eta_ratchet(state, now)
        if est is not None:
            values.append(est[0])
            labels.append(state_holder["method"])

    assert len(values) >= 3, f"no estimate during a 119 s stage: {values!r}"
    grows = [(a, b) for a, b in zip(values, values[1:]) if b > a + 1e-9]
    assert not grows, (
        f"estimate GREW while the stage progressed in {len(grows)}/"
        f"{len(values) - 1} steps; series head {values[:8]}")
    assert set(labels) == {"stage_rate_model"}, sorted(set(labels))
    # honesty at the stage's end: the panel run published 73.4 s left
    # after a 119 s stage; the tail must trend toward the stage's true
    # (zero) remaining, not away from it.
    assert values[-1] <= values[0], (values[0], values[-1])


def test_estimate_within_tolerance_of_observed_stage_rate(tmp_path,
                                                          monkeypatch):
    """AC1 — seeded from the diarization stage's own cost, not the ASR's.

    At the stage's start (first poll after the band jump), the published
    remaining must sit inside a stated tolerance of the MEASURED stage
    wall (119 s on the panel). The ASR-shaped number (29.8 s in the
    card's live poll — 0.25x of the truth) must fail this; a
    diarization-shaped seed must land within 1.5x/0.66x of 119 s.
    """
    _clean_eta_env(monkeypatch, tmp_path)
    now = time.time()
    # attempt lived through the ASR phase at the measured 0.1245 rate:
    asr_wall = PANEL_DUR * PANEL_ASR_RATE
    state = {"token": "tol", "status": "processing",
             "attempt_started_at": now - asr_wall,
             "started_at": now - asr_wall,
             "duration_seconds": PANEL_DUR,
             "audio_duration_seconds": PANEL_DUR,
             "stage": "diarizing",
             "progress": main.STAGE_PROGRESS["diarizing"],
             "last_progress_at": now}
    main._running_eta_floor.clear()
    est = main._running_estimate(state, now)
    assert est is not None, "stage start must be estimable"
    lo, hi = PANEL_STAGE_WALL * 0.66, PANEL_STAGE_WALL * 1.5
    assert lo <= est[0] <= hi, (
        f"estimate {est[0]:.1f}s off the measured stage wall "
        f"{PANEL_STAGE_WALL:.1f}s (tolerance {lo:.1f}-{hi:.1f}s)")


def test_existing_ratchet_contract_still_holds(tmp_path, monkeypatch):
    """AC3 (planner test 3) — pinned directly: never inflated relative
    to its floor, null with no honest estimate, expires when a frozen
    stage passes the re-anchor allowance. The EXISTING ratchet tests
    remain unmodified and are the real gate."""
    _clean_eta_env(monkeypatch, tmp_path)
    now = time.time()
    reanchor = main._running_eta_reanchor_seconds()

    def st(prog, **over):
        base = {"token": "rk", "status": "processing",
                "attempt_started_at": now - 4000.0,
                "duration_seconds": DUR, "audio_duration_seconds": DUR,
                "progress": prog, "stage": "transcribing",
                "last_progress_at": now - 5.0}
        base.update(over)
        return base

    # 1. never inflated: progress advances, the published value may only
    #    tighten, never rise.
    a = main._running_eta_ratchet(st(0.40), now)
    b = main._running_eta_ratchet(st(0.45), now + 60.0)
    assert a is not None and b is not None
    assert b[0] <= a[0] + 1e-9, (a, b)
    assert b[1] <= a[1] + 1e-9, (a, b)

    # 2. null when there is no honest estimate.
    main._running_eta_floor.clear()
    assert main._running_eta_ratchet(
        {"token": "n1", "status": "processing", "progress": 0.5},
        now) is None                      # no start stamp
    assert main._running_eta_ratchet(
        {"token": "n2", "status": "processing", "progress": 0.0,
         "attempt_started_at": now - 100.0}, now) is None

    # 3. a frozen stage expires to null past the re-anchor allowance.
    main._running_eta_floor.clear()
    assert main._running_eta_ratchet(st(0.40), now) is not None
    assert main._running_eta_ratchet(
        st(0.40, last_progress_at=now + reanchor + 60.0),
        now + reanchor + 60.0) is None
