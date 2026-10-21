"""V2 retention sweep + disk-pressure eviction (card t_5e6200fc, AC1-AC7).

Deterministic strategy: the sweep/eviction primitives are pure functions over
an injected jobs root and an injected free-space probe, so no test fakes the
filesystem itself. Clock injection is a `now` parameter on the age sweep and
a `sleep_func` parameter on the interval loops; the boot path is exercised
through the real lifespan.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main

ROOT = Path(__file__).resolve().parents[2]
DAY_SECONDS = 86400
MIB = 1024 * 1024


def _setup(tmp_path, monkeypatch, retention="30", disk_mib="0"):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("RETENTION_DAYS", retention)
    monkeypatch.setenv("DISK_MIN_FREE_MIB", disk_mib)
    monkeypatch.delenv("DISK_CHECK_INTERVAL_SECONDS", raising=False)
    return root


def _make_job(root, token, status="done", age_days=0.0, seq=1):
    job_dir = root / token
    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "token": token,
                "status": status,
                "error": None,
                "seq": seq,
                "session": "s1",
                "filename": "a.wav",
            }
        )
    )
    if age_days:
        stamp = time.time() - age_days * DAY_SECONDS
        os.utime(status_file, (stamp, stamp))
        os.utime(job_dir, (stamp, stamp))
    return status_file


def _fake_free(script_mib):
    """free-space stub returning scripted MiB values, one per call."""
    values = iter(script_mib)

    def free(_path):
        return next(values) * MIB

    return free


# --- AC1: age-based expiry ---------------------------------------------------


def test_age_sweep_removes_expired_dir_and_logs_token(tmp_path, monkeypatch, caplog):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "tok_old", status="done", age_days=40)
    _make_job(root, "tok_young", status="done", age_days=5)

    with caplog.at_level(logging.INFO, logger="api.main"):
        removed = main.sweep_expired_jobs(root, retention_days=30)

    assert removed == ["tok_old"]
    assert not (root / "tok_old").exists()
    assert (root / "tok_young").exists()
    # the removal is logged with the token and a reason
    assert "tok_old" in caplog.text
    assert "age-expiry" in caplog.text


def test_age_sweep_ages_out_failed_and_cancelled(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_done", status="done", age_days=40)
    _make_job(root, "t_failed", status="failed", age_days=40)
    _make_job(root, "t_cancel", status="cancelled", age_days=40)

    removed = main.sweep_expired_jobs(root, retention_days=30)

    assert sorted(removed) == ["t_cancel", "t_done", "t_failed"]


def test_age_sweep_leaves_live_pair_and_statusless_dirs(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_queued", status="queued", age_days=40)
    _make_job(root, "t_processing", status="processing", age_days=40)
    junk = root / "no-status-file"
    junk.mkdir()
    stamp = time.time() - 40 * DAY_SECONDS
    os.utime(junk, (stamp, stamp))

    removed = main.sweep_expired_jobs(root, retention_days=30)

    assert removed == []
    assert (root / "t_queued").exists()
    assert (root / "t_processing").exists()
    assert junk.exists()


# --- AC2: RETENTION_DAYS=0 disables age expiry --------------------------------


def test_retention_days_zero_disables_age_expiry(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, retention="0")
    _make_job(root, "tok_old", status="done", age_days=40)

    assert main._retention_days() == 0
    removed = main.sweep_expired_jobs(root, retention_days=main._retention_days())
    assert removed == []
    assert (root / "tok_old").exists()


def test_retention_days_default_is_30(tmp_path, monkeypatch):
    monkeypatch.delenv("RETENTION_DAYS", raising=False)
    assert main._retention_days() == 30
    assert main.DEFAULT_RETENTION_DAYS == 30
    assert main.RETENTION_SWEEP_INTERVAL_SECONDS == 3600  # hourly


# --- AC3: disk-pressure eviction, oldest first --------------------------------


def test_disk_eviction_oldest_first_until_watermark(tmp_path, monkeypatch, caplog):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_old", status="done", age_days=40)
    _make_job(root, "t_mid", status="failed", age_days=20)
    _make_job(root, "t_new", status="done", age_days=5)

    free = _fake_free([0, 4000, 9000])
    with caplog.at_level(logging.INFO, logger="api.main"):
        evicted = main.enforce_disk_pressure(
            root, min_free_mib=8192, free_bytes_fn=free
        )

    assert evicted == ["t_old", "t_mid"]  # oldest mtime first, stop at watermark
    assert not (root / "t_old").exists()
    assert not (root / "t_mid").exists()
    assert (root / "t_new").exists()
    assert "disk-pressure" in caplog.text
    assert "t_old" in caplog.text and "t_mid" in caplog.text


def test_disk_eviction_never_evicts_processing(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_a", status="done", age_days=10)
    _make_job(root, "t_b", status="processing", age_days=30)
    _make_job(root, "t_c", status="done", age_days=20)

    free = _fake_free([0, 0, 9000])
    evicted = main.enforce_disk_pressure(
        root, min_free_mib=8192, free_bytes_fn=free
    )

    assert "t_b" not in evicted
    assert evicted == ["t_c", "t_a"]  # t_b is oldest but skipped; stop at watermark
    assert (root / "t_b").exists()


def test_disk_eviction_terminates_when_only_live_jobs_remain(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_queued", status="queued", age_days=20)
    _make_job(root, "t_processing", status="processing", age_days=40)

    evicted = main.enforce_disk_pressure(
        root, min_free_mib=8192, free_bytes_fn=lambda _p: 0
    )

    assert evicted == ["t_queued"]
    assert (root / "t_processing").exists()


def test_disk_eviction_disabled_when_watermark_zero(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    _make_job(root, "t_old", status="done", age_days=40)

    def explode(_path):  # must not even measure free space when disabled
        raise AssertionError("free-space probe called with DISK_MIN_FREE_MIB=0")

    assert main.enforce_disk_pressure(root, min_free_mib=0, free_bytes_fn=explode) == []
    assert (root / "t_old").exists()


# --- AC5: boot-time + hourly wiring -------------------------------------------


def test_boot_sweeps_at_lifespan_startup(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, retention="30", disk_mib="0")
    _make_job(root, "tok_old", status="done", age_days=40)
    _make_job(root, "tok_running", status="processing", age_days=40)
    _make_job(root, "tok_young", status="done", age_days=1)

    with TestClient(main.app):
        # lifespan startup ran the boot pass synchronously before yielding
        assert not (root / "tok_old").exists()
        assert (root / "tok_running").exists()
        assert (root / "tok_young").exists()


def test_hourly_age_loop_wiring_injected_clock(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch, retention="30", disk_mib="0")
    _make_job(root, "tok_old", status="done", age_days=40)

    requested_intervals = []
    sweep_args = []
    real_sweep = main.sweep_expired_jobs

    def counting_sweep(jobs_dir, retention_days, now=None):
        sweep_args.append(retention_days)
        return real_sweep(jobs_dir, retention_days, now=now)

    monkeypatch.setattr(main, "sweep_expired_jobs", counting_sweep)

    async def fake_sleep(seconds):
        requested_intervals.append(seconds)
        if len(requested_intervals) >= 3:
            raise RuntimeError("stop-loop")

    with pytest.raises(RuntimeError, match="stop-loop"):
        asyncio.run(main._retention_loop(sleep_func=fake_sleep))

    assert requested_intervals == [3600, 3600, 3600]  # hourly
    assert sweep_args == [30, 30]  # RETENTION_DAYS read per pass, env-driven
    assert not (root / "tok_old").exists()


def test_disk_loop_interval_env_and_wiring(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("DISK_MIN_FREE_MIB", raising=False)  # exercise the 8192 default
    monkeypatch.setenv("DISK_CHECK_INTERVAL_SECONDS", "42")

    requested_intervals = []
    enforce_args = []
    enforced_dirs = []
    real_enforce = main.enforce_disk_pressure

    def counting_enforce(jobs_dir, min_free_mib, free_bytes_fn=None):
        enforce_args.append(min_free_mib)
        enforced_dirs.append(jobs_dir)
        return real_enforce(jobs_dir, min_free_mib, free_bytes_fn=free_bytes_fn)

    monkeypatch.setattr(main, "enforce_disk_pressure", counting_enforce)

    async def fake_sleep(seconds):
        requested_intervals.append(seconds)
        if len(requested_intervals) >= 3:
            raise RuntimeError("stop-loop")

    with pytest.raises(RuntimeError, match="stop-loop"):
        asyncio.run(main._disk_pressure_loop(sleep_func=fake_sleep))

    assert requested_intervals == [42, 42, 42]
    assert enforce_args == [8192, 8192]  # DISK_MIN_FREE_MIB default honored
    assert enforced_dirs == [root, root]  # sweeps the live module JOBS_DIR


# --- AC6: no dead-knob caveat survives in code or docs -------------------------


def test_shipped_code_reads_retention_knobs():
    src = Path(main.__file__).read_text(encoding="utf-8")
    assert '"RETENTION_DAYS"' in src
    assert '"DISK_MIN_FREE_MIB"' in src
    assert '"DISK_CHECK_INTERVAL_SECONDS"' in src


def test_docs_no_longer_caveat_retention_as_dead_knob():
    config_md = (ROOT / "CONFIG.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "no shipped code path reads it" not in config_md
    assert "reserved for a future retention sweep" not in readme
    assert "`RETENTION_DAYS` | `30`" in config_md
    assert "DISK_MIN_FREE_MIB" in config_md
    assert "DISK_CHECK_INTERVAL_SECONDS" in config_md
    assert "RETENTION_DAYS" in readme and "DISK_MIN_FREE_MIB" in readme
