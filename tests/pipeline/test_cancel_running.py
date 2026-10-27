"""Cancel a RUNNING engine mid-flight (card t_fef54150).

The DELETE endpoint used to 409 on ``processing`` because graceful
mid-chunk cancel "was out of scope" while the service ran a single
concurrent worker. Both premises are gone: the engine is chunked, runs
in its own session (``start_new_session=True``) and ``_kill_process_group``
already terminates the whole group on the timeout path. These tests pin
the new client-facing cancel machinery at the pipeline layer:

* AC1: a per-job ``CancelControl`` reaches the running engine; cancelling
  drives ``JobCancelled`` (NOT a timeout, NOT a generic CrispasrError)
  and the engine's whole process group is terminated then reaped.
* AC2: exit-by-signal after a cancel kill must never be classified as a
  generic engine failure — the exception type is the cancel terminal.
* AC3: a cancel requested before the engine spawns must short-circuit
  the stage boundary (no spawn at all) — a retry may not resurrect a
  cancelled job by re-spawning the engine.
* AC4: the mock stage honours the same cancellation contract so the
  queue/API layer is testable GPU-free.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pipeline import transcribe as tr

CONFIG_ENVS = (
    "CRISPASR_BIN", "GGUF_MODEL", "DIAR_METHOD", "CRISPASR_LANGUAGE",
    "CRISPASR_GPU_BACKEND", "CRISPASR_CACHE_DIR", "CRISPASR_LIB_DIR",
    "CRISPASR_TIMEOUT_SECONDS", "MIN_FRAGMENT_SECONDS", "VRAM_LIMIT_MIB",
    "MOCK_PIPELINE", "CHUNK_SECONDS", "OVERLAP_SECONDS",
    "SHERPA_SEGMENT_MODEL", "SHERPA_EMBEDDING_MODEL", "SHERPA_BIN",
    "MOCK_TRANSCRIBE_SECONDS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in CONFIG_ENVS:
        monkeypatch.delenv(name, raising=False)
    fake = tmp_path / "crispasr"
    fake.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo "crispasr v0.8.43-test"; exit 0; fi\n'
        "exec /bin/sleep 300\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    gguf = tmp_path / "model.gguf"
    gguf.write_text("fake-gguf")
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    monkeypatch.setenv("CRISPASR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CRISPASR_TIMEOUT_SECONDS", "300")
    tr._BACKEND_PROBE_CACHE.clear()
    yield


def _run_in_thread(fn, *args, box: list, **kwargs):
    """Run fn in a thread, capturing any exception into ``box`` (3.15
    Thread.exception is not guaranteed on the CI interpreter)."""
    def target():
        try:
            fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - captured for the test
            box.append(exc)
    t = threading.Thread(target=target)
    t.start()
    return t


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(os.getpgid(pid), 0)
        return True
    except OSError:
        return False


# --- AC1: CancelControl kill semantics ---------------------------------------


def test_cancel_control_request_terminates_live_process_group():
    """request() on an attached engine kills the WHOLE group, not just the
    direct child — the same guarantee the timeout path already gives."""
    proc = subprocess.Popen(["/bin/sleep", "300"], start_new_session=True)
    ctrl = tr.CancelControl("tok-a")
    ctrl.attach(proc)
    assert ctrl.cancelled() is False

    killed = ctrl.request()
    assert killed is True
    deadline = time.monotonic() + 10.0
    while _group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _group_alive(proc.pid), "engine process group survived the cancel"
    proc.wait(timeout=5)
    assert proc.returncode != 0  # died of the signal


def test_cancel_requested_before_attach_kills_on_attach():
    """Close the spawn/race window: if the cancel lands while the engine is
    being spawned, attach() must kill the fresh process group immediately."""
    ctrl = tr.CancelControl("tok-b")
    assert ctrl.request() is False  # nothing attached yet
    proc = subprocess.Popen(["/bin/sleep", "300"], start_new_session=True)
    ctrl.attach(proc)
    deadline = time.monotonic() + 10.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None, "engine spawned after cancel survived"


# --- AC1/AC2: _run_crispasr cancel path --------------------------------------


def _streaming_fake(tmp_path: Path, body: str) -> Path:
    fake = tmp_path / "crispasr"
    fake.write_text("#!/bin/bash\n" + body + "\n")
    fake.chmod(0o755)
    return fake


def test_cancel_midrun_raises_job_cancelled_and_reaps_group(tmp_path, monkeypatch):
    """A cancel DURING the engine run: JobCancelled (not timeout, not
    CrispasrError), engine group dead, no misclassification of exit -15."""
    spawn_log = tmp_path / "spawns.log"
    fake = _streaming_fake(tmp_path, (
        f'echo spawn >> "{spawn_log}"\n'
        'echo "crispasr: progress =  16% (1/6 slices)" 1>&2\n'
        "exec /bin/sleep 300\n"
    ))
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp_path / "audio_16k.wav"
    wav.write_bytes(b"RIFFfake")

    # Spy the real Popen so the test can assert on the spawned engine after
    # the control detaches its reference in the cleanup path.
    created: list = []
    real_popen = subprocess.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        created.append(p)
        return p

    monkeypatch.setattr(tr.subprocess, "Popen", spy)

    reports = []
    ctrl = tr.CancelControl("tok-c")
    box: list = []

    def progress(stage, done, total):
        reports.append((stage, done, total))
        if stage == "asr" and not ctrl.cancelled():
            # Cancel the moment the first REAL engine boundary is observed.
            ctrl.request()

    t = _run_in_thread(tr._run_crispasr, str(wav), str(tmp_path), progress, 1,
                       box=box, cancel=ctrl)
    t.join(timeout=30)
    assert not t.is_alive(), "engine run did not honour the cancel within 30s"
    assert box, "engine run returned instead of raising"
    exc = box[0]
    assert isinstance(exc, tr.JobCancelled), f"expected JobCancelled, got {exc!r}"
    # AC2: it must not be the timeout or the generic failure classes.
    assert not isinstance(exc, tr.CrispasrError)

    proc = created[0]
    deadline = time.monotonic() + 10.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None, "engine survived the cancel kill"
    assert proc.returncode != 0  # killed by the signal, not a clean exit
    # The kill must have gone to the WHOLE group: the engine exec'd sleep in
    # its own session; nothing of it may remain reachable.
    assert not _group_alive(proc.pid)


def test_cancelled_job_is_not_respawned(tmp_path, monkeypatch):
    """Retry can never resurrect a cancelled run: exactly one engine
    invocation, then the JobCancelled terminates the attempt for good."""
    spawn_log = tmp_path / "spawns.log"
    fake = _streaming_fake(tmp_path, (
        f'echo spawn >> "{spawn_log}"\n'
        'echo "crispasr: progress =  16% (1/6 slices)" 1>&2\n'
        "exec /bin/sleep 300\n"
    ))
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    wav = tmp_path / "audio_16k.wav"
    wav.write_bytes(b"RIFFfake")

    ctrl = tr.CancelControl("tok-d")
    box: list = []

    def progress(stage, done, total):
        if stage == "asr" and not ctrl.cancelled():
            ctrl.request()

    t = _run_in_thread(tr._run_crispasr, str(wav), str(tmp_path), progress, 1,
                       box=box, cancel=ctrl)
    t.join(timeout=30)
    assert box and isinstance(box[0], tr.JobCancelled), \
        f"expected JobCancelled, got {box!r}"
    spawns = spawn_log.read_text().splitlines()
    assert len(spawns) == 1, f"engine respawned after cancel: {len(spawns)} spawns"


# --- AC3: cancel before spawn never reaches the engine ------------------------


def test_cancel_before_stage_boundary_never_spawns(tmp_path, monkeypatch):
    """Stage-boundary polling: with the flag already set,
    transcribe_and_diarize raises before Popen is ever called."""
    def tripwire(*a, **k):
        raise AssertionError("engine spawned despite pre-attached cancel")

    monkeypatch.setattr(tr.subprocess, "Popen", tripwire)
    wav = tmp_path / "audio_16k.wav"
    wav.write_bytes(b"RIFFfake")
    ctrl = tr.CancelControl("tok-e")
    ctrl.request()
    with pytest.raises(tr.JobCancelled):
        tr.transcribe_and_diarize(str(wav), mock=False, duration=600.0, cancel=ctrl)


# --- AC4: mock stage honours the same contract (GPU-free queue tests) --------


def test_mock_transcribe_parks_and_honours_cancel(monkeypatch):
    monkeypatch.setenv("MOCK_TRANSCRIBE_SECONDS", "3")
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    ctrl = tr.CancelControl("tok-f")
    threading.Timer(0.3, ctrl.request).start()
    start = time.monotonic()
    with pytest.raises(tr.JobCancelled):
        tr.transcribe_and_diarize("ignored.wav", mock=True, cancel=ctrl)
    assert time.monotonic() - start < 2.0, "mock stage ignored the cancel until timeout"
