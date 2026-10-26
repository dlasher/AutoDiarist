"""Diarization threading (card t_3cb8f97c).

The sherpa-onnx offline diarization CLI defaults to ONE thread for both the
segmentation and the embedding network; measured on the NVIDIA T400 host
(4 cores, 137.5 s clip) the diarization phase runs 58.3 s at 1 thread and
20.5 s at 4 (2.85x). It REGRESSES above the core count (8 threads = 68 s,
worse than 1), so a fixed default of 4/8 is harmful on smaller/larger boxes
and catastrophic under concurrency (N workers x T threads each).

The rule under test (implemented in scripts/sherpa-diarization-wrapper.sh,
the ONE place shared by the CUDA and Vulkan images):

  SHERPA_NUM_THREADS (default unset/0/negative/non-integer = AUTO):
      auto = floor(cores / MAX_PROCESSING_GLOBAL), clamped to >= 1 and
      <= min(SHERPA_MAX_THREADS, cores) with SHERPA_MAX_THREADS defaulting
      to 8. With the shipped default N=1 that is the core count (the
      measured optimum) capped at 8; at N=3 on the 4-core box it degrades
      to 1 — never worse than today's single-thread behaviour.
      MAX_PROCESSING_GLOBAL that is unset, non-integer or <1 counts as 1
      (same parse rule as api/main.py).
      An explicit valid value is passed VERBATIM (only clamped down to the
      core count, and never below 1); cores unknown -> cap 8.
  Both flags (``--segmentation.num-threads`` and ``--embedding.num-threads``)
  are injected on EVERY sherpa invocation: the ceiling-disabled exec path,
  pass 1, AND the ceiling re-run pass (the re-run overlaps pass 1's memory,
  so the cap keeps it conservative).

Threading is a PERFORMANCE knob, not a quality knob: the same audio must
produce byte-identical wrapper stdout at 1, 2 and 4 threads (hermetic half
of AC5; the live half is re-verified with the real CLI on the host).

All tests drive the real wrapper script against a stub SHERPA_REAL, so they
run hermetically in CI — no GPU, no models, no engine. The core count is
simulated with a stub ``nproc`` on PATH (the wrapper probes nproc first,
then getconf _NPROCESSORS_ONLN, then gives up -> unknown).
"""

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts" / "sherpa-diarization-wrapper.sh"

# Exactly the argument shape the crispasr engine forwards today.
ENGINE_ARGS = [
    "--clustering.num-clusters=0",
    "--segmentation.pyannote-model=/opt/models/seg.onnx",
    "--embedding.model=/opt/models/embed.onnx",
    "/tmp/input.wav",
]

# Deterministic 3-label diarization stdout (same label set the ceiling stub
# emits, so the default ceiling of 8 never re-runs in threading tests).
_STUB_OUT = "\n".join(
    [
        "0.016 -- 1.549 speaker_01",
        "1.549 -- 3.010 speaker_02",
        "3.010 -- 4.900 speaker_01",
        "5.200 -- 7.800 speaker_03",
    ]
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture()
def stub(tmp_path: Path):
    """Fake sherpa CLI + argv log; also provides mktemp for the wrapper."""
    log = tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    real = _write(
        tmp_path / "sherpa-real",
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$STUB_LOG"\n'
        f"printf '%s\\n' '{_STUB_OUT}'\n",
    )
    _write(tmp_path / "mktemp", "#!/bin/sh\ntmp=$(mktemp /tmp/w.XXXXXX)\n"
                                "printf '%s\\n' \"$tmp\"\n")
    return real, log


def _calls(log: Path):
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(line.split())
    return out


def _path_dir(tmp_path: Path, name: str, script: str) -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    _write(d / "nproc", script)
    return d


def _run(tmp_path: Path, real: Path, log: Path, args, *, env_extra=None,
         cores=None, path_only=False):
    """Run the real wrapper. cores -> stub nproc printing that value.

    path_only=True restricts PATH to the stub bin dir (simulates a box
    where neither nproc nor getconf can report cores).
    """
    env = {**os.environ, "SHERPA_REAL": str(real), "STUB_LOG": str(log)}
    env.pop("SHERPA_NUM_THREADS", None)
    env.pop("MAX_PROCESSING_GLOBAL", None)
    env.pop("DIARIZATION_MAX_SPEAKERS", None)
    env.pop("SHERPA_CLUSTER_THRESHOLD", None)
    env.update(env_extra or {})
    bindirs = []
    if cores is not None:
        bindirs.append(_path_dir(
            tmp_path, "bin", f"#!/bin/sh\necho {cores}\n"))
    if path_only:
        # no nproc stub (fail), plus a getconf stub that also fails, so the
        # wrapper must conclude "cores unknown".
        d = tmp_path / "binfail"
        d.mkdir(exist_ok=True)
        _write(d / "nproc", "#!/bin/sh\nexit 1\n")
        _write(d / "getconf", "#!/bin/sh\nexit 1\n")
        bindirs = [d]
    if bindirs:
        sep = os.pathsep
        env["PATH"] = sep.join(str(d) for d in bindirs) + (
            "" if path_only else sep + env.get("PATH", ""))
    # the script is 644 in git (the Dockerfile chmods it in the image);
    # drive it through /bin/sh, exactly like the ceiling tests.
    proc = subprocess.run(
        ["/bin/sh", str(WRAPPER), *args], capture_output=True, text=True,
        env=env, timeout=60,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def _thread_flags(argv):
    return {a for a in argv
            if a.startswith(("--segmentation.num-threads=",
                             "--embedding.num-threads="))}


# --------------------------------------------------------------------------
# AC1 — the wrapper passes BOTH thread flags; exact argument list.
# --------------------------------------------------------------------------

def test_ac1_both_thread_flags_present(stub, tmp_path):
    """Unset knob on a 4-core box: both flags present on the single call."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4)
    (argv,) = _calls(log)
    seg = [a for a in argv if a == "--segmentation.num-threads=4"]
    emb = [a for a in argv if a == "--embedding.num-threads=4"]
    assert seg and emb, argv


def test_ac1_exact_argument_list(stub, tmp_path):
    """The injected argv is exactly threshold + both thread flags + engine args."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
         env_extra={"SHERPA_NUM_THREADS": "2"})
    (argv,) = _calls(log)
    assert argv == [
        "--clustering.cluster-threshold=0.40",
        "--segmentation.num-threads=2",
        "--embedding.num-threads=2",
        *ENGINE_ARGS,
    ], argv


def test_ac1_flags_on_ceiling_disabled_path(stub, tmp_path):
    """DIARIZATION_MAX_SPEAKERS=0 (exec path) still carries the flags."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
         env_extra={"DIARIZATION_MAX_SPEAKERS": "0"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=4", "--embedding.num-threads=4"}


# --------------------------------------------------------------------------
# AC2 — the safe default: auto = floor(cores / workers), floor 1, cap 8.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cores, workers, expected",
    [
        (4, None, 4),    # shipped default N=1 -> the measured optimum
        (4, 1, 4),
        (4, 2, 2),
        (4, 3, 1),       # 4//3=1: never oversubscribe
        (4, 4, 1),
        (4, 5, 1),       # floor clamps at 1 — regression floor
        (2, 3, 1),
        (1, 3, 1),
        (12, 1, 8),      # cap: "a sane range (e.g. 1..8)"; 8 on AMD's 12
        (12, 2, 6),
        (12, 4, 3),      # the planner's arithmetic table
        (16, 1, 8),
        (8, 8, 1),
    ],
)
def test_ac2_auto_default_is_cores_over_workers(stub, tmp_path, cores,
                                                workers, expected):
    real, log = stub
    env = {} if workers is None else {"MAX_PROCESSING_GLOBAL": str(workers)}
    _run(tmp_path, real, log, ENGINE_ARGS, cores=cores, env_extra=env)
    (argv,) = _calls(log)
    flags = _thread_flags(argv)
    assert flags == {f"--segmentation.num-threads={expected}",
                     f"--embedding.num-threads={expected}"}, argv
    # regression floor: never 0, never above the core count, never above 8
    assert 1 <= expected <= min(cores, 8)


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "", "2.5", "1x"])
def test_ac2_invalid_workers_count_as_one(stub, tmp_path, bad):
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
         env_extra={"MAX_PROCESSING_GLOBAL": bad})
    (argv,) = _calls(log)
    assert "--segmentation.num-threads=4" in argv, argv


def test_ac2_auto_resolution_is_logged_to_stderr(stub, tmp_path):
    """The planner note: an automatically computed default is deterministic
    and LOGGED (ops must be able to see what the box resolved to)."""
    real, log = stub
    proc = _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
                env_extra={"MAX_PROCESSING_GLOBAL": "3"})
    err = proc.stderr
    assert re.search(r"cores=4", err), err
    assert re.search(r"workers=3", err), err
    assert re.search(r"threads=1\b", err), err


def test_ac2_unknown_cores_resolves_to_one_not_zero(stub, tmp_path):
    """A box that reports no core count must fall to 1 (== today's single
    thread), NEVER 0 (which would hang or fail the CLI)."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, path_only=True,
         env_extra={"DIARIZATION_MAX_SPEAKERS": "0"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=1", "--embedding.num-threads=1"}, argv


# --------------------------------------------------------------------------
# AC3 — an explicit knob is passed verbatim to BOTH networks.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ac3_explicit_value_passed_verbatim(stub, tmp_path, n):
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
         env_extra={"SHERPA_NUM_THREADS": str(n)})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        f"--segmentation.num-threads={n}", f"--embedding.num-threads={n}"}, argv


def test_ac3_explicit_overrides_auto_and_workers(stub, tmp_path):
    """A valid explicit value ignores MAX_PROCESSING_GLOBAL entirely."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=12,
         env_extra={"SHERPA_NUM_THREADS": "2", "MAX_PROCESSING_GLOBAL": "4"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=2", "--embedding.num-threads=2"}, argv


# --------------------------------------------------------------------------
# AC4 — clamping: unsafe values resolve to a safe value, never 0, and an
# explicit value never exceeds the core count.
# --------------------------------------------------------------------------

def test_ac4_zero_negative_noninteger_auto(stub, tmp_path):
    """'0', '-1', 'abc' and '' all fall back to the auto default (here
    cores=4, workers=1 -> 4) — never 0."""
    real, log = stub
    for bad in ["0", "-1", "abc", ""]:
        _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
             env_extra={"SHERPA_NUM_THREADS": bad})
    for argv in _calls(log):
        assert _thread_flags(argv) == {
            "--segmentation.num-threads=4", "--embedding.num-threads=4"}, argv


def test_ac4_huge_value_clamped_to_cores(stub, tmp_path):
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
         env_extra={"SHERPA_NUM_THREADS": "9999"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=4", "--embedding.num-threads=4"}, argv


def test_ac4_explicit_above_cap_allowed_up_to_cores(stub, tmp_path):
    """The 1..8 cap bounds the AUTO default, not an operator's explicit
    choice: an explicit 12 on a 12-core box is a valid (if measured-slower)
    override."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=12,
         env_extra={"SHERPA_NUM_THREADS": "12"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=12", "--embedding.num-threads=12"}, argv


# --------------------------------------------------------------------------
# Planner note 14:45 — the cap. Option (c): SHERPA_MAX_THREADS (default 8)
# so a big box can raise the auto ceiling WITHOUT a rebuild (pending the
# AMD 8-vs-10-vs-12 sweep), but the resolved default can NEVER exceed the
# core count: 8 on a 2-core host must become 2, not 8.
# --------------------------------------------------------------------------

def test_cap_default_8_leaves_2_core_host_at_2_not_8(stub, tmp_path):
    """A fixed 8 would violate the never-oversubscribe rule: the cap
    applies as min(cap, cores)."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=2)
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=2", "--embedding.num-threads=2"}, argv


def test_cap_knob_raises_the_default_on_a_big_box(stub, tmp_path):
    """AMD at N=1: auto stops at the cap (8) by default; the operator lifts
    it to the core count via SHERPA_MAX_THREADS — runtime knob, no
    rebuild (the option the planner note accepts for the pending sweep)."""
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=12,
         env_extra={"SHERPA_MAX_THREADS": "12"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=12", "--embedding.num-threads=12"}, argv


def test_cap_knob_is_itself_clamped_to_cores(stub, tmp_path):
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=12,
         env_extra={"SHERPA_MAX_THREADS": "16"})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=12", "--embedding.num-threads=12"}, argv


@pytest.mark.parametrize("bad", ["0", "-2", "abc", ""])
def test_cap_knob_invalid_falls_back_to_8(stub, tmp_path, bad):
    real, log = stub
    _run(tmp_path, real, log, ENGINE_ARGS, cores=12,
         env_extra={"SHERPA_MAX_THREADS": bad})
    (argv,) = _calls(log)
    assert _thread_flags(argv) == {
        "--segmentation.num-threads=8", "--embedding.num-threads=8"}, argv


# --------------------------------------------------------------------------
# AC5 (hermetic half) — threading must not change results.
# --------------------------------------------------------------------------

def test_ac5_stdout_identical_at_1_2_4_threads(stub, tmp_path):
    """Same audio, same engine args: byte-identical wrapper stdout at 1, 2
    and 4 threads (the real-CLI half of this claim is verified live on the
    host; the wrapper's own job is to be thread-count-transparent: no
    re-runs, no label changes, no stdout mutation)."""
    real, log = stub
    outputs = []
    for n in (1, 2, 4):
        proc = _run(tmp_path, real, log, ENGINE_ARGS, cores=4,
                    env_extra={"SHERPA_NUM_THREADS": str(n)})
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    # ... and the segment/speaker counts are unchanged (3 labels, 4 segments)
    lines = [ln for ln in outputs[0].splitlines()
             if re.match(r"^[\d.]+ -- [\d.]+ speaker_\d+$", ln)]
    assert len(lines) == 4
    assert len({ln.split()[-1] for ln in lines}) == 3


# --------------------------------------------------------------------------
# Planner note — the ceiling re-run is a SECOND sherpa invocation; the
# thread count must apply to BOTH passes.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# AC10 (planner follow-up, card t_3cb8f97c) — the test SUITE must grow on
# this card: a count regression means tests were silently dropped during
# the refactor. Baseline was 405 collected (main @ 75e231e); every later
# card must only ADD to that number.
# --------------------------------------------------------------------------

def test_ac10_suite_collected_count_does_not_shrink():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=str(REPO), capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode in (0, 5), proc.stdout[-2000:] + proc.stderr[-2000:]
    m = re.search(r"(\d+) tests? collected", proc.stdout)
    assert m, proc.stdout[-2000:]
    collected = int(m.group(1))
    assert collected > 405, (
        f"suite collected {collected} tests; baseline on main was 405 and "
        "this card may only add tests (AC10)"
    )


def test_thread_flags_survive_the_ceiling_rerun(stub, tmp_path):
    """Forcing an over-explode pass-1 label count through the ceiling makes
    the wrapper re-invoke sherpa with num-clusters=N; the re-run must carry
    the same thread flags as pass 1."""
    real, log = tmp_path / "sherpa-real", tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    # stub: first call reports AUTOLABELS speaker labels (> ceiling), a
    # second call with an explicit --clustering.num-clusters=N reports N.
    _write(real, textwrap.dedent(
        """\
        #!/bin/sh
        printf '%s\\n' "$*" >> "$STUB_LOG"
        nc=0
        for a in "$@"; do
          case "$a" in --clustering.num-clusters=*) nc="${a#*=}" ;; esac
        done
        n=$AUTOLABELS
        if [ "$nc" -gt 0 ] 2>/dev/null; then n=$nc; fi
        i=1
        while [ "$i" -le "$n" ]; do
          printf '0.%02d -- 9.%02d speaker_%02d\\n' "$i" "$i" "$i"
          i=$((i + 1))
        done
        """))
    _write(tmp_path / "mktemp", "#!/bin/sh\ntmp=$(mktemp /tmp/w.XXXXXX)\n"
                                "printf '%s\\n' \"$tmp\"\n")
    env = {**os.environ, "SHERPA_REAL": str(real), "STUB_LOG": str(log),
           "AUTOLABELS": "10", "DIARIZATION_MAX_SPEAKERS": "8",
           "SHERPA_NUM_THREADS": "2", "MAX_PROCESSING_GLOBAL": "1"}
    proc = subprocess.run(["/bin/sh", str(WRAPPER), *ENGINE_ARGS],
                          capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    calls = _calls(log)
    assert len(calls) == 2, calls
    want = {"--segmentation.num-threads=2", "--embedding.num-threads=2"}
    assert _thread_flags(calls[0]) == want, calls[0]
    assert _thread_flags(calls[1]) == want, calls[1]
    assert "--clustering.num-clusters=8" in calls[1], calls[1]
    # capped re-run output: exactly the ceiling's label count
    labels = {ln.split()[-1] for ln in proc.stdout.splitlines()
              if re.match(r"^[\d.]+ -- [\d.]+ speaker_\d+$", ln)}
    assert len(labels) == 8
