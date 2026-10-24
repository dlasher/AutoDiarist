"""Speaker-count ceiling (card t_38e52d89).

Root cause established by two sweeps on the VM (see the card):
* The NASA SEWP Q&A file (118 min, reference 6-7 speakers) explodes to 100
  distinct labels through the auto-threshold clustering path.
* The engine ALWAYS passes its own ``--clustering.num-clusters=0`` to the
  sherpa CLI, and per the CLI's --help a num-clusters > 0 IGNORES the
  threshold and wins outright. A wrapper that merely APPENDS
  ``--clustering.num-clusters=N`` is therefore a silent no-op — the first
  sweep proved that (byte-identical output for every N). The wrapper must
  REMOVE the engine's flag from "$@" before injecting ours (AC7).
* A BARE fixed num-clusters over-splits cleanly-separated audio (panel 2 -> 4
  at NC=8), so the ceiling may only rewrite the run when the auto pass
  ACTUALLY exceeds it. Below the ceiling the wrapper's stdout must stay
  byte-identical to the current single-pass behaviour.

The rule under test (implemented in scripts/sherpa-diarization-wrapper.sh,
which is the ONE place shared by the CUDA and Vulkan images):

  DIARIZATION_MAX_SPEAKERS (default 8, 0 = disabled):
    pass 1: run the real CLI exactly as before (threshold injected), capture
            stdout, count distinct speaker labels;
    <= ceiling (or unparseable/0 labels): replay pass-1 stdout untouched;
    > ceiling: re-run the real CLI with the engine's own
            --clustering.num-clusters STRIPPED and
            --clustering.num-clusters=<ceiling> injected.

All tests drive the real wrapper script against a stub SHERPA_REAL, so they
run hermetically in CI — no GPU, no models, no engine.
"""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts" / "sherpa-diarization-wrapper.sh"

# Exactly the argument shape the crispasr engine forwards today (observed in
# the engine's argv; num-clusters is ALWAYS present and ALWAYS 0).
ENGINE_ARGS = [
    "--clustering.num-clusters=0",
    "--segmentation.pyannote-model=/opt/models/seg.onnx",
    "--embedding.model=/opt/models/embed.onnx",
    "/tmp/input.wav",
]


def _stub(tmp_path: Path, auto_labels: int) -> Path:
    """Fake sherpa CLI.

    * appends its argv (one invocation per line) to $STUB_LOG;
    * emits ``auto_labels`` distinct speaker segments, unless a
      ``--clustering.num-clusters=N`` with N > 0 is in its argv -- then it
      emits exactly N, like the real CLI forcing that many clusters.
    """
    stub = tmp_path / "sherpa-real"
    stub.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\\n' "$*" >> "$STUB_LOG"
            nc=0
            for a in "$@"; do
              case "$a" in --clustering.num-clusters=*) nc="${a#*=}" ;; esac
            done
            n=$AUTOLABELS
            if [ "$nc" -gt 0 ] 2>/dev/null; then n=$nc; fi
            # real banner (measured stdout line 1): echoes the model PATHS,
            # and the shipped embedding name embeds "speaker_16" — a
            # substring count would inflate every file's label count by 1.
            echo "OfflineSpeakerDiarizationConfig(segmentation=OfflineSpeakerSegmentationModelConfig(pyannote=OfflineSpeakerSegmentationPyannoteModelConfig(model=\\"/opt/models/sherpa-onnx-reverb-diarization-v1.onnx\\", window_shift_ratio=0.1), num_threads=1, debug=False, provider=\\"cpu\\"), embedding=SpeakerEmbeddingExtractorConfig(model=\\"/opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx\\", num_threads=1, debug=False, provider=\\"cpu\\"), clustering=FastClusteringConfig(num_clusters=0, threshold=0.4, compute_confidence=False), min_duration_on=0.3, min_duration_off=0.5)"
            echo "Started"
            i=1
            while [ "$i" -le "$n" ]; do
              printf '%d.000 -- %d.500 speaker_%02d\\n' "$i" "$i" "$((i - 1))"
              i=$((i + 1))
            done
            """
        )
    )
    stub.chmod(0o755)
    # PosixPath rejects extra attributes; stash the auto-pass label count in
    # a side file that _run_wrapper feeds to the stub as $AUTOLABELS.
    Path(str(stub) + ".labels").write_text(str(auto_labels))
    return stub


def _run_wrapper(tmp_path: Path, stub: Path, extra_env: dict | None = None):
    env = {
        "PATH": os.environ["PATH"],
        "SHERPA_REAL": str(stub),
        "STUB_LOG": str(tmp_path / "calls.log"),
        "AUTOLABELS": Path(str(stub) + ".labels").read_text(),
    }
    if extra_env:
        env.update(extra_env)
    # script is 644 in git (Dockerfile chmods it in the image); drive it
    # through /bin/sh, which is exactly what its shebang selects.
    proc = subprocess.run(
        ["/bin/sh", str(WRAPPER), *ENGINE_ARGS],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return proc


def _calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "calls.log"
    if not log.is_file():
        return []
    return [line.split() for line in log.read_text().splitlines() if line.strip()]


def _distinct_labels(stdout: str) -> int:
    return len({tok for line in stdout.splitlines() for tok in line.split()
                if tok.startswith("speaker_")})


# ---------------------------------------------------------------------------
# AC1 — a diarization result exceeding the ceiling yields at most the ceiling
# ---------------------------------------------------------------------------

def test_auto_pass_exceeding_ceiling_is_capped_to_ceiling(tmp_path):
    """The 100-speaker explosion case: auto pass > ceiling => final stdout
    carries at most DIARIZATION_MAX_SPEAKERS distinct labels."""
    stub = _stub(tmp_path, 100)
    proc = _run_wrapper(tmp_path, stub)
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) <= 8, (
        f"ceiling not enforced: {_distinct_labels(proc.stdout)} labels survived"
    )


def test_ceiling_default_is_eight(tmp_path):
    """Unset env behaves as DIARIZATION_MAX_SPEAKERS=8."""
    stub = _stub(tmp_path, 9)  # just over the default ceiling
    proc = _run_wrapper(tmp_path, stub)
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) == 8
    calls = _calls(tmp_path)
    assert len(calls) == 2, f"expected auto pass + ceiling pass, got {len(calls)}"
    assert any("--clustering.num-clusters=8" in a for a in calls[1])


def test_env_var_raises_the_ceiling(tmp_path):
    stub = _stub(tmp_path, 12)
    proc = _run_wrapper(tmp_path, stub, {"DIARIZATION_MAX_SPEAKERS": "12"})
    # 12 auto labels with ceiling 12 => no rewrite at all.
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) == 12
    assert len(_calls(tmp_path)) == 1


def test_env_var_lowers_the_ceiling(tmp_path):
    stub = _stub(tmp_path, 12)
    proc = _run_wrapper(tmp_path, stub, {"DIARIZATION_MAX_SPEAKERS": "4"})
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) == 4
    calls = _calls(tmp_path)
    assert any("--clustering.num-clusters=4" in a for a in calls[-1])


def test_zero_disables_the_ceiling(tmp_path):
    """DIARIZATION_MAX_SPEAKERS=0 => exactly the pre-card behaviour: one
    invocation, engine args untouched, no num-clusters injection."""
    stub = _stub(tmp_path, 100)
    proc = _run_wrapper(tmp_path, stub, {"DIARIZATION_MAX_SPEAKERS": "0"})
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) == 100  # unchanged, ceiling off
    calls = _calls(tmp_path)
    assert len(calls) == 1
    assert any(a == "--clustering.num-clusters=0" for a in calls[0])
    assert not any(a.startswith("--clustering.num-clusters=") and a != "--clustering.num-clusters=0"
                   for a in calls[0])


# ---------------------------------------------------------------------------
# Clean-audio guarantee — below the ceiling the auto pass is replayed untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("labels", [1, 2, 3, 5, 8])
def test_auto_pass_at_or_below_ceiling_is_untouched(tmp_path, labels):
    """Panel/promo/marklex case: auto output <= 8 must NOT trigger any
    rewrite — a forced num-clusters would over-split it (sweep 2: panel
    2 -> 4 at NC=8)."""
    stub = _stub(tmp_path, labels)
    proc = _run_wrapper(tmp_path, stub)
    assert proc.returncode == 0, proc.stderr
    assert _distinct_labels(proc.stdout) == labels
    calls = _calls(tmp_path)
    assert len(calls) == 1, f"ceiling fired on compliant output: {calls}"
    assert any(a == "--clustering.num-clusters=0" for a in calls[0])
    assert any(a == "--clustering.cluster-threshold=0.40" for a in calls[0])


def test_threshold_injection_still_present(tmp_path):
    """Out-of-scope guard: the ceiling is ADDITIVE — cluster-threshold stays
    injected on every pass, including the ceiling pass (where the CLI simply
    ignores it because num-clusters > 0)."""
    stub = _stub(tmp_path, 20)
    proc = _run_wrapper(tmp_path, stub, {"SHERPA_CLUSTER_THRESHOLD": "0.35"})
    assert proc.returncode == 0, proc.stderr
    for call in _calls(tmp_path):
        assert "--clustering.cluster-threshold=0.35" in call


def test_pass_one_failure_propagates(tmp_path):
    """A failing auto pass must not silently fall into a ceiling pass: the
    wrapper exits non-zero with the CLI's stdout."""
    stub = tmp_path / "sherpa-real"
    stub.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            echo partial-output
            exit 3
            """
        )
    )
    stub.chmod(0o755)
    Path(str(stub) + ".labels").write_text("0")
    proc = _run_wrapper(tmp_path, stub)
    assert proc.returncode == 3, f"exit {proc.returncode}, stdout {proc.stdout!r}"
    assert "partial-output" in proc.stdout
    assert _calls(tmp_path) == []  # stub had no log hook; ceiling must not run


# ---------------------------------------------------------------------------
# AC7 — the exact failure mode of sweep 1: injection WITHOUT stripping the
# engine's own --clustering.num-clusters is a silent no-op.
# ---------------------------------------------------------------------------

def test_ceiling_pass_strips_engine_num_clusters_before_injecting(tmp_path):
    stub = _stub(tmp_path, 30)
    proc = _run_wrapper(tmp_path, stub)
    assert proc.returncode == 0, proc.stderr
    ceiling_call = _calls(tmp_path)[-1]
    injected = [a for a in ceiling_call if a.startswith("--clustering.num-clusters")]
    assert injected == ["--clustering.num-clusters=8"], (
        f"AC7 violated — engine flag not stripped / value not injected: {injected}"
    )


# ---------------------------------------------------------------------------
# AC2 / packaging / docs — the knob is wired identically through every layer
# ---------------------------------------------------------------------------

def test_dockerfile_sets_default_env(tmp_path):
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "DIARIZATION_MAX_SPEAKERS=8" in dockerfile


def test_both_compose_files_pass_the_env_through():
    for name in ("docker-compose.yml", "docker-compose.amd.yml"):
        compose = (REPO / name).read_text()
        assert "${DIARIZATION_MAX_SPEAKERS:-8}" in compose, f"{name} missing ceiling passthrough"


def test_wrapper_is_single_shared_layer_for_both_images():
    """CUDA and Vulkan images are one Dockerfile with ARG GPU_BACKEND, so a
    single COPY of the wrapper script carries the ceiling to BOTH images
    identically (AC8: one change, not two)."""
    dockerfile = (REPO / "Dockerfile").read_text()
    assert dockerfile.count("COPY scripts/sherpa-diarization-wrapper.sh") == 1
    assert "ARG GPU_BACKEND" in dockerfile
    wrapper = WRAPPER.read_text()
    assert "DIARIZATION_MAX_SPEAKERS" in wrapper


@pytest.mark.parametrize(
    "doc",
    ["README.md", "CONFIG.md", ".env.example", "autodiarist-speaker-count-ceiling.md"],
)
def test_knob_is_documented(doc):
    text = (REPO / doc).read_text()
    assert "DIARIZATION_MAX_SPEAKERS" in text, f"{doc} does not document the knob"


def test_docs_explain_how_to_disable(tmp_path):
    for doc in ("README.md", "CONFIG.md", ".env.example"):
        text = (REPO / doc).read_text()
        assert "DIARIZATION_MAX_SPEAKERS=0" in text or "0 disables" in text.lower(), (
            f"{doc} does not document disabling the ceiling"
        )


def test_changelog_mentions_the_ceiling():
    assert "DIARIZATION_MAX_SPEAKERS" in (REPO / "CHANGELOG.md").read_text()
