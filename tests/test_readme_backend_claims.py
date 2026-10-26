"""README backend-claim tests (card t_ee479fdf).

The "Both backends produced identical output" claim was FALSIFIED by direct
measurement on the matched build b0335f9 (same 137.5 s clip, same app layer,
both hosts on the same commit):

    NV  text_sha 811e0c9b8cc81ee0   8 segments, 2 speakers, 2213 chars
    AMD text_sha 11a24a7b2b728066   8 segments, 2 speakers, 2220 chars
    Word-level similarity 0.9442

Segment and speaker COUNTS are identical; the transcript TEXT is not. The
remainder is CUDA-vs-Vulkan backend ASR variation (same model, same
quantisation, different GPU arithmetic) — not a diarization or spell-check
difference. These tests pin the corrected wording so the falsified claim
cannot silently come back, and pin the service-level concurrency curve for
BOTH hosts (AC 1-9 of the card).

All prose checks run against the section with whitespace FLATTENED, so the
README can keep its ~79-column hard wrapping without breaking a phrase
match (and a claim split across a line break is still caught).
"""

import re
from pathlib import Path

import pytest

README_PATH = Path(__file__).resolve().parents[1] / "README.md"
README = README_PATH.read_text(encoding="utf-8")

_AB_START = "### AMD vs NVIDIA — measured A/B"
_AB_END = "### Long-form inputs"


def _ab_flat() -> str:
    """The A/B region (headings through the Notes list, everything this
    card touches), whitespace-flattened."""
    start = README.index(_AB_START)
    end = README.index(_AB_END, start)
    return re.sub(r"\s+", " ", README[start:end])


def _ab_low() -> str:
    return _ab_flat().lower()


def test_ab_section_present_and_wellformed():
    assert _AB_START in README
    assert _AB_END in README
    assert README.index(_AB_START) < README.index(_AB_END)


# AC 1 — the falsified claim must be gone from BOTH places, phrased as a
# prohibition so any reintroduction ("**identical output**", "identical
# output shape", line-wrapped variants included) fails the same way.
def test_no_identical_output_claim_in_ab_section():
    low = _ab_low()
    assert "identical output" not in low, (
        "The outputs are NOT identical (word similarity 0.9442 on the matched "
        "build); the README must distinguish identical segment/speaker COUNTS "
        "from non-identical transcript TEXT. See card t_ee479fdf."
    )
    assert "difference is throughput, not transcript content" not in low, (
        "The falsified framing: the difference is not only throughput."
    )


# AC 1 (second half) — positive wording: counts identical, text not.
def test_counts_vs_text_distinction_stated():
    low = _ab_low()
    assert "segment" in low and "speaker" in low
    assert "text" in low
    assert "not the same" in low or "not identical" in low


# AC 2 — the measured similarity figure and the nature of the difference.
def test_similarity_figure_and_nature_stated():
    section = _ab_flat()
    low = section.lower()
    assert "0.9442" in section
    assert "~94%" in section
    assert "asr variation" in low
    assert "cuda" in low and "vulkan" in low
    # explicitly NOT a diarization/spell-check/structural difference
    assert "not a diarization" in low
    assert "spell-check" in low
    assert "not a structural" in low
    # both measured text shas, so a reader can re-find the runs
    assert "811e0c9b8cc81ee0" in section
    assert "11a24a7b2b728066" in section


# AC 3 — service-level concurrency curve, BOTH hosts, with in-flight,
# throughput, and the safe worker count per host.
def test_service_concurrency_curve_published_for_both_hosts():
    section = _ab_flat()
    assert "### Concurrency — measured service-level curve" in section
    # AMD curve (measured to N=4)
    for wall, thr in [("41.3 s", "0.024"), ("47.9 s", "0.042"),
                      ("49.5 s", "0.061"), ("57.2 s", "0.070")]:
        assert wall in section, wall
        assert thr in section, thr
    # NVIDIA curve (measured to N=3)
    for wall, thr in [("79.3 s", "0.013"), ("92.6 s", "0.022"),
                      ("108.9 s", "0.028")]:
        assert wall in section, wall
        assert thr in section, thr
    # in-flight reached N at every rung, on both hosts
    assert "in-flight reached n" in section.lower()
    # matched-build provenance
    assert "b0335f9" in section
    assert "137.5 s" in section
    # the safe worker count per host, stated in one sentence
    m = re.search(r"[Ss]afe worker count[^.]*\.", section)
    assert m, "no sentence states the safe worker count per host"
    sentence = m.group(0)
    assert "AMD" in sentence and "N=4" in sentence
    assert "NVIDIA" in sentence and "N=3" in sentence


# AC 3 (nature) — the shape of the curve: throughput climbs while per-job
# latency degrades.
def test_curve_shape_stated():
    low = _ab_low()
    assert "throughput climbs" in low
    assert "per-job" in low
    assert "latency rises" in low


# AC 4 — the NVIDIA N=3 ceiling and its CAUSE, with the verbatim error.
def test_nvidia_ceiling_and_cause():
    low = _ab_low()
    assert "sched alloc encoder graph failed" in low
    assert "nemotron" in low
    assert "vram" in low
    assert "3/4 in every wave" in low
    # the 4096 / ~1069 MiB numbers that make it a VRAM, not CPU, ceiling
    assert "1069" in low
    assert "4096" in low


# AC 5 — AMD-specific caveat: VRAM_LIMIT_MIB is inert on the AMD image.
def test_amd_vram_limit_inert():
    section = _ab_flat()
    low = section.lower()
    assert "VRAM_LIMIT_MIB" in section
    assert "inert" in low
    assert "nvidia-smi" in low
    assert "sampler" in low
    # and why there is no VRAM ceiling there at all: UMA / system RAM
    assert "UMA" in section


# AC 6 — thread-count-vs-worker-count interaction with the measured example
# and the explicit escape hatch.
def test_thread_worker_interaction_documented():
    section = _ab_flat()
    low = section.lower()
    assert "SHERPA_NUM_THREADS" in section
    assert "MAX_PROCESSING_GLOBAL" in section
    assert "43 s" in section
    assert "78.7 s" in section
    # the rule itself, as the shipped wrapper computes it
    assert "cores // max_processing_global" in low


# AC 7 — additive except the falsified claim: the prior measurements survive.
def test_existing_measurements_preserved():
    section = _ab_flat()
    # the original t_97bbd597 A/B speed table (untouched)
    assert "222.7 s" in section and "105.4 s" in section
    assert "78.4 s" in section and "39.2 s" in section
    # the matched-build (000bc2f) prose and its supersession note
    assert "000bc2f" in section
    assert "3.66x" in section and "3.69x" in section


# AC 7 (second half) — link to the thread curves, do not duplicate them.
def test_thread_curve_link_not_duplicate():
    section = _ab_flat()
    assert "CONFIG.md" in section
    assert "t_71a5768f" in section
    # the thread-curve tables (1t..12t rows) live in "Diarization threading"
    # above and in CONFIG.md — not duplicated into the A/B region
    assert "110.4 s" not in section
    config = (README_PATH.parent / "CONFIG.md").read_text(encoding="utf-8")
    assert "110.4 s" in config


# AC 3 (consistency) — the feature-summary "Global concurrency" section must
# not keep calling the AMD value unmeasured now that both curves ship here.
def test_amd_concurrency_not_claimed_unmeasured():
    assert "still to be measured" not in README
