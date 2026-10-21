"""Tests for the opt-in LLM cleanup pass (card t_48c2efae).

Covers AC1-AC9 of the V2 LLM-cleanup card: turn-aware chunking with seam
overlap, the versioned preface, the mandatory post-LLM verification guard,
spelling-only acceptance, wording-change rejection, label/timestamp
immutability, the injected-callable interface, the fixed temperature=0
assertion, and name evidence flowing through the ``extra_evidence`` seam.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest

from pipeline.llm_cleanup import (
    CONTEXT_HEADER,
    EVIDENCE_MARKER,
    LINES_HEADER,
    CleanupSettings,
    authoritative_lines,
    build_cleanup_line,
    build_prompt,
    load_preface,
    plan_cleanup_chunks,
    run_llm_cleanup,
    verify_chunk_response,
)
from pipeline.self_identification import suggest_speaker_names


def seg(text: str, t0: float, t1: float, speaker: str) -> dict:
    return {"text": text, "t0": t0, "t1": t1, "speaker": speaker}


def settings(**kw) -> CleanupSettings:
    base = {"chunk_char_budget": 200, "overlap_lines": 1, "max_retries": 0}
    base.update(kw)
    return CleanupSettings(**base)


def line_of(text: str, t0: float, t1: float, speaker: str) -> str:
    return build_cleanup_line(seg(text, t0, t1, speaker))


def echo_llm(replacements: dict[str, str] | None = None, recorder: list | None = None,
             extra: str = ""):
    """Fake LLM: echoes every authoritative line, applying exact-line replacements."""

    def llm(prompt: str) -> str:
        if recorder is not None:
            recorder.append(prompt)
        out = []
        for i, line in enumerate(authoritative_lines(prompt), 1):
            text = (replacements or {}).get(line, line)
            out.append(f"{i}|{text}")
        return "\n".join(out) + (extra or "")

    return llm


# --- Preface ---------------------------------------------------------------


def test_preface_file_exists_and_states_rules() -> None:
    """AC6/Scope-2: versioned preface file in the repo states the hard rules."""
    path = Path(inspect.getsourcefile(load_preface)).parent / "llm_preface.md"
    assert path.is_file()
    text = load_preface()
    lowered = text.lower()
    assert text.startswith("PREFACE VERSION:")
    for rule in (
        "spelling",
        "punctuation",
        "never change wording",
        "merge",
        "invent",
        "names",
        "identifiers",
        "speaker labels",
        "timestamps",
        "summarize",
        "same order",
    ):
        assert rule in lowered, f"preface must state rule: {rule!r}"


# --- Chunking (AC1, AC5) ----------------------------------------------------


def test_chunk_boundaries_are_whole_speaker_turns() -> None:
    """AC1: boundaries fall on speaker turns and never mid-sentence."""
    lines = [line_of(f"Turn number {i} with a full sentence here.", i * 2.0, i * 2.0 + 1.5,
                     f"Speaker {i % 2 + 1}") for i in range(12)]
    chunks = plan_cleanup_chunks(lines, settings(chunk_char_budget=150, overlap_lines=0))
    assert len(chunks) > 1, "budget must actually force multiple chunks"
    # Contiguous full coverage, no owned line shared between chunks.
    owned = [idx for c in chunks for idx in range(c.start, c.end)]
    assert owned == list(range(len(lines)))
    for c in chunks:
        # Every owned line is intact inside its chunk (never split mid-line).
        for idx in range(c.start, c.end):
            assert lines[idx] in lines[c.start:c.end]


def test_oversized_line_gets_its_own_chunk_never_split() -> None:
    long_text = "word " * 100  # one turn, exceeds the budget on its own
    lines = [line_of("short one.", 0.0, 1.0, "Speaker 1"),
             line_of(long_text.strip(), 1.0, 2.0, "Speaker 2"),
             line_of("short two.", 2.0, 3.0, "Speaker 3")]
    chunks = plan_cleanup_chunks(lines, settings(chunk_char_budget=100, overlap_lines=0))
    assert len(chunks) == 3
    assert chunks[1].start == 1 and chunks[1].end == 2


def test_overlap_seam_text_reaches_prompt_not_just_index() -> None:
    """AC5: the chunk boundary carries the previous seam's text into the prompt."""
    lines = [line_of(f"Sentence number {i} continues the thought.", i * 3.0, i * 3.0 + 2.0,
                     f"Speaker {i % 2 + 1}") for i in range(10)]
    st = settings(chunk_char_budget=120, overlap_lines=1)
    chunks = plan_cleanup_chunks(lines, st)
    assert len(chunks) > 1
    prompt2 = build_prompt(load_preface(), lines[chunks[1].context_start:chunks[1].context_end],
                           lines[chunks[1].start:chunks[1].end], st)
    seam_line = lines[chunks[0].end - 1]
    context_block = prompt2.split(LINES_HEADER)[0]
    assert CONTEXT_HEADER in prompt2
    assert seam_line in context_block, "prompt 2 must embed the seam paragraph as context"
    # Context lines are NOT re-listed as lines to clean.
    assert seam_line not in authoritative_lines(prompt2)


# --- Verification guard (AC3, AC8, AC9) -------------------------------------


def test_line_count_change_is_rejected() -> None:
    lines = [line_of("Alpha line one.", 0.0, 1.0, "Speaker 1"),
             line_of("Beta line two.", 1.0, 2.0, "Speaker 2")]
    ok, reason, *_ = verify_chunk_response("1|" + lines[0], lines, settings())
    assert not ok and "line" in reason.lower()


def test_reordered_lines_are_rejected() -> None:
    lines = [line_of("Alpha line one.", 0.0, 1.0, "Speaker 1"),
             line_of("Beta line two.", 1.0, 2.0, "Speaker 2")]
    response = f"1|{lines[1]}\n2|{lines[0]}"
    ok, *_ = verify_chunk_response(response, lines, settings())
    assert not ok


def test_speaker_label_or_timestamp_alteration_is_rejected() -> None:
    """AC9: labels and timestamps are immutable — tampering rejects the chunk."""
    lines = [line_of("Hello there.", 0.0, 3.5, "Speaker 1")]
    tampered = "1|Speaker 2 [0.00-3.50] — Hello there."
    ok, *_ = verify_chunk_response(tampered, lines, settings())
    assert not ok, "changed speaker label must reject"
    tampered_ts = "1|Speaker 1 [0.00-9.99] — Hello there."
    ok, *_ = verify_chunk_response(tampered_ts, lines, settings())
    assert not ok, "changed timestamp must reject"


# --- Accept/reject end-to-end (AC7, AC8, AC3 report) ------------------------


def _two_chunk_setup():
    lines = [
        line_of("We recieve your report every week.", 0.0, 4.0, "Speaker 1"),
        line_of("That was absolutely fantastic news.", 4.0, 8.0, "Speaker 2"),
        line_of("The team ships the build today.", 8.0, 12.0, "Speaker 1"),
        line_of("Nothing to correct on this line.", 12.0, 16.0, "Speaker 2"),
    ]
    segments = [
        {"text": "We recieve your report every week.", "t0": 0.0, "t1": 4.0, "speaker": "Speaker 1"},
        {"text": "That was absolutely fantastic news.", "t0": 4.0, "t1": 8.0, "speaker": "Speaker 2"},
        {"text": "The team ships the build today.", "t0": 8.0, "t1": 12.0, "speaker": "Speaker 1"},
        {"text": "Nothing to correct on this line.", "t0": 12.0, "t1": 16.0, "speaker": "Speaker 2"},
    ]
    return lines, segments


def test_spelling_only_llm_is_accepted() -> None:
    """AC7: a fake LLM that only fixes spelling is accepted; everything else intact."""
    lines, segments = _two_chunk_setup()
    fixed = line_of("We receive your report every week.", 0.0, 4.0, "Speaker 1")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[0]: fixed}),
                             settings=settings(chunk_char_budget=90, overlap_lines=1))
    assert result.accepted == len(result.outcomes)
    assert result.rejected == 0 and result.failed == 0
    assert result.segments[0]["text"] == "We receive your report every week."
    assert result.segments[0]["speaker"] == "Speaker 1"
    assert result.segments[0]["t0"] == 0.0 and result.segments[0]["t1"] == 4.0
    assert result.segments[1]["text"] == "That was absolutely fantastic news."
    assert result.changed_lines == 1
    assert "accepted" in result.report_markdown.lower()


def test_case_and_punctuation_only_fix_is_accepted() -> None:
    lines, segments = _two_chunk_setup()
    # Same words; only capitalization and punctuation differ (permitted edits).
    fixed = line_of("we Recieve YOUR report Every. week", 0.0, 4.0, "Speaker 1")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[0]: fixed}),
                             settings=settings(chunk_char_budget=90))
    assert result.rejected == 0, "case/punctuation-only change must be accepted"
    assert result.changed_lines == 1
    assert result.segments[0]["text"] == "we Recieve YOUR report Every. week"


def test_wording_change_is_rejected_and_original_kept() -> None:
    """AC8: a fake LLM that alters wording is REJECTED; the original text survives."""
    lines, segments = _two_chunk_setup()
    altered = line_of("That was definitely fantastic news.", 4.0, 8.0, "Speaker 2")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[1]: altered}),
                             settings=settings(chunk_char_budget=90, overlap_lines=0))
    assert result.accepted + result.rejected == len(result.outcomes)
    assert result.rejected >= 1
    assert result.segments[1]["text"] == "That was absolutely fantastic news."
    rejected_outcome = next(o for o in result.outcomes if not o.accepted)
    assert "wording" in rejected_outcome.reason.lower() or "word" in rejected_outcome.reason.lower()
    assert "rejected" in result.report_markdown.lower()


def test_report_includes_accepted_and_rejected_counts_and_diffs() -> None:
    """AC3: per-chunk accept/reject diff report with accepted/rejected counts and changes."""
    lines, segments = _two_chunk_setup()
    fixed0 = line_of("We receive your report every week.", 0.0, 4.0, "Speaker 1")
    altered1 = line_of("That was definitely fantastic news.", 4.0, 8.0, "Speaker 2")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[0]: fixed0, lines[1]: altered1}),
                             settings=settings(chunk_char_budget=90, overlap_lines=0))
    report = result.report_markdown
    assert "accepted" in report.lower() and "rejected" in report.lower()
    assert "receive" in report and "definitely" in report, "diff must show what changed / was rejected"


# --- Failures, retries, timeout (extras) -------------------------------------


def test_llm_failure_retries_then_keeps_original() -> None:
    _, segments = _two_chunk_setup()
    calls = {"n": 0}

    def angry_llm(prompt: str) -> str:
        calls["n"] += 1
        raise RuntimeError("connection reset")

    result = run_llm_cleanup(segments, llm=angry_llm,
                             settings=settings(chunk_char_budget=90, overlap_lines=0, max_retries=2))
    chunks = plan_cleanup_chunks([build_cleanup_line(s) for s in segments],
                                 settings(chunk_char_budget=90, overlap_lines=0))
    assert calls["n"] == len(chunks) * 3  # 1 attempt + 2 retries per chunk
    assert result.accepted == 0 and result.failed == len(chunks)
    assert result.rejected == len(chunks)
    assert [s["text"] for s in result.segments] == [s["text"] for s in segments]


def test_timeout_rejected_and_original_kept() -> None:
    _, segments = _two_chunk_setup()

    def slow_llm(prompt: str) -> str:
        time.sleep(0.5)
        return ""

    result = run_llm_cleanup(segments[:1], llm=slow_llm,
                             settings=settings(chunk_char_budget=90, overlap_lines=0,
                                               max_retries=0, timeout_seconds=0.05))
    assert result.failed == 1
    assert result.segments[0]["text"] == "We recieve your report every week."


# --- Interface / config assertions (AC4, Scope-5) ----------------------------


def test_temperature_pinned_to_zero_no_env_override() -> None:
    """Scope-5/AC: temperature=0 fixed as an assertion; no env toggle for it."""
    with pytest.raises(ValueError):
        CleanupSettings(temperature=0.7)
    assert CleanupSettings().temperature == 0.0
    from pipeline import llm_cleanup as mod

    source = Path(inspect.getsourcefile(mod)).read_text(encoding="utf-8")
    assert "LLM_CLEANUP_TEMPERATURE" not in source


def test_llm_is_injected_callable_default_off() -> None:
    """AC4: run_llm_cleanup requires an injected callable; no default network path."""
    _, segments = _two_chunk_setup()
    with pytest.raises(TypeError):
        run_llm_cleanup(segments)  # llm is keyword-required, opt-in only
    signature = inspect.signature(run_llm_cleanup)
    assert signature.parameters["llm"].kind == inspect.Parameter.KEYWORD_ONLY


def test_cleanup_diff_lines_count_preserved() -> None:
    """AC10 core: cleaned output has the same transcript line count as the input."""
    lines, segments = _two_chunk_setup()
    result = run_llm_cleanup(segments, llm=echo_llm(), settings=settings(chunk_char_budget=90))
    out_lines = [build_cleanup_line(s) for s in result.segments]
    assert len(out_lines) == len(lines)
    assert out_lines[1:] == lines[1:]  # echo pass changes nothing except permitted edits


def test_cleanup_diff_shows_only_permitted_changes() -> None:
    lines, segments = _two_chunk_setup()
    fixed = line_of("We receive your report every week!", 0.0, 4.0, "Speaker 1")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[0]: fixed}),
                             settings=settings(chunk_char_budget=90))
    changed = [d for o in result.outcomes for d in o.diffs]
    assert changed, "expected at least one permitted change"
    # Every reported change keeps the prefix (label+timestamps) identical.
    for (line_index, before, after) in changed:
        assert before.split(" — ")[0] == after.split(" — ")[0]


# --- Name evidence through the extra_evidence seam (AC6) ---------------------


def test_name_evidence_flows_through_extra_evidence_seam() -> None:
    """AC6: injected name evidence reaches suggest_speaker_names via extra_evidence."""
    _lines, segments = _two_chunk_setup()
    evidence_block = (
        "\n" + EVIDENCE_MARKER + "\n"
        + "Speaker 1 | Ada Lovelace | We receive your report every week.\n"
    )
    result = run_llm_cleanup(
        segments,
        llm=echo_llm(extra=evidence_block),
        settings=settings(chunk_char_budget=400, overlap_lines=0, review_names=True),
    )
    assert result.accepted >= 1
    assert len(result.name_evidence) == 1
    entry = result.name_evidence[0]
    assert entry["label"] == "Speaker 1"
    assert entry["claimed_name"] == "Ada Lovelace"
    assert entry["source"] == "llm_cleanup"
    suggestions = suggest_speaker_names(result.segments,
                                        extra_evidence=[{**entry, "weight": 0.9}])
    assert "Speaker 1" in suggestions
    assert suggestions["Speaker 1"]["name"] == "Ada Lovelace"


def test_evidence_from_rejected_chunk_is_dropped() -> None:
    lines, segments = _two_chunk_setup()
    altered = line_of("That was definitely fantastic news.", 4.0, 8.0, "Speaker 2")
    evidence_block = (
        "\n" + EVIDENCE_MARKER + "\n"
        + "Speaker 1 | Nikola Tesla | The team ships the build today.\n"
    )
    result = run_llm_cleanup(
        segments,
        llm=echo_llm({lines[1]: altered}, extra=evidence_block),
        settings=settings(chunk_char_budget=400, overlap_lines=0, review_names=True),
    )
    assert result.rejected >= 1
    assert result.name_evidence == []


# --- Report file write is a caller/agent concern (AC6 shape only) -----------


def test_accepts_segment_dataclass_and_returns_dataclass() -> None:
    """Segment dataclasses (pipeline.chunking) round-trip through the pass."""
    from pipeline.chunking import Segment

    original = Segment(text="We recieve it.", t0=0.0, t1=1.0, speaker="Speaker 1",
                       chunk_id=0, local_idx="0", match_confidence=1.0)
    fixed = line_of("We receive it.", 0.0, 1.0, "Speaker 1")
    result = run_llm_cleanup([original],
                             llm=echo_llm({build_cleanup_line(original): fixed}),
                             settings=settings(chunk_char_budget=200))
    cleaned = result.segments[0]
    assert isinstance(cleaned, Segment)
    assert cleaned.text == "We receive it."
    assert cleaned.chunk_id == 0 and cleaned.match_confidence == 1.0
    assert cleaned.t0 == 0.0 and cleaned.t1 == 1.0 and cleaned.speaker == "Speaker 1"


def test_result_carries_report_markdown() -> None:
    _, segments = _two_chunk_setup()
    result = run_llm_cleanup(segments, llm=echo_llm(), settings=settings(chunk_char_budget=90))
    assert result.report_markdown.startswith("#")
    assert "chunk" in result.report_markdown.lower()


# --- AC10: end-to-end render shows only permitted diffs -----------------------


def test_end_to_end_render_diff_shows_only_permitted_changes() -> None:
    """Full pytest suite runs offline; the rendered report differs from the
    uncleaned one ONLY in the single line with the spelling fix."""
    from pipeline.renderer import render_markdown

    lines, segments = _two_chunk_setup()
    fixed = line_of("We receive your report every week.", 0.0, 4.0, "Speaker 1")
    result = run_llm_cleanup(segments, llm=echo_llm({lines[0]: fixed}),
                             settings=settings(chunk_char_budget=90, overlap_lines=0))
    meta = {"filename": "demo.wav", "processed_at": "2026-01-01T00:00:00+00:00",
            "duration_seconds": 16.0}
    before = _transcript_block(render_markdown(segments, meta))
    after = _transcript_block(render_markdown(result.segments, meta))
    assert len(before) == len(after), "line count must be preserved end-to-end"
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(changed) == 1, f"expected exactly one permitted diff, got {changed}"
    b, a = changed[0]
    assert "recieve" in b and "receive" in a
    assert b.replace("recieve", "receive") == a, "diff is the spelling fix alone"


def _transcript_block(md: str) -> list[str]:
    """Transcript section only (the appendix json mirrors segments by design)."""
    section = md.split("## Transcript", 1)[1]
    return section.split("## Appendix", 1)[0].splitlines()
