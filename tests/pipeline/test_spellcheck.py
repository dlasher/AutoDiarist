from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pipeline.chunking import Segment
from pipeline.spellcheck import spell_check_segments


def segment(text: str, *, t0: float = 1.0, t1: float = 2.0, speaker: str = "g1") -> Segment:
    return Segment(
        text=text,
        t0=t0,
        t1=t1,
        speaker=speaker,
        chunk_id=0,
        local_idx="0",
        match_confidence=1.0,
    )


def write_vocab(tmp_path: Path, contents: str) -> Path:
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(contents)
    return vocab_path


def test_exact_insensitive_casing_corrects_vocab_term_only(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "NeMo\n")
    segments = [segment("nemo pipelines")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "NeMo pipelines"
    assert result.n_fixed == 1
    assert result.examples == ["nemo -> NeMo"]
    assert result.segments[0].t0 == 1.0
    assert result.segments[0].t1 == 2.0


def test_distant_multiword_near_miss_is_left_verbatim(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "Kubernetes\n")
    segments = [segment("cuber neties")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "cuber neties"
    assert result.n_fixed == 0


def test_multiword_term_casing_preserves_segment_timing_and_count(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "Jane Wu\n")
    segments = [segment("jane wu said", t0=3.5, t1=5.25), segment("next", t0=6.0, t1=6.5)]
    original_timing = [(item.t0, item.t1) for item in segments]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert [item.text for item in result.segments] == ["Jane Wu said", "next"]
    assert len(result.segments) == len(segments)
    assert [(item.t0, item.t1) for item in result.segments] == original_timing


def test_non_vocab_words_are_never_altered(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "Kubernetes\n")
    segments = [segment("kubernetess customword")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "Kubernetes customword"
    assert "customword" in result.segments[0].text


def test_three_char_terms_only_receive_exact_case_insensitive_fixes(tmp_path: Path) -> None:
    """Near-miss correction starts at four-character terms: three-character
    acronyms stay exact-match-only so 'usa' is never fuzzy-routed to 'GSA'."""
    vocab_path = write_vocab(tmp_path, "GSA\n")
    segments = [segment("gsa usa")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "GSA usa"
    assert result.n_fixed == 1


def test_default_vocab_path_is_absolute_and_exists() -> None:
    """AC3: the shipped default resolves by absolute path, independent of CWD."""
    from pipeline.spellcheck import DEFAULT_VOCAB_PATH

    assert DEFAULT_VOCAB_PATH.is_absolute()
    assert DEFAULT_VOCAB_PATH.exists()
    text = DEFAULT_VOCAB_PATH.read_text(encoding="utf-8")
    for term in ("SEWP", "ITCAV", "GWAC", "OMB", "NASA", "contract holder"):
        assert term in text


def test_shipped_vocab_corrects_measured_error_classes() -> None:
    """AC2: the shipped vocab + rules fix the MEASURED error classes end to end
    through spell_check_segments with no explicit vocab_path."""
    shipped_segments = [
        segment("the gwec program", t0=1.0, t1=2.0),
        segment("sout contracts pricing", t0=3.0, t1=4.0),
        segment("its av pricing model", t0=5.0, t1=6.0),
        segment("soubcontracts list", t0=7.0, t1=8.0),
        segment("an acceptional result", t0=9.0, t1=10.0),
        segment("the contractholders list", t0=11.0, t1=12.0),
        segment("it is c c c c c c c c c cuture work", t0=13.0, t1=14.0),
    ]
    original_timing = [(item.t0, item.t1) for item in shipped_segments]

    result = spell_check_segments(shipped_segments)

    assert [item.text for item in result.segments] == [
        "the GWAC program",
        "SEWP contracts pricing",
        "ITCAV pricing model",
        "soup contracts list",
        "an exceptional result",
        "the contract holders list",
        "it is cuture work",
    ]
    assert [(item.t0, item.t1) for item in result.segments] == original_timing
    assert result.n_fixed == 7


def test_shipped_vocab_leaves_clean_domain_text_unchanged() -> None:
    """AC7: correct domain text comes out byte-identical."""
    clean = (
        "The SEWP program at NASA works with contract holders under NASA SEWP "
        "policy, referencing the ITCAV, GWAC, and OMB guidance in an "
        "exceptional soup contracts year."
    )

    result = spell_check_segments([segment(clean)])

    assert result.segments[0].text == clean
    assert result.n_fixed == 0


def test_stutter_run_with_more_than_two_repeats_is_left_verbatim(tmp_path: Path) -> None:
    """Conservative stutter guard: natural repetition like 'no no no' survives."""
    vocab_path = write_vocab(tmp_path, "SEWP\n")
    segments = [segment("no no no"), segment("the a a a b"), segment("c c c cuture now")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert [item.text for item in result.segments] == ["no no no", "the a b", "cuture now"]


def test_alias_lines_fix_measured_garbles(tmp_path: Path) -> None:
    """Vocab lines 'garble => Canonical' carry the MEASURED error forms."""
    vocab_path = write_vocab(tmp_path, "sout, soupes => SEWP\n")

    result = spell_check_segments([segment("sout and soupes pricing")], vocab_path=vocab_path)

    assert result.segments[0].text == "SEWP and SEWP pricing"
    assert result.n_fixed == 2


def test_malformed_alias_line_is_warning_logged_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vocab_path = write_vocab(tmp_path, "SEWP\nsout => SEWP\n=> broken\nsout2 =>\n")

    with caplog.at_level(logging.WARNING):
        result = spell_check_segments([segment("sout pricing")], vocab_path=vocab_path)

    assert result.segments[0].text == "SEWP pricing"
    assert "malformed" in caplog.text.lower()


def test_join_and_split_rules_may_change_word_count_but_keep_timing(tmp_path: Path) -> None:
    """AC5 (stated change): term-boundary join/split and stutter collapse are
    the only word-count reducers, and split grows; segment timing is kept."""
    vocab_path = write_vocab(tmp_path, "ITCAV\ncontract holder\n")
    segments = [segment("its av pricing", t0=2.5, t1=4.0), segment("the contractholder", t0=5.0, t1=6.25)]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert [item.text for item in result.segments] == ["ITCAV pricing", "the contract holder"]
    assert [(item.t0, item.t1) for item in result.segments] == [(2.5, 4.0), (5.0, 6.25)]


def test_four_char_terms_are_near_miss_corrected(tmp_path: Path) -> None:
    """AC4: the >=6 floor excluded the measured 4-char acronyms; >=4 catches them."""
    vocab_path = write_vocab(tmp_path, "GWAC\n")

    result = spell_check_segments([segment("the gwec program")], vocab_path=vocab_path)

    assert result.segments[0].text == "the GWAC program"


def test_near_miss_stays_unique_match_only(tmp_path: Path) -> None:
    """Ambiguous distance-one candidates are never corrected (safety AC6)."""
    vocab_path = write_vocab(tmp_path, "SEWP\nSEWQ\n")

    result = spell_check_segments([segment("sewr pricing")], vocab_path=vocab_path)

    assert result.segments[0].text == "sewr pricing"
    assert result.n_fixed == 0


def test_common_prose_is_never_join_corrected() -> None:
    """AC6/AC7 guards proven necessary by the corpus probe (card t_72405565):
    distance-one joins refuse windows of ordinary words ('as a'/'was a' are
    distance one from NASA, 'it can' from ITCAV) and unequal-length joined
    spans, so articles and singular/plural prose survive."""
    shipped_segments = [
        segment("it was a concern as a rule"),
        segment("it can run"),
        segment("a contract holder list"),
        segment("one soup contract remains"),
        segment("its value was noted"),
    ]

    result = spell_check_segments(shipped_segments)

    assert [item.text for item in result.segments] == [
        "it was a concern as a rule",
        "it can run",
        "a contract holder list",
        "one soup contract remains",
        "its value was noted",
    ]
    assert result.n_fixed == 0


def test_lowercase_canonical_terms_keep_source_case() -> None:
    """All-lowercase canonical terms are case-preserving (no correct-text
    damage to capitalized or shouted prose); capitalized canonicals (the
    acronyms) are emitted exactly as spelled."""
    shipped_segments = [
        segment("Contract Holders are listed"),
        segment("CONTRACT HOLDERS only"),
        segment("nasa and sewp guidance"),
    ]

    result = spell_check_segments(shipped_segments)

    assert [item.text for item in result.segments] == [
        "Contract Holders are listed",
        "CONTRACT HOLDERS only",
        "NASA and SEWP guidance",
    ]


def test_underscored_merged_terms_are_supported(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "NVIDIA_NeMo\n")
    segments = [segment("nvidia_nemo works")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "NVIDIA_NeMo works"


def test_disabled_pass_returns_original_text_and_zero_statistics(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "NeMo\n")
    segments = [segment("nemo pipelines")]

    result = spell_check_segments(segments, vocab_path=vocab_path, enabled=False)

    assert result.segments[0].text == "nemo pipelines"
    assert result.n_fixed == 0
    assert result.examples == []


def test_missing_or_malformed_vocab_is_warning_logged_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing_path = tmp_path / "missing-vocab.txt"

    with caplog.at_level(logging.WARNING):
        missing_result = spell_check_segments([segment("nemo")], vocab_path=missing_path)

    assert missing_result.segments[0].text == "nemo"
    assert "vocabulary" in caplog.text.lower()

    malformed_path = write_vocab(tmp_path, "\nJane Wu\nJane   Wu\n")
    with caplog.at_level(logging.WARNING):
        malformed_result = spell_check_segments([segment("jane wu")], vocab_path=malformed_path)

    assert malformed_result.segments[0].text == "Jane Wu"
    assert "malformed" in caplog.text.lower()
