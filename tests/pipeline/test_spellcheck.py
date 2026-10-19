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


def test_short_terms_only_receive_exact_case_insensitive_fixes(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path, "NeMo\n")
    segments = [segment("nemo demo")]

    result = spell_check_segments(segments, vocab_path=vocab_path)

    assert result.segments[0].text == "NeMo demo"


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
