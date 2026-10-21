"""Regression tests for GitHub issue dlasher/AutoDiarist#1 (card t_16c871a6).

Two capture defects in the self-introduction pass:

1. Greedy ``_LOWER_NAME`` capture accepted any casing, so ordinary prose
   matched as a name (``re.IGNORECASE`` in ``_compile`` defeated the
   capitalized ``_NAME`` requirement).
2. A single trailing function word rejected the whole match instead of
   retrying with a shorter capture, so valid names like "Shelley Festing"
   were discarded.

These tests cover the acceptance criteria: lowercase captures are rejected,
capitalized self-intros across all four phrases still match, the issue's
prose false-positive class yields zero suggestions, function-word truncation
yields a shorter valid name, rejection happens only at the single-word floor,
and env-override patterns keep the documented case-insensitive lead-phrase
guarantee with a case-sensitive name capture.
"""

from __future__ import annotations

from pipeline.chunking import Segment
from pipeline.self_identification import (
    DEFAULT_SELF_ID_PATTERNS,
    _compile,
    suggest_speaker_names,
)


def segment(
    text: str,
    *,
    speaker: str,
    t0: float,
    t1: float,
) -> Segment:
    return Segment(text=text, t0=t0, t1=t1, speaker=speaker)


# --- AC: all four self-intro phrases capture capitalized names (lead phrase
# --- casing is free; the name itself must be Title-case words).


def test_all_self_intro_phrases_capture_capitalized_names() -> None:
    segments = [
        segment("MY NAME IS Jane Wu.", speaker="g1", t0=0.0, t1=1.0),
        segment("I'm Ada Lovelace, from the second cohort.", speaker="g2", t0=1.1, t1=2.0),
        segment("I am Grace Hopper.", speaker="g3", t0=2.1, t1=3.0),
        segment("This is Katherine Johnson.", speaker="g4", t0=3.1, t1=4.0),
        segment("Annie Easley here.", speaker="g5", t0=4.1, t1=5.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert {label: item["name"] for label, item in suggestions.items()} == {
        "g1": "Jane Wu",
        "g2": "Ada Lovelace",
        "g3": "Grace Hopper",
        "g4": "Katherine Johnson",
        "g5": "Annie Easley",
    }


# --- AC: lowercase captures are rejected (the issue's false-positive class).


def test_lowercase_self_intro_capture_is_rejected() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("So my name is rajiv is from the second cohort.", speaker="g2", t0=1.1, t1=3.0),
    ]

    assert suggest_speaker_names(segments) == {}


def test_capitalized_self_intro_still_matches_single_word_name() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("My name is Rajiv.", speaker="g2", t0=1.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Rajiv"


# --- AC: trailing function words no longer reject a valid capture. The
# --- strict grammar drops lowercase prose words from the capture itself...


def test_lowercase_prose_tail_does_not_reject_the_name() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment(
            "My name is Shelley Festing the rest of the agenda is short.",
            speaker="g2",
            t0=1.1,
            t1=3.0,
        ),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Shelley Festing"


# --- ...and the truncation retry rescues captured function words (e.g. from
# --- ASR casing quirks) by shortening the capture from the tail.


def test_capitalized_trailing_function_word_truncates_to_valid_name() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("My name is Shelley Festing The.", speaker="g2", t0=1.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Shelley Festing"


def test_backoff_continues_until_remainder_is_clean() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("I'm Rajiv And Then we will start.", speaker="g2", t0=1.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Rajiv"


# --- AC: rejection only when the sole captured word is a function word.


def test_single_function_word_floor_is_rejected() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is The next topic is queued.", speaker="g2", t0=1.1, t1=3.0),
    ]

    assert suggest_speaker_names(segments) == {}


# --- AC: truncation is implemented independently of capitalization. A custom
# --- loose class ([A-Za-z]+) still captures lowercase prose words even under
# --- (?-i:) scoping; the retry must still shorten to the valid name.


def test_truncation_retry_is_independent_of_capitalization(monkeypatch) -> None:
    monkeypatch.setenv(
        "SELF_ID_PATTERNS", r"call me (?P<name>[A-Za-z]+(?: [A-Za-z]+){0,2})"
    )
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment(
            "Please call me Shelley Festing the agenda is short.",
            speaker="g2",
            t0=1.1,
            t1=3.0,
        ),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Shelley Festing"


# --- AC: env-override patterns keep the documented case-insensitivity
# --- guarantee for the lead phrase while the name capture stays case-
# --- sensitive (compiled under re.IGNORECASE, so [A-Z] must not fold).


def test_env_custom_pattern_keeps_case_insensitive_lead_and_strict_capture(monkeypatch) -> None:
    monkeypatch.setenv(
        "SELF_ID_PATTERNS", r"call me (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+){0,2})"
    )

    lowercase_name = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("Please call me diana.", speaker="g2", t0=1.1, t1=3.0),
    ]
    assert suggest_speaker_names(lowercase_name) == {}

    uppercase_lead = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("Please CALL ME Diana.", speaker="g2", t0=1.1, t1=3.0),
    ]
    assert suggest_speaker_names(uppercase_lead)["g2"]["name"] == "Diana"


def test_compile_wraps_custom_name_capture_case_strictly() -> None:
    (pattern,) = _compile(
        [r"call me (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+){0,2})"], strict_name=True
    )

    match = pattern.search("CALL ME Diana")
    assert match is not None
    assert match.group("name") == "Diana"

    assert pattern.search("call me diana") is None


# --- AC: the default self-intro grammar itself adopts the strict capture.


def test_default_self_intro_patterns_use_case_sensitive_capture() -> None:
    for pattern in DEFAULT_SELF_ID_PATTERNS:
        assert "(?-i:" in pattern, f"pattern lacks case-sensitive name capture: {pattern}"
