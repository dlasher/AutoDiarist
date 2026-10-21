from __future__ import annotations

import logging

from pipeline.chunking import Segment
from pipeline.self_identification import suggest_speaker_names


def segment(
    text: str,
    *,
    speaker: str,
    t0: float,
    t1: float,
) -> Segment:
    return Segment(text=text, t0=t0, t1=t1, speaker=speaker)


def test_self_identification_attaches_name_to_active_speaker_at_turn() -> None:
    segments = [
        segment("Welcome everyone.", speaker="g1", t0=0.0, t1=1.0),
        segment("Hi, my name is Jane Wu.", speaker="g2", t0=1.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Jane Wu"
    assert suggestions["g2"]["confidence"] == 0.7
    assert suggestions["g2"]["utterance_t0"] == 1.1
    assert suggestions["g2"]["evidence_snippet"] == "Hi, my name is Jane Wu."
    assert suggestions["g2"]["sources"] == ["self_intro"]
    assert suggestions["g2"]["action"] == "resolved"
    assert suggestions["g2"]["conflicts"] == []
    assert [(item.text, item.t0, item.t1, item.speaker) for item in segments] == [
        ("Welcome everyone.", 0.0, 1.0, "g1"),
        ("Hi, my name is Jane Wu.", 1.1, 3.0, "g2"),
    ]


def test_floor_handoff_attaches_name_to_next_different_speaker() -> None:
    segments = [
        segment("Now I'll turn the floor over to Bob.", speaker="g1", t0=0.0, t1=2.0),
        segment("Thanks, everyone.", speaker="g2", t0=2.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Bob"
    assert suggestions["g2"]["confidence"] == 0.7  # handoff 0.4 + tight window 0.3
    assert suggestions["g2"]["utterance_t0"] == 0.0
    assert suggestions["g2"]["evidence_snippet"] == "Now I'll turn the floor over to Bob."
    assert suggestions["g2"]["sources"] == ["handoff"]
    assert suggestions["g2"]["action"] == "resolved"


def test_floor_handoff_to_same_next_speaker_is_dropped_and_logged(caplog) -> None:
    segments = [
        segment("Now I'll turn the floor over to Bob.", speaker="g1", t0=0.0, t1=2.0),
        segment("Let me add one more thing.", speaker="g1", t0=2.1, t1=3.0),
    ]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments)

    assert suggestions == {}
    assert "self-ID handoff ignored" in caplog.text


def test_handoff_outside_attribution_window_is_dropped(caplog) -> None:
    segments = [
        segment("I'll hand it over to Bob.", speaker="g1", t0=0.0, t1=2.0),
        segment("Still talking.", speaker="g1", t0=2.1, t1=3.0),
        segment("A different voice much later.", speaker="g2", t0=90.0, t1=92.0),
    ]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments)

    assert suggestions == {}
    assert "outside attribution window" in caplog.text


def test_loose_window_handoff_stays_advisory_not_resolved() -> None:
    segments = [
        segment("I'll hand it over to Bob.", speaker="g1", t0=0.0, t1=2.0),
        segment("Thanks, everyone.", speaker="g2", t0=25.0, t1=27.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["confidence"] == 0.5  # 0.4 + loose window 0.1
    assert suggestions["g2"]["action"] == "suggested"


def test_third_party_intro_attributes_next_speaker() -> None:
    segments = [
        segment(
            "I have with me Jane Wu, the program manager.",
            speaker="g1",
            t0=0.0,
            t1=2.0,
        ),
        segment("Thanks for having me.", speaker="g2", t0=2.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["name"] == "Jane Wu"
    assert suggestions["g2"]["sources"] == ["third_party_intro"]
    assert suggestions["g2"]["confidence"] == 0.8  # intro 0.5 + tight 0.3
    assert suggestions["g2"]["action"] == "resolved"


def test_self_identification_without_turn_context_is_ignored(caplog) -> None:
    segments = [
        segment("We will begin now.", speaker="g1", t0=0.0, t1=1.0),
        segment("My name is Jane Wu.", speaker="g1", t0=1.1, t1=3.0),
        segment("The agenda is short.", speaker="g1", t0=3.1, t1=4.0),
    ]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments)

    assert suggestions == {}
    assert "self-ID ignored without turn context" in caplog.text


def test_no_pattern_occurrences_returns_empty_map() -> None:
    segments = [
        segment("The deployment begins today.", speaker="g1", t0=0.0, t1=2.0),
        segment("I agree with the proposal.", speaker="g2", t0=2.1, t1=4.0),
    ]

    assert suggest_speaker_names(segments) == {}


def test_function_words_reject_prose_like_self_intro_candidates(caplog) -> None:
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is a short brief.", speaker="g2", t0=1.1, t1=2.0),
        # Issue #1: the strict grammar drops lowercase prose words before
        # the validator ever sees them, so keep a capitalized function word
        # in the capture to exercise the non-name-function-word rejection
        # (single-word floor: "New Manager" backs off to "New", still a
        # function word).
        segment("My name is New Manager.", speaker="g3", t0=2.1, t1=3.0),
    ]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments)

    assert suggestions == {}
    assert "non-name function words" in caplog.text


def test_all_documented_default_patterns_and_ascii_capitalization() -> None:
    segments = [
        segment("This is Ada Lovelace.", speaker="g1", t0=0.0, t1=1.0),
        segment("Xavier here.", speaker="g2", t0=1.1, t1=2.0),
        segment("I am Terry Fields.", speaker="g3", t0=2.1, t1=3.0),
        segment("The floor goes to Carol Smith.", speaker="g4", t0=3.1, t1=4.0),
        segment("Thank you.", speaker="g5", t0=4.1, t1=5.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g1"]["name"] == "Ada Lovelace"
    assert suggestions["g2"]["name"] == "Xavier"
    assert suggestions["g3"]["name"] == "Terry Fields"
    assert suggestions["g5"]["name"] == "Carol Smith"


def test_case_insensitive_lead_phrase_requires_capitalized_name_capture() -> None:
    # Issue #1: the lead phrase stays case-insensitive, but the name capture
    # is strict — lowercase prose like "this is jane wu." is no longer a
    # name suggestion, while a capitalized name keeps its original casing.
    segments = [
        segment("THIS IS Jane Wu.", speaker="g1", t0=0.0, t1=1.0),
        segment("Welcome.", speaker="g2", t0=1.1, t1=2.0),
    ]

    assert suggest_speaker_names(segments)["g1"]["name"] == "Jane Wu"

    lowercase_capture = [
        segment("this is jane wu.", speaker="g1", t0=0.0, t1=1.0),
        segment("Welcome.", speaker="g2", t0=1.1, t1=2.0),
    ]

    assert suggest_speaker_names(lowercase_capture) == {}


def test_patterns_may_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SELF_ID_PATTERNS", r"call me (?P<name>[A-Z][a-z]+)")
    segments = [
        segment("Please call me Diana.", speaker="g1", t0=0.0, t1=1.0),
        segment("Nice to meet you.", speaker="g2", t0=1.1, t1=2.0),
    ]

    assert suggest_speaker_names(segments)["g1"]["name"] == "Diana"


def test_blocklist_rejects_known_false_positives_by_default() -> None:
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is Marine Mammal.", speaker="g2", t0=1.1, t1=2.0),
    ]

    assert suggest_speaker_names(segments) == {}


def test_blocklist_is_extensible_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SELF_ID_NAME_BLOCKLIST", '["Foo Bar"]')
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is Foo Bar.", speaker="g2", t0=1.1, t1=2.0),
        segment("This is Jane Wu.", speaker="g3", t0=2.1, t1=3.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert "g2" not in suggestions
    assert suggestions["g3"]["name"] == "Jane Wu"


def test_blocklist_is_extensible_from_vocab_file(monkeypatch, tmp_path) -> None:
    vocab = tmp_path / "blocklist.txt"
    vocab.write_text("# project phrases\nGhost Panel\n", encoding="utf-8")
    monkeypatch.setenv("SELF_ID_BLOCKLIST_PATH", str(vocab))
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is Ghost Panel.", speaker="g2", t0=1.1, t1=2.0),
    ]

    assert suggest_speaker_names(segments) == {}


def test_weak_single_word_here_form_requires_minimal_length() -> None:
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("Bo here.", speaker="g2", t0=1.1, t1=2.0),
        segment("Thanks.", speaker="g3", t0=2.1, t1=3.0),
        segment("Xavier here.", speaker="g4", t0=3.1, t1=4.0),
        segment("Bye.", speaker="g5", t0=4.1, t1=5.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert "g2" not in suggestions
    assert suggestions["g4"]["name"] == "Xavier"


def test_stacked_evidence_raises_confidence_and_lists_sources() -> None:
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is Jane Wu.", speaker="g2", t0=1.1, t1=2.0),
        segment("I'll turn the floor over to Jane Wu.", speaker="g3", t0=2.1, t1=3.0),
        segment("Thanks.", speaker="g2", t0=3.1, t1=4.0),
    ]

    suggestions = suggest_speaker_names(segments)

    assert suggestions["g2"]["sources"] == ["handoff", "self_intro"]
    assert suggestions["g2"]["confidence"] == 1.4 - 0.45  # stacked, capped at 0.95
    assert len(suggestions["g2"]["evidence"]) == 2


def test_conflicting_names_are_penalized_and_recorded() -> None:
    segments = [
        segment("Welcome.", speaker="g1", t0=0.0, t1=1.0),
        segment("This is Jane Wu.", speaker="g2", t0=1.1, t1=2.0),
    ]
    extra = [
        {
            "source": "llm_review",
            "label": "g2",
            "claimed_name": "Mark Lee",
            "weight": 0.5,
            "text": "model suggests Mark Lee",
            "timestamp": 1.2,
        }
    ]

    suggestions = suggest_speaker_names(segments, extra_evidence=extra)

    assert suggestions["g2"]["name"] == "Jane Wu"
    assert suggestions["g2"]["conflicts"] == ["Mark Lee"]
    assert suggestions["g2"]["confidence"] == 0.5  # 0.7 - 0.2 conflict penalty
    assert suggestions["g2"]["action"] == "suggested"


def test_extra_evidence_for_unknown_label_is_ignored(caplog) -> None:
    segments = [segment("Hello.", speaker="g1", t0=0.0, t1=1.0)]
    extra = [{"source": "llm_review", "label": "g9", "claimed_name": "Nobody", "weight": 0.5}]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments, extra_evidence=extra)

    assert suggestions == {}
    assert "extra evidence ignored" in caplog.text


def test_cross_label_collision_demotes_lower_confidence_label(caplog) -> None:
    segments = [
        segment("Welcome.", speaker="g0", t0=0.0, t1=1.0),
        segment("This is Jane Wu.", speaker="g1", t0=1.1, t1=2.0),
        segment("My name is Jane Wu.", speaker="g2", t0=2.1, t1=3.0),
        segment("Closing remarks now.", speaker="g3", t0=3.1, t1=4.0),
    ]

    with caplog.at_level(logging.INFO):
        suggestions = suggest_speaker_names(segments)

    assert "g1" in suggestions
    assert "g2" not in suggestions
    assert "cross-label collision" in caplog.text


def test_cross_label_collision_keeps_confidence_ranked_winner() -> None:
    segments = [
        segment("This is Jane Wu.", speaker="g1", t0=0.0, t1=1.0),
        segment("Welcome.", speaker="g2", t0=1.1, t1=2.0),
        segment(
            "I'll hand it over to Jane Wu.",
            speaker="g2",
            t0=2.1,
            t1=3.0,
        ),
        segment("Thanks, all.", speaker="g3", t0=3.1, t1=4.0),
    ]

    suggestions = suggest_speaker_names(segments)

    # g1: self-intro (0.7) and g3: handoff-tight (0.7) collide in the
    # "jane" first-name bucket; the tie is broken by label order (g1 wins).
    assert suggestions["g1"]["name"] == "Jane Wu"
    assert "g3" not in suggestions
