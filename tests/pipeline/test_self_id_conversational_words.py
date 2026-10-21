"""Regression tests for the conversational-word hole (card t_ac7889e4).

Live defect (found on the VM in t_90e1752b, strict tier, PRE-EXISTING):
CrispASR case-restores prose mid-sentence, so a decoy like ``I'm sorry``
arrives in the diarizer output as ``I'm Sorry``. The strict ``\\bi'm
{NAME_STRICT}`` line must capture capitalized words (that is how ``I'm
Shelley Festing`` works), "Sorry" is 5 chars so it clears the <4-char
weak-single-word guard, and it is not in the phrase blocklist. The result:
Speaker 2 resolves as the name "Sorry" at conf 0.70.

Fix: a conservative CONVERSATIONAL-WORD set applied to every candidate at the
single ``_candidate_name`` choke point (all tiers: strict self-intro, option
(b) lowercase self-intro, handoff, third-party intro). A candidate is
rejected when ANY of its words is conversational -- that covers the single
word ("Sorry"), the conversational surname ("Smith Sorry", AC5) and the
conversational given word ("Sorry Smith", the card's example).

The set must stay conservative: real surnames (May, Young, Poe, Hart, Cook,
Good, Small, Strange, Rich, Wise, Brown, White, Green, Black) must NOT be
in it, so "I'm Sarah Cook" and friends keep resolving (AC4).
"""

from __future__ import annotations

from pipeline.chunking import Segment
from pipeline.self_identification import suggest_speaker_names


def segment(text: str, *, speaker: str, t0: float, t1: float) -> Segment:
    return Segment(text=text, t0=t0, t1=t1, speaker=speaker)


def _suggest(text: str, *, intro: str = "Welcome everyone.") -> dict:
    return suggest_speaker_names(
        [
            segment(intro, speaker="g1", t0=0.0, t1=1.0),
            segment(text, speaker="g2", t0=1.1, t1=3.0),
        ]
    )


# --- AC1: "I'm sorry" -> NO suggestion (live-verified on the VM clip) ------
# The live engine renders the decoy case-restored; the raw clip string from
# t_90e1752b (verbatim, including the mangled tail) is the regression case.
# The lowercase spelling must of course stay rejected too (pre-existing).


def test_im_sorry_capitalized_live_form_resolves_no_name() -> None:
    assert _suggest("I'm Sorry.") == {}


def test_live_vm_clip_decoy_string_verbatim_resolves_no_name() -> None:
    live_snippet = (
        "I'm Sorry, I did not catch that first line. co counting on from "
        "three, two, one, and we"
    )
    assert _suggest(live_snippet) == {}


def test_im_sorry_lowercase_stays_rejected() -> None:
    assert _suggest("I'm sorry.") == {}


# --- AC2: "I'm going to explain that" / "this is going to be quick" --------
# Both tiers, both spellings. On the strict tier the greedy capture trims the
# function words "to" (and "that") down to "Going", which then clears the
# 4-char guard -- exactly the hole this card closes.


def test_im_going_to_explain_resolves_no_name() -> None:
    assert _suggest("I'm Going To Explain That.") == {}
    assert _suggest("I'm going to explain that.") == {}


def test_this_is_going_to_be_quick_resolves_no_name() -> None:
    # "this is" is an option (b) phrase: the lowercase tier must apply the
    # conversational set too, not just the strict tier.
    assert _suggest("This Is Going To Be Quick.") == {}
    assert _suggest("this is going to be quick.") == {}


# --- Conversational word in the surname position (AC5) and as one word of a
# --- two-word capture (the card's "Sorry Smith" example).


def test_two_word_capture_with_conversational_surname_is_rejected() -> None:
    assert _suggest("I'm Smith Sorry.") == {}


def test_two_word_capture_with_conversational_given_word_is_rejected() -> None:
    assert _suggest("I'm Sorry Smith.") == {}


def test_lowercase_tier_conversational_surname_is_rejected() -> None:
    # Two-word all-lowercase captures pass the given+surname shape guard by
    # design; the conversational filter is what rejects prose surnames.
    assert _suggest("my name is jane sorry.") == {}


# --- AC3/AC4: real names must keep resolving; the set must not swallow
# --- plausible surnames.


def test_capitalized_two_word_names_still_resolve() -> None:
    for text, expected in (
        ("I'm Shelley Festing.", "Shelley Festing"),
        ("I'm Jane Smith.", "Jane Smith"),
        ("I'm Sarah Cook.", "Sarah Cook"),
        ("I'm Mary Young.", "Mary Young"),
        ("I'm Fred Rich.", "Fred Rich"),
    ):
        suggestions = _suggest(text)
        assert "g2" in suggestions, text
        assert suggestions["g2"]["name"] == expected


def test_lowercase_tier_names_still_resolve() -> None:
    for text, expected in (
        ("my name is sarah cook.", "sarah cook"),
        ("my name is mary young.", "mary young"),
        ("my name is fred rich.", "fred rich"),
    ):
        suggestions = _suggest(text)
        assert "g2" in suggestions, text
        assert suggestions["g2"]["name"] == expected


def test_surname_like_function_ish_words_are_not_blocked_midname() -> None:
    # "May"/"Young"/"Cook" must remain capturable as the leading given word
    # as well -- the conservative set excludes them entirely.
    for text, expected in (
        ("I'm May West.", "May West"),
        ("I'm Young Lee.", "Young Lee"),
        ("I'm Cook Daniels.", "Cook Daniels"),
    ):
        suggestions = _suggest(text)
        assert suggestions.get("g2", {}).get("name") == expected, text
