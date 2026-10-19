from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.chunking import Segment
from pipeline.renderer import format_duration, render_markdown

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_segments() -> list[Segment]:
    return [
        Segment(
            text="Welcome everyone.",
            t0=3.0,
            t1=4.0,
            speaker="g1",
            chunk_id=0,
            local_idx="0",
            match_confidence=1.0,
        ),
        Segment(
            text="Let's start.",
            t0=4.2,
            t1=5.0,
            speaker="g1",
            chunk_id=0,
            local_idx="0",
            match_confidence=1.0,
        ),
        Segment(
            text="Thanks, Jane.",
            t0=5.4,
            t1=6.4,
            speaker="g2",
            chunk_id=0,
            local_idx="1",
            match_confidence=1.0,
        ),
    ]


def fixture_meta() -> dict[str, object]:
    return {
        "filename": "team-sync.mp3",
        "duration_seconds": 47 * 60 + 12,
        "language": "en-US",
        "processed_at": "2026-10-16T11:30:00-07:00",
        "backend_label": "CrispASR nemotron Q4_K @v0.8.43",
        "suggestions": {"g1": "Jane Wu"},
    }


def test_render_markdown_matches_golden_fixture() -> None:
    actual = render_markdown(fixture_segments(), fixture_meta())

    assert actual == (FIXTURES / "team-sync.md").read_text()


def test_same_speaker_segments_form_one_labeled_paragraph() -> None:
    rendered = render_markdown(fixture_segments()[:2], fixture_meta())
    transcript = rendered.split("## Appendix", maxsplit=1)[0]

    # Resolved names replace the "Speaker N" label in the body per the
    # current render contract; "Speaker N" stays greppable via the mapping.
    assert transcript.count("**Jane Wu** [00:00:03]") == 1
    assert "**Jane Wu** [00:00:03] — Welcome everyone. Let's start." in transcript
    assert "(suggested:" not in transcript
    assert "| Speaker 1 | Jane Wu |" in rendered


def test_renderer_embeds_structured_self_id_suggestions_in_appendix() -> None:
    rendered = render_markdown(
        fixture_segments()[:2],
        {
            **fixture_meta(),
            "suggestions": {
                "g1": {
                    "name": "Jane Wu",
                    "confidence": 0.9,
                    "utterance_t0": 3.0,
                    "evidence_snippet": "Hi, my name is Jane Wu.",
                }
            },
        },
    )

    appendix = rendered.rsplit("```json\n", maxsplit=1)[1].split("\n```", maxsplit=1)[0]
    provenance = json.loads(appendix)
    assert provenance["g1"]["name"] == "Jane Wu"
    assert provenance["g1"]["evidence_snippet"] == "Hi, my name is Jane Wu."


def test_two_second_silence_splits_same_speaker_run_without_repeating_label() -> None:
    segments = [
        Segment(text="First thought.", t0=1.0, t1=2.0, speaker="g1"),
        Segment(text="Second thought.", t0=4.0, t1=5.0, speaker="g1"),
    ]
    rendered = render_markdown(segments, fixture_meta())
    transcript = rendered.split("## Appendix", maxsplit=1)[0]

    assert "**Jane Wu** [00:00:01] — First thought." in transcript
    assert "[00:00:04] — Second thought." in transcript
    assert transcript.count("**Jane Wu** [") == 1


def test_short_silence_does_not_split_same_speaker_run() -> None:
    segments = [
        Segment(text="First thought.", t0=1.0, t1=2.0, speaker="g1"),
        Segment(text="Second thought.", t0=3.2, t1=4.0, speaker="g1"),
    ]

    rendered = render_markdown(segments, fixture_meta())

    assert "First thought. Second thought." in rendered
    assert "[00:00:03] — Second thought." not in rendered


def test_duration_formatting_uses_minutes_or_hours_as_required() -> None:
    assert format_duration(47 * 60 + 12) == "47:12"
    assert format_duration(60 * 60 + 2 * 60 + 45) == "1:02:45"


def test_empty_segment_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        render_markdown([], fixture_meta())


def test_missing_suggestions_uses_plain_speaker_labels() -> None:
    meta = fixture_meta()
    meta.pop("suggestions")

    rendered = render_markdown(fixture_segments()[:1], meta)

    assert "**Speaker 1** [00:00:03]" in rendered
    assert "suggested:" not in rendered


def test_appendix_is_parseable_json_with_provenance_fields() -> None:
    rendered = render_markdown(fixture_segments(), fixture_meta())
    appendix = rendered.split("```json\n", maxsplit=1)[1].split("\n```", maxsplit=1)[0]

    segments = json.loads(appendix)

    assert segments[0] == {
        "speaker": "g1",
        "t0": 3.0,
        "t1": 4.0,
        "text": "Welcome everyone.",
        "chunk_id": 0,
        "local_idx": "0",
        "match_confidence": 1.0,
    }


def test_below_resolve_threshold_suggestion_keeps_plain_body_label() -> None:
    segments = fixture_segments()[:1]
    meta = fixture_meta()
    meta["suggestions"] = {
        "g1": {"name": "Carol Wu", "confidence": 0.5, "action": "suggested", "sources": ["handoff"]}
    }

    rendered = render_markdown(segments, meta)
    transcript = rendered.split("## Speaker identification", maxsplit=1)[0]

    assert "**Speaker 1** [00:00:03]" in transcript
    assert "## Speaker mapping" not in rendered
    assert "Speaker 1 (suggested: Carol Wu)" in rendered
    assert "likely **Carol Wu**" in rendered


def test_confidence_of_zero_dict_is_not_resolved_but_stays_advisory() -> None:
    segments = fixture_segments()[:1]
    meta = fixture_meta()
    meta["suggestions"] = {"g1": {"name": "Carol Wu", "confidence": 0.0}}

    rendered = render_markdown(segments, meta)

    assert "**Speaker 1** [00:00:03]" in rendered
    assert "Speaker 1 (suggested: Carol Wu)" in rendered
    assert "## Speaker mapping" not in rendered


def test_dict_without_threshold_metadata_falls_back_to_confidence() -> None:
    segments = fixture_segments()[:1]
    meta = fixture_meta()
    meta["suggestions"] = {"g1": {"name": "Jane Wu", "confidence": 0.9}}

    rendered = render_markdown(segments, meta)

    assert "**Jane Wu** [00:00:03]" in rendered
    assert "| Speaker 1 | Jane Wu |" in rendered


def test_speaker_identification_section_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SELF_ID_SECTION", "0")

    rendered = render_markdown(fixture_segments(), fixture_meta())

    assert "## Speaker identification" not in rendered
    assert "## Speaker mapping" in rendered


def test_speaker_identification_section_lists_weighted_evidence() -> None:
    segments = fixture_segments()
    meta = fixture_meta()
    meta["suggestions"] = {
        "g1": {
            "name": "Jane Wu",
            "confidence": 0.95,
            "action": "resolved",
            "sources": ["handoff", "self_intro"],
            "utterance_t0": 3.0,
            "evidence_snippet": "My name is Jane Wu.",
            "conflicts": ["Janet Wu"],
            "evidence": [
                {
                    "source": "self_intro",
                    "label": "g1",
                    "claimed_name": "Jane Wu",
                    "weight": 0.7,
                    "snippet": "My name is Jane Wu.",
                    "t0": 3.0,
                    "extra": {},
                }
            ],
        },
        "g2": {
            "name": "Bob Lee",
            "confidence": 0.5,
            "action": "suggested",
            "sources": ["handoff"],
            "utterance_t0": 5.4,
            "evidence_snippet": "I'll turn the floor over to Bob Lee.",
            "conflicts": [],
            "evidence": [],
        },
    }

    rendered = render_markdown(segments, meta)

    assert "**Jane Wu** [00:00:03]" in rendered
    assert "| Speaker 1 | Jane Wu |" in rendered
    assert "sources: handoff, self_intro" in rendered
    assert "competing names: Janet Wu" in rendered
    assert "Resolved 1 of 2 speakers by name (1 with lower confidence" in rendered
    assert "kept as `Speaker 2` in the transcript" in rendered


def test_unidentified_speakers_are_listed_and_request_emitted() -> None:
    segments = [
        Segment(text="Hello.", t0=0.0, t1=1.0, speaker="g1"),
        Segment(text="Hi.", t0=1.1, t1=2.0, speaker="g2"),
    ]

    rendered = render_markdown(segments, fixture_meta())

    assert "**Speaker 2** (`g2`) UNIDENTIFIED" in rendered
    assert "### Speaker identification request" in rendered


# ---------------------------------------------------------------------------
# Pipeline stamp (card t_3275e056): backend identity replaces the legacy
# "NeMo Speech @<sha>" label; mock mode stays deterministic.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_mock_env(monkeypatch):
    # test_crispasr's mock test writes MOCK_PIPELINE into os.environ
    # directly; pin the env so stamp tests are order-independent.
    monkeypatch.delenv("MOCK_PIPELINE", raising=False)


def _meta_without_label() -> dict[str, object]:
    meta = dict(fixture_meta())
    meta.pop("backend_label")
    return meta


def test_pipeline_stamp_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_PIPELINE", "1")

    rendered = render_markdown(fixture_segments()[:1], fixture_meta())

    assert "| Pipeline | autodiarist v1 (mock) |" in rendered
    # mock wins even when a caller supplied a backend label
    assert "CrispASR" not in rendered


def test_pipeline_stamp_uses_cached_probe_identity(monkeypatch):
    from pipeline import transcribe as tr

    monkeypatch.setattr(
        tr, "_BACKEND_PROBE_CACHE",
        {"/opt/crispasr/crispasr": {
            "path": "/opt/crispasr/crispasr",
            "backends": ["cuda"],
            "version": "v0.8.43",
        }},
    )

    rendered = render_markdown(fixture_segments()[:1], _meta_without_label())

    assert "| Pipeline | autodiarist v1 (CrispASR nemotron Q4_K @v0.8.43) |" in rendered


def test_pipeline_stamp_unprobed_falls_back_to_unknown(monkeypatch):
    from pipeline import transcribe as tr

    monkeypatch.setattr(tr, "_BACKEND_PROBE_CACHE", {})

    rendered = render_markdown(fixture_segments()[:1], _meta_without_label())

    assert "| Pipeline | autodiarist v1 (CrispASR nemotron Q4_K @unknown) |" in rendered
    assert "NeMo Speech" not in rendered
    # legacy meta keys must no longer leak into the stamp
    legacy = render_markdown(
        fixture_segments()[:1], {**_meta_without_label(), "nemo_speech_sha": "deadbeef"},
    )
    assert "deadbeef" not in legacy
