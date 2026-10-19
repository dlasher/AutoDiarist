"""Render reconciled diarized transcript segments as the v1 markdown format."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SILENCE_SPLIT_SECONDS = float(os.environ.get("SILENCE_SPLIT_SECONDS", "1.5"))

# Confidence needed for a suggestion to promote a name into the transcript
# body; matches pipeline.self_identification.THRESHOLD_RESOLVE by default.
RESOLVE_CONFIDENCE_THRESHOLD = float(os.environ.get("SELF_ID_RESOLVE_THRESHOLD", "0.7"))


def format_duration(duration_seconds: float) -> str:
    """Format a duration as mm:ss below an hour and h:mm:ss otherwise."""
    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def render_markdown(segments: Sequence[Any], meta: Mapping[str, Any]) -> str:
    """Render reconciled segments into the specified human and machine views.

    ``segments`` may contain the pipeline ``Segment`` dataclass or mapping-like
    records. Metadata is deliberately a mapping so callers can enrich it without
    coupling the pure renderer to API or job-storage types.
    """
    normalized = [_segment_dict(segment) for segment in segments]
    if not normalized:
        raise ValueError("at least one segment is required")
    _validate_silence_split()
    speaker_numbers = _speaker_numbers(normalized)
    suggestions = meta.get("suggestions", {})
    if not isinstance(suggestions, Mapping):
        raise TypeError("meta['suggestions'] must be a mapping when provided")

    filename = str(meta.get("filename") or meta.get("source") or "transcript")
    source = str(meta.get("source") or filename)
    title = Path(filename).stem
    processed_at = meta.get("processed_at") or datetime.now().astimezone().isoformat()
    resolved = _resolved_names(speaker_numbers, suggestions)
    speaker_value = _speakers_value(speaker_numbers, suggestions, resolved)

    lines = [
        f"# {title}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source | {source} |",
        f"| Duration | {format_duration(float(meta.get('duration_seconds', 0)))} |",
        f"| Language | {meta.get('language', 'en-US')} |",
        f"| Processed | {processed_at} |",
        f"| Speakers | {speaker_value} |",
        f"| Pipeline | {_pipeline_value(meta)} |",
        "",
    ]
    lines.extend(_mapping_table_lines(speaker_numbers, resolved))
    lines.extend(["## Transcript", ""])
    lines.extend(_transcript_lines(normalized, speaker_numbers, resolved))
    if _speaker_identification_section_enabled():
        lines.extend(_speaker_identification_lines(speaker_numbers, suggestions, resolved))
    lines.extend(
        [
            "## Appendix — Raw Diarizer Segments",
            "",
            "```json",
            json.dumps(normalized, indent=2, ensure_ascii=False),
            "```",
            "",
            "### Suggestion Provenance",
            "",
            "```json",
            json.dumps(suggestions, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _pipeline_value(meta: Mapping[str, Any]) -> str:
    """Pipeline stamp showing the actual backend identity (card t_3275e056).

    Precedence: ``MOCK_PIPELINE=1`` renders the deterministic ``(mock)``
    stamp (the probe never runs in mock mode); else an explicit
    ``meta['backend_label']`` supplied by the caller; else the CrispASR
    probe cache via ``pipeline.transcribe.backend_identity``; else
    ``@unknown``. The pre-pivot ``nemo_speech_sha``/``nemo_sha`` keys are
    deliberately ignored — the NeMo label no longer describes the backend.
    """
    if os.environ.get("MOCK_PIPELINE") == "1":
        return "autodiarist v1 (mock)"
    label = meta.get("backend_label")
    if label:
        return f"autodiarist v1 ({label})"
    try:
        from pipeline.transcribe import backend_identity
        identity = backend_identity()
    except ImportError:  # pragma: no cover - transcribe is stdlib-only
        identity = "CrispASR nemotron Q4_K @unknown"
    return f"autodiarist v1 ({identity})"


def _segment_dict(segment: Any) -> dict[str, Any]:
    if is_dataclass(segment) and not isinstance(segment, type):
        source = asdict(segment)
    elif isinstance(segment, Mapping):
        source = dict(segment)
    else:
        raise TypeError("segments must be dataclasses or mappings")

    required = ("speaker", "t0", "t1", "text")
    missing = [field for field in required if field not in source]
    if missing:
        raise ValueError(f"segment missing required field(s): {', '.join(missing)}")

    result = {field: source[field] for field in required}
    for field in ("chunk_id", "local_idx", "match_confidence"):
        if field in source and source[field] is not None:
            result[field] = source[field]
    return result


def _speaker_numbers(segments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for segment in segments:
        label = str(segment["speaker"])
        if label not in labels:
            labels[label] = len(labels) + 1
    return labels


def _speaker_identification_section_enabled() -> bool:
    return os.environ.get("SELF_ID_SECTION", "1") == "1"


def _resolved_names(
    speaker_numbers: Mapping[str, int], suggestions: Mapping[str, Any]
) -> dict[str, str]:
    """Labels whose suggestion crosses the resolve threshold.

    ``resolved``/``action`` on a mapping entry is authoritative when present
    (the suggestion pass owns the weighted thresholds); otherwise confidence
    is compared against ``RESOLVE_CONFIDENCE_THRESHOLD``. Plain-string
    suggestions carry no provenance and are treated as resolved.
    """
    resolved: dict[str, str] = {}
    for label in speaker_numbers:
        value = suggestions.get(label)
        name = _suggestion_name(value)
        if name is None:
            continue
        if isinstance(value, Mapping):
            explicit = value.get("resolved")
            if isinstance(explicit, bool):
                if explicit:
                    resolved[label] = name
                continue
            action = value.get("action")
            if isinstance(action, str) and action:
                if action == "resolved":
                    resolved[label] = name
                continue
            confidence = value.get("confidence")
            if isinstance(confidence, (int, float)) and confidence < RESOLVE_CONFIDENCE_THRESHOLD:
                continue
        resolved[label] = name
    return resolved


def _mapping_table_lines(
    speaker_numbers: Mapping[str, int], resolved: Mapping[str, str]
) -> list[str]:
    """Speaker→name table at the top of the document (grep anchor for labels)."""
    if not resolved:
        return []
    ordered = sorted(speaker_numbers.items(), key=lambda item: item[1])
    lines = ["## Speaker mapping", "", "| Speaker | Name |", "|---|---|"]
    lines.extend(
        [
            f"| Speaker {number} | {resolved[label]} |"
            for label, number in ordered
            if label in resolved
        ]
    )
    lines.append("")
    return lines


def _speaker_identification_lines(
    speaker_numbers: Mapping[str, int],
    suggestions: Mapping[str, Any],
    resolved: Mapping[str, str],
) -> list[str]:
    """Human-readable evidence trail for every naming decision."""
    total = len(speaker_numbers)
    resolved_count = sum(1 for label in speaker_numbers if label in resolved)
    suggested_count = sum(
        1
        for label in speaker_numbers
        if label not in resolved and _suggestion_name(suggestions.get(label))
    )
    unidentified_count = total - resolved_count - suggested_count
    lines = [
        "## Speaker identification",
        "",
        (
            f"Resolved {resolved_count} of {total} speakers by name "
            f"({suggested_count} with lower confidence, "
            f"{unidentified_count} unidentified)."
        ),
        "",
    ]
    ordered = sorted(speaker_numbers.items(), key=lambda item: item[1])
    for label, number in ordered:
        value = suggestions.get(label)
        name = _suggestion_name(value)
        if name is None:
            lines.append(f"- **Speaker {number}** (`{label}`) UNIDENTIFIED (no evidence).")
            continue
        entry = value if isinstance(value, Mapping) else {}
        confidence = entry.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
        sources = entry.get("sources")
        sources_text = ", ".join(str(item) for item in sources) if sources else "user-supplied"
        if label in resolved:
            lines.append(
                f"- **{name}** — Speaker {number} (`{label}`), confidence={confidence_text}; "
                f"sources: {sources_text}."
            )
        else:
            lines.append(
                f"- `{label}` likely **{name}** (confidence={confidence_text}; "
                f"sources: {sources_text}) — kept as `Speaker {number}` in the transcript."
            )
        for evidence in entry.get("evidence") or []:
            if not isinstance(evidence, Mapping):
                continue
            snippet = str(evidence.get("snippet") or "").strip()
            if not snippet:
                continue
            t0 = evidence.get("t0")
            stamp = _format_timestamp(float(t0)) if isinstance(t0, (int, float)) else "--:--:--"
            lines.append(
                f"  - `{stamp}` {evidence.get('source', 'evidence')} "
                f"(w={evidence.get('weight', '?')}) “{snippet[:100]}”"
            )
        conflicts = entry.get("conflicts") or []
        if conflicts:
            lines.append(f"  - competing names: {', '.join(str(c) for c in conflicts)}")
    if resolved_count < total:
        lines.extend(
            [
                "",
                "### Speaker identification request",
                "",
                (
                    "Names are advisory evidence, not source truth: transcript labels in the "
                    "appendix remain the authoritative speaker keys. Set `SELF_ID_SECTION=0` "
                    "to omit this section."
                ),
            ]
        )
    lines.append("")
    return lines


def _speakers_value(
    numbers: Mapping[str, int],
    suggestions: Mapping[str, Any],
    resolved: Mapping[str, str],
) -> str:
    rendered = []
    for label, number in numbers.items():
        suggestion = _suggestion_name(suggestions.get(label))
        if label in resolved:
            label_text = f"Speaker {number} ({suggestion})"
        elif suggestion:
            label_text = f"Speaker {number} (suggested: {suggestion})"
        else:
            label_text = f"Speaker {number}"
        rendered.append(label_text)
    return f"{len(numbers)} ({'; '.join(rendered)})" if rendered else "0"


def _transcript_lines(
    segments: Sequence[Mapping[str, Any]],
    speaker_numbers: Mapping[str, int],
    resolved: Mapping[str, str],
) -> list[str]:
    paragraphs: list[str] = []
    run_speaker: str | None = None
    prior_t1: float | None = None
    first_paragraph_in_run = False
    text_parts: list[str] = []
    paragraph_start = 0.0

    def flush() -> None:
        if not text_parts or run_speaker is None:
            return
        number = speaker_numbers[run_speaker]
        timestamp = _format_timestamp(paragraph_start)
        text = " ".join(text_parts)
        if first_paragraph_in_run:
            resolved_name = resolved.get(run_speaker)
            label = resolved_name if resolved_name else f"Speaker {number}"
            paragraphs.append(f"**{label}** [{timestamp}] — {text}")
        else:
            paragraphs.append(f"[{timestamp}] — {text}")

    for segment in segments:
        speaker = str(segment["speaker"])
        t0 = float(segment["t0"])
        t1 = float(segment["t1"])
        is_new_run = speaker != run_speaker
        is_silence_split = not is_new_run and prior_t1 is not None and t0 - prior_t1 > SILENCE_SPLIT_SECONDS
        if is_new_run or is_silence_split:
            flush()
            text_parts = []
            paragraph_start = t0
            if is_new_run:
                run_speaker = speaker
                first_paragraph_in_run = True
            else:
                first_paragraph_in_run = False
        text_parts.append(str(segment["text"]).strip())
        prior_t1 = t1
    flush()

    return [line for paragraph in paragraphs for line in (paragraph, "")]


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _suggestion_name(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        candidate = value[0]
        return candidate.strip() if isinstance(candidate, str) and candidate.strip() else None
    if isinstance(value, Mapping):
        candidate = value.get("name")
        return candidate.strip() if isinstance(candidate, str) and candidate.strip() else None
    return None


def _validate_silence_split() -> None:
    if SILENCE_SPLIT_SECONDS < 0:
        raise ValueError("SILENCE_SPLIT_SECONDS must be non-negative")
