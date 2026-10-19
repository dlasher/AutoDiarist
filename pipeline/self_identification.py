"""Weighted-evidence speaker-name suggestions derived from diarized turns.

Ported from the contributed ``speaker_naming.py`` analysis script, adapted to
this pipeline's segment records and advisory contract:

- Evidence sources carry per-source weights and are aggregated per
  (label, name) pair; stacking the same name adds weight, competing names
  trigger a conflict penalty.
- Handoff and third-party-intro patterns attribute the named person to the
  *next different speaker*, gated by tight/loose timing windows.
- A plausibility validator with a configurable false-positive blocklist
  rejects phrases that look like names but are not.
- Cross-label first-name collisions are deduplicated: only the
  highest-confidence label keeps the name.

This pass is deliberately advisory: it returns provenance-rich suggestions
but never mutates transcript segments, timing, or speaker labels. The
``action`` field ("resolved" / "suggested") is what the renderer consumes to
decide whether the transcript body shows the name directly.

v2 hooks (deliberately not implemented here):
- E3a/E3b slide and panelist-strip OCR (video scope; audio-only pipeline).
- E4 LLM review pass (inject via ``extra_evidence``; default off).
- E5 cross-reference inference (weakest-weight heuristics; revisit on QA need).
- Hard label remap of transcript segments (out of contract by decision).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# The name grammar is intentionally ASCII-only as required by the v1 heuristic.
_NAME = r"(?P<name>[A-Z][a-z]+(?:[ ][A-Z][a-z]+){0,2})"
_LOWER_NAME = r"(?P<name>[A-Za-z]+(?: [A-Za-z]+){0,2})"

SOURCE_SELF_INTRO = "self_intro"
SOURCE_HANDOFF = "handoff"
SOURCE_THIRD_PARTY_INTRO = "third_party_intro"

EVIDENCE_WEIGHTS: dict[str, float] = {
    SOURCE_SELF_INTRO: 0.7,
    SOURCE_HANDOFF: 0.4,
    SOURCE_THIRD_PARTY_INTRO: 0.5,
}

# Thresholds (ported from the contributed script's remap/annotate pair and
# renamed for our advisory vocabulary: "resolved" means the renderer may put
# the name in the transcript body; "suggested" is evidence-section only).
THRESHOLD_RESOLVE = float(os.environ.get("SELF_ID_RESOLVE_THRESHOLD", "0.7"))
THRESHOLD_ANNOTATE = float(os.environ.get("SELF_ID_ANNOTATE_THRESHOLD", "0.4"))
CONFLICT_PENALTY = 0.2
CONFLICT_MIN_WEIGHT = 0.3
CONFIDENCE_CAP = 0.95

# Next-speaker attribution windows for handoff/intro patterns, in seconds
# measured from the start of the saying segment to the next different
# speaker's start. Outside the loose window the attribution is dropped.
HANDOFF_TIGHT_WINDOW = 5.0
HANDOFF_TIGHT_BONUS = 0.3
HANDOFF_LOOSE_WINDOW = 30.0
HANDOFF_LOOSE_BONUS = 0.1

# Optional line-per-phrase blocklist file (SELF_ID_BLOCKLIST_PATH, default
# ./self_id_blocklist.txt), mirroring the spellcheck vocab convention: missing
# files warn and are ignored; '#' starts a comment. Also extendable via
# SELF_ID_NAME_BLOCKLIST (JSON array or comma-separated string).
DEFAULT_NAME_BLOCKLIST: frozenset[str] = frozenset(
    {
        "Q and A",
        "Federal Government",
        "United States",
        "North America",
        "New Mexico",
        "Prime Contract",
        "Contract Holder",
        "Recording Permission",
        "Section Header",
        "General Information",
        "Deputy Director",
        "Deputy Program",
        "Program Manager",
        "Convention Center",
        "Marine Mammal",
        "Blue River",
    }
)

# Optional line-per-phrase blocklist file, mirroring the spellcheck vocab
# convention (missing file warns and is ignored; '#' lines are comments).
DEFAULT_SELF_ID_BLOCKLIST_PATH = Path("self_id_blocklist.txt")

DEFAULT_SELF_ID_PATTERNS: tuple[str, ...] = (
    rf"\bmy[ ]+name[ ]+is[ ]+{_LOWER_NAME}\b",
    rf"\bi(?:'|’)[ ]?m[ ]+{_LOWER_NAME}\b",
    rf"\bi[ ]+am[ ]+{_LOWER_NAME}\b",
    rf"\bthis[ ]+is[ ]+{_LOWER_NAME}\b",
    rf"\b{_NAME}[ ]+here\b",
)
DEFAULT_HANDOFF_PATTERNS: tuple[str, ...] = (
    (
        rf"\b(?:i(?:'|’)ll|i[ ]+will)[ ]+(?:turn|hand|pass)[ ]+(?:it|this|the)[ ]+over"
        rf"(?:[ ]+(?:now|please|today|next))?[ ]+to[ ]+{_NAME}\b"
    ),
    rf"\b(?:i(?:'|’)ll|i[ ]+will)[ ]+turn[ ]+the[ ]+floor[ ]+over[ ]+to[ ]+{_NAME}\b",
    rf"\bthe[ ]+floor[ ]+goes[ ]+to[ ]+{_NAME}\b",
    rf"\blet[ ]+me[ ]+hand[ ]+it[ ]+over[ ]+to[ ]+{_NAME}\b",
    rf"\bhanding[ ]+(?:it|this)[ ]+over[ ]+to[ ]+{_NAME}\b",
)
DEFAULT_THIRD_PARTY_INTRO_PATTERNS: tuple[str, ...] = (
    # "I have (in) with me [in the room/today] X, the [role]"
    rf"\b(?:i[ ]+)?have[ ]+(?:in[ ]+)?with[ ]+me\b[^.]*?\b{_NAME}\b[ ,;]+(?:uh[ ]+|um[ ]+)?(?:the|our|who'?s)\b",
    # "X, the program manager / director / deputy" role-apposition
    rf"\b{_NAME}\b[ ,;]+(?:uh[ ]+|um[ ]+)?(?:the|our)\b[ ]+(?:[A-Za-z]+[ ]+){{0,3}}\b(?:manager|director|deputy|lead)\b",
    rf"\bjoining[ ]+me(?:[ ]+today)?[ ]+is[ ]+{_NAME}\b",
)


@dataclass
class Evidence:
    """One weighted observation tying a speaker label to a claimed name."""

    source: str
    label: str
    claimed_name: str
    weight: float
    snippet: str
    t0: float
    extra: dict = field(default_factory=dict)


def suggest_speaker_names(
    segments: Sequence[Any],
    *,
    extra_evidence: Iterable[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    """Return weighted advisory speaker-name suggestions per diar label.

    ``segments`` keeps the historical pipeline contract (dataclass or mapping
    records with ``speaker``/``t0``/``t1``/``text``). Each returned entry
    carries the legacy keys (``name``, ``confidence``, ``utterance_t0``,
    ``evidence_snippet``) plus provenance (``sources``, ``evidence``,
    ``conflicts``, ``action``). ``extra_evidence`` is the v2 seam for
    out-of-band sources such as an LLM review pass; entries are ``Evidence``
    mappings and are folded into the same weighted aggregation.

    Labels never reach ``resolved`` through ``extra_evidence`` alone unless
    the weighted total crosses ``SELF_ID_RESOLVE_THRESHOLD``; nothing here
    mutates the input segments.
    """
    normalized = [_segment_dict(segment) for segment in segments]
    blocklist = _name_blocklist()
    evidence = _collect_evidence(normalized, blocklist)
    evidence.extend(_coerce_extra(extra_evidence, normalized))
    return _aggregate(evidence, normalized)


def _collect_evidence(
    segments: Sequence[Mapping[str, Any]], blocklist: frozenset[str]
) -> list[Evidence]:
    evidence: list[Evidence] = []
    self_patterns, handoff_patterns, intro_patterns = _patterns_from_environment()
    for index, segment in enumerate(segments):
        text = segment["text"]
        self_hit = _first_match(self_patterns, text)
        if self_hit is not None:
            pattern, match = self_hit
            weak_here_form = pattern is self_patterns[-1] and not os.environ.get(
                "SELF_ID_PATTERNS"
            )
            name = _candidate_name(match, weak_single_word=weak_here_form, blocklist=blocklist)
            if name is not None:
                if _has_turn_boundary(segments, index):
                    evidence.append(
                        Evidence(
                            source=SOURCE_SELF_INTRO,
                            label=segment["speaker"],
                            claimed_name=name,
                            weight=EVIDENCE_WEIGHTS[SOURCE_SELF_INTRO],
                            snippet=text,
                            t0=segment["t0"],
                        )
                    )
                else:
                    LOGGER.info(
                        "self-ID ignored without turn context at t0=%s", segment["t0"]
                    )
            continue

        attribution = _first_attribution_match(handoff_patterns, text, SOURCE_HANDOFF)
        if attribution is None:
            attribution = _first_attribution_match(
                intro_patterns, text, SOURCE_THIRD_PARTY_INTRO
            )
        if attribution is None:
            continue
        source, match = attribution
        name = _candidate_name(match, weak_single_word=False, blocklist=blocklist)
        if name is None:
            continue
        timing = _next_speaker_timing(segments, index)
        if timing is None:
            continue
        label, bonus = timing
        evidence.append(
            Evidence(
                source=source,
                label=label,
                claimed_name=name,
                weight=EVIDENCE_WEIGHTS[source] + bonus,
                snippet=text,
                t0=segment["t0"],
                extra={"attributed_from": segment["speaker"]},
            )
        )
    return evidence


def _first_attribution_match(
    patterns: Sequence[re.Pattern[str]], text: str, source: str
) -> tuple[str, re.Match[str]] | None:
    hit = _first_match(patterns, text)
    if hit is None:
        return None
    return source, hit[1]


def _next_speaker_timing(
    segments: Sequence[Mapping[str, Any]], index: int
) -> tuple[str, float] | None:
    """Find the next different speaker after ``index`` and its timing bonus.

    Returns ``(label, bonus)`` within the tight/loose windows, or ``None``
    when no different speaker follows or the follower is beyond the loose
    window. Drop reasons are logged with the legacy attribution messages.
    """
    saying = segments[index]
    for follower in segments[index + 1 :]:
        if follower["speaker"] == saying["speaker"]:
            continue
        delay = follower["t0"] - saying["t0"]
        if delay <= HANDOFF_TIGHT_WINDOW:
            return follower["speaker"], HANDOFF_TIGHT_BONUS
        if delay <= HANDOFF_LOOSE_WINDOW:
            return follower["speaker"], HANDOFF_LOOSE_BONUS
        LOGGER.info(
            "self-ID handoff ignored: next speaker %s outside attribution window "
            "at t0=%s (delay=%ss)",
            follower["speaker"],
            saying["t0"],
            round(delay, 1),
        )
        return None
    LOGGER.info(
        "self-ID handoff ignored: next speaker did not change at t0=%s",
        saying["t0"],
    )
    return None


def _aggregate(
    evidence: Sequence[Evidence], segments: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_label: dict[str, dict[str, list[Evidence]]] = {}
    for item in evidence:
        by_label.setdefault(item.label, {}).setdefault(item.claimed_name, []).append(item)

    suggestions: dict[str, dict[str, Any]] = {}
    for label, by_name in by_label.items():
        totals = {name: sum(ev.weight for ev in evs) for name, evs in by_name.items()}
        ranked = sorted(
            by_name,
            key=lambda name: (
                -totals[name],
                min(ev.t0 for ev in by_name[name]),
                name,
            ),
        )
        best_name = ranked[0]
        confidence = min(CONFIDENCE_CAP, round(totals[best_name], 3))
        conflicts = sorted(
            name
            for name, weight in totals.items()
            if name != best_name and weight > CONFLICT_MIN_WEIGHT
        )
        if conflicts:
            confidence = round(max(0.0, confidence - CONFLICT_PENALTY), 3)
        if confidence < THRESHOLD_ANNOTATE:
            LOGGER.info(
                "self-ID below annotate threshold, dropped: label=%s name=%r conf=%s",
                label,
                best_name,
                confidence,
            )
            continue
        action = "resolved" if confidence >= THRESHOLD_RESOLVE else "suggested"
        primary = min(by_name[best_name], key=lambda ev: (ev.t0, ev.source))
        suggestions[label] = {
            "name": best_name,
            "confidence": confidence,
            "utterance_t0": primary.t0,
            "evidence_snippet": primary.snippet,
            "sources": sorted({ev.source for ev in by_name[best_name]}),
            "evidence": [asdict(ev) for ev in sorted(by_name[best_name], key=lambda ev: ev.t0)],
            "conflicts": conflicts,
            "action": action,
        }
    _dedup_cross_label_collisions(suggestions)
    return suggestions


def _dedup_cross_label_collisions(
    suggestions: dict[str, dict[str, Any]]
) -> None:
    """Keep one label per first-name bucket, demote the rest (ported as-is).

    Two diar labels claiming the same first name almost always mean one
    real speaker was split into two clusters; naming both would be
    misleading, so only the highest-confidence label keeps the name.
    """
    buckets: dict[str, list[str]] = {}
    for label, entry in suggestions.items():
        if entry["action"] != "resolved":
            continue
        bucket = entry["name"].split()[0].lower()
        buckets.setdefault(bucket, []).append(label)
    for bucket, labels in buckets.items():
        if len(labels) <= 1:
            continue
        ranked = sorted(labels, key=lambda label: (-suggestions[label]["confidence"], label))
        winner = ranked[0]
        for loser in ranked[1:]:
            LOGGER.info(
                "self-ID demoted for cross-label collision on first name %r: "
                "label=%s loses to label=%s",
                bucket,
                loser,
                winner,
            )
            suggestions.pop(loser)


def _coerce_extra(
    extra_evidence: Iterable[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]
) -> list[Evidence]:
    known_labels = {segment["speaker"] for segment in segments}
    out: list[Evidence] = []
    for raw in extra_evidence:
        label = str(raw.get("label", ""))
        name = str(raw.get("claimed_name", "")).strip()
        if not name or label not in known_labels:
            LOGGER.info("extra evidence ignored for unknown label or empty name: %r", raw)
            continue
        out.append(
            Evidence(
                source=str(raw.get("source", "extra")),
                label=label,
                claimed_name=name,
                weight=float(raw.get("weight", 0.0)),
                snippet=str(raw.get("text") or raw.get("snippet") or ""),
                t0=float(raw.get("timestamp") or raw.get("t0") or 0.0),
                extra=dict(raw.get("extra") or {}),
            )
        )
    return out


def _patterns_from_environment() -> (
    tuple[tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]]
):
    return (
        _compile(_configured_patterns("SELF_ID_PATTERNS", DEFAULT_SELF_ID_PATTERNS)),
        _compile(_configured_patterns("SELF_ID_HANDOFF_PATTERNS", DEFAULT_HANDOFF_PATTERNS)),
        _compile(_configured_patterns("SELF_ID_INTRO_PATTERNS", DEFAULT_THIRD_PARTY_INTRO_PATTERNS)),
    )


def _configured_patterns(name: str, defaults: Sequence[str]) -> tuple[str, ...]:
    configured = os.environ.get(name)
    if not configured:
        return tuple(defaults)
    try:
        decoded = json.loads(configured)
    except json.JSONDecodeError:
        patterns: tuple[str, ...] = (configured,)
    else:
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError(f"{name} must be a regex string or JSON array of regex strings")
        patterns = tuple(decoded)
    return patterns


def _compile(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        expression = re.compile(pattern, re.IGNORECASE | re.ASCII)
        if "name" not in expression.groupindex:
            raise ValueError("each self-identification pattern must define a named 'name' capture group")
        compiled.append(expression)
    return tuple(compiled)


def _first_match(
    patterns: Sequence[re.Pattern[str]], text: str
) -> tuple[re.Pattern[str], re.Match[str]] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return pattern, match
    return None


# Words that must never appear inside a candidate name: they turn ordinary
# sentences ("this is a short brief") into false-positive names.
_NAME_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "and", "any", "all", "are", "as", "at", "be", "been", "but",
        "by", "can", "cant", "cannot", "did", "do", "does", "every", "for",
        "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "let", "me", "might", "more",
        "must", "my", "new", "no", "not", "of", "on", "or", "our", "out",
        "she", "should", "some", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "to", "too", "up",
        "us", "was", "we", "were", "what", "when", "which", "with",
        "would", "you", "your",
    }
)


def _candidate_name(
    match: re.Match[str], *, weak_single_word: bool, blocklist: frozenset[str]
) -> str | None:
    name = match.group("name").strip()
    if re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){0,2}", name) is None:
        LOGGER.info("self-ID ignored: name is not plain ASCII words: %r", name)
        return None
    words = name.split()
    if any(word.lower() in _NAME_FUNCTION_WORDS for word in words):
        LOGGER.info("self-ID ignored: %r contains non-name function words", name)
        return None
    lowered = name.lower()
    if any(
        lowered == blocked.lower() or lowered.startswith(blocked.lower() + " ")
        for blocked in blocklist
    ):
        LOGGER.info("self-ID ignored: %r is a blocked false-positive phrase", name)
        return None
    if weak_single_word and len(words) == 1 and len(words[0]) < 4:
        LOGGER.info("self-ID ignored: weak single-word candidate %r is too short", name)
        return None
    return name


def _name_blocklist() -> frozenset[str]:
    entries: set[str] = set(DEFAULT_NAME_BLOCKLIST)
    path = Path(os.environ.get("SELF_ID_BLOCKLIST_PATH", DEFAULT_SELF_ID_BLOCKLIST_PATH))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as error:
        LOGGER.warning("Blocklist file %s is unavailable; using defaults: %s", path, error)
        lines = []
    for raw_line in lines:
        phrase = raw_line.split("#", 1)[0].strip()
        if phrase:
            entries.add(phrase)
    configured = os.environ.get("SELF_ID_NAME_BLOCKLIST")
    if configured:
        try:
            decoded = json.loads(configured)
        except json.JSONDecodeError:
            items = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            if not isinstance(decoded, list) or not all(
                isinstance(item, str) for item in decoded
            ):
                raise ValueError(
                    "SELF_ID_NAME_BLOCKLIST must be a comma-separated string or JSON array"
                )
            items = [item.strip() for item in decoded if item.strip()]
        entries.update(items)
    return frozenset(entries)


def _has_turn_boundary(segments: Sequence[Mapping[str, Any]], index: int) -> bool:
    speaker = segments[index]["speaker"]
    return (
        index > 0 and segments[index - 1]["speaker"] != speaker
    ) or (
        index + 1 < len(segments) and segments[index + 1]["speaker"] != speaker
    )


def _segment_dict(segment: Any) -> dict[str, Any]:
    if is_dataclass(segment) and not isinstance(segment, type):
        source = asdict(segment)
    elif isinstance(segment, Mapping):
        source = dict(segment)
    else:
        raise TypeError("segments must be dataclasses or mappings")
    required = ("text", "t0", "t1", "speaker")
    missing = [f for f in required if f not in source]
    if missing:
        raise ValueError(f"segment is missing required fields: {', '.join(missing)}")
    return {
        "text": str(source["text"]),
        "t0": float(source["t0"]),
        "t1": float(source["t1"]),
        "speaker": str(source["speaker"]),
    }
