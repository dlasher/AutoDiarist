"""Deterministic project-vocabulary spelling and casing corrections.

Only vocabulary terms are candidates for replacement. Corrections preserve the
number of whitespace-separated words, so callers may retain segment timing and
boundary alignment without remapping offsets.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_VOCAB_PATH = Path("vocab.txt")
_WORD_RE = re.compile(r"\b[\w]+\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpellCheckResult:
    """Spell-check output with job-log-ready correction statistics."""

    segments: list[Any]
    n_fixed: int
    examples: list[str]


@dataclass(frozen=True)
class _VocabularyTerm:
    canonical_words: tuple[str, ...]
    normalized_words: tuple[str, ...]


def spell_check_segments(
    segments: Sequence[Any],
    *,
    vocab_path: str | Path | None = None,
    enabled: bool | None = None,
) -> SpellCheckResult:
    """Apply vocabulary-only corrections while preserving segment identity/timing.

    Exact case-insensitive matches are canonicalized at every term length. For
    single-word vocabulary terms with at least six characters, a unique
    Damerau-Levenshtein distance-one near miss is also canonicalized. No text
    absent from the vocabulary may be introduced or modified.
    """
    copied_segments = list(segments)
    if enabled is None:
        enabled = _env_enabled("SPELL_CHECK", default=True)
    if not enabled:
        return SpellCheckResult(segments=copied_segments, n_fixed=0, examples=[])

    terms = load_vocabulary(_resolve_vocab_path(vocab_path))
    if not terms:
        return SpellCheckResult(segments=copied_segments, n_fixed=0, examples=[])

    corrected: list[Any] = []
    examples: list[str] = []
    n_fixed = 0
    for segment in copied_segments:
        original_text = _get_text(segment)
        corrected_text, fixes = _correct_text(original_text, terms)
        corrected.append(_replace_text(segment, corrected_text))
        n_fixed += len(fixes)
        examples.extend(fixes)

    if n_fixed:
        LOGGER.info("Vocabulary spell-check fixed %d term(s): %s", n_fixed, examples[:5])
    return SpellCheckResult(segments=corrected, n_fixed=n_fixed, examples=examples[:5])


def load_vocabulary(vocab_path: str | Path) -> list[_VocabularyTerm]:
    """Load valid canonical terms, warning rather than failing on bad input."""
    path = Path(vocab_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        LOGGER.warning("Vocabulary file %s is unavailable; spell-check skipped: %s", path, error)
        return []

    terms: list[_VocabularyTerm] = []
    seen: set[tuple[str, ...]] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        term = raw_line.strip()
        if not term:
            LOGGER.warning("Malformed vocabulary line %d in %s: blank term ignored", line_number, path)
            continue
        if _WHITESPACE_RE.search(term) and "  " in term:
            LOGGER.warning("Malformed vocabulary line %d in %s: repeated whitespace ignored", line_number, path)
            continue
        words = tuple(_WORD_RE.findall(term))
        if not words or " ".join(words) != term:
            LOGGER.warning("Malformed vocabulary line %d in %s: %r ignored", line_number, path, raw_line)
            continue
        normalized = tuple(word.casefold() for word in words)
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(_VocabularyTerm(canonical_words=words, normalized_words=normalized))
    return terms


def _correct_text(text: str, terms: Sequence[_VocabularyTerm]) -> tuple[str, list[str]]:
    tokens = list(_WORD_RE.finditer(text))
    replacements: list[tuple[int, int, str, str]] = []
    index = 0
    while index < len(tokens):
        match = _exact_term_match(tokens, index, terms)
        if match is not None:
            term, size = match
            source = text[tokens[index].start() : tokens[index + size - 1].end()]
            replacement = " ".join(term.canonical_words)
            if source != replacement:
                replacements.append((tokens[index].start(), tokens[index + size - 1].end(), replacement, f"{source} -> {replacement}"))
            index += size
            continue

        token = tokens[index]
        source = token.group(0)
        fuzzy = _unique_fuzzy_term(source, terms)
        if fuzzy is not None:
            replacement = fuzzy.canonical_words[0]
            replacements.append((token.start(), token.end(), replacement, f"{source} -> {replacement}"))
        index += 1

    if not replacements:
        return text, []
    corrected = text
    for start, end, replacement, _ in reversed(replacements):
        corrected = corrected[:start] + replacement + corrected[end:]
    return corrected, [example for _, _, _, example in replacements]


def _exact_term_match(
    tokens: Sequence[re.Match[str]], index: int, terms: Sequence[_VocabularyTerm]
) -> tuple[_VocabularyTerm, int] | None:
    candidates = sorted(terms, key=lambda term: len(term.normalized_words), reverse=True)
    for term in candidates:
        size = len(term.normalized_words)
        if index + size > len(tokens):
            continue
        actual = tuple(tokens[index + offset].group(0).casefold() for offset in range(size))
        if actual == term.normalized_words:
            return term, size
    return None


def _unique_fuzzy_term(word: str, terms: Sequence[_VocabularyTerm]) -> _VocabularyTerm | None:
    normalized = word.casefold()
    candidates = [
        term
        for term in terms
        if len(term.normalized_words) == 1
        and len(term.normalized_words[0]) >= 6
        and _damerau_levenshtein_at_most_one(normalized, term.normalized_words[0])
    ]
    return candidates[0] if len(candidates) == 1 else None


def _damerau_levenshtein_at_most_one(left: str, right: str) -> bool:
    """Return whether two words have edit distance at most one, transposition included."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differing = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(differing) == 1:
            return True
        return (
            len(differing) == 2
            and differing[1] == differing[0] + 1
            and left[differing[0]] == right[differing[1]]
            and left[differing[1]] == right[differing[0]]
        )

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    offset = 0
    while offset < len(shorter) and shorter[offset] == longer[offset]:
        offset += 1
    return shorter[offset:] == longer[offset + 1 :]


def _get_text(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment["text"])
    return str(segment.text)


def _replace_text(segment: Any, text: str) -> Any:
    if isinstance(segment, dict):
        return {**segment, "text": text}
    return replace(segment, text=text)


def _resolve_vocab_path(vocab_path: str | Path | None) -> Path:
    if vocab_path is not None:
        return Path(vocab_path)
    return Path(os.environ.get("VOCAB_PATH", DEFAULT_VOCAB_PATH))


def _env_enabled(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}
