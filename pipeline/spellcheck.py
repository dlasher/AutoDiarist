"""Deterministic project-vocabulary spelling and casing corrections.

Only vocabulary terms (and their explicitly recorded garble aliases) are
candidates for replacement. No text absent from the vocabulary may be
introduced or modified.

Word-count contract (revisited deliberately, card t_72405565 AC5): exact
phrase matches preserve the number of whitespace-separated words. The
term-boundary and stutter rules added for the measured Q4_K error classes
change the word count inside a segment BY DESIGN, because those garbles
themselves change it ('its av' -> 'ITCAV' merges, 'contractholders' ->
'contract holders' splits, a 'c c c c c' stutter run collapses). Each such
rule fires only on an exact or unique Damerau-Levenshtein distance-one match
against a vocabulary form, so corrections never introduce invented words.
Segment count, order and per-segment timing (t0/t1) are preserved by
construction, which is the alignment property callers actually depend on.
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
# Absolute, package-relative default: repo root next to pipeline/ in a
# checkout, /app/vocab.txt in the shipped image (WORKDIR /app). No CWD
# dependence (card t_72405565 blocker 3).
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.txt"
_WORD_RE = re.compile(r"\b[\w]+\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_ALIAS_SEPARATOR = "=>"
_NEAR_MISS_MIN_TERM_LEN = 4
_JOIN_MIN_TERM_LEN = 6
_JOIN_MAX_WINDOW_TOKENS = 3
_STUTTER_MIN_RUN = 3
# Guard set (never an output source): high-frequency function words. A
# distance-one JOIN/SPLIT window whose tokens are ALL ordinary prose words is
# never corrected — that is what keeps 'as a', 'was a' and 'it can' from
# collapsing into NASA/ITCAV while 'its av' (av is not an English word) still
# reaches the measured ITCAV garble. Exact-boundary joins stay allowed.
_COMMON_WORDS = frozenset(
    """a an the as at by for from how in of off on or to up us is was were we
    had has have he she it they you your theirs his her this that these those
    and but not no yes with without do does did done can could will would
    should may might must be been being am me him them my our its i you all
    any some such only own same too very s t re ve ll just don now here there
    what when where who which why if then else so than once again over under
    out about against between during into through ooh ah hmm ok okay""".split()
)


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
    aliases: tuple[tuple[str, ...], ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.normalized_words)

    @property
    def joined(self) -> str:
        return "".join(self.normalized_words)


def spell_check_enabled() -> bool:
    """Single enablement rule shared by the API seam and the module (AC8).

    The layer is ON by default; SPELL_CHECK in {0, false, no, off} opts out.
    """
    return _env_enabled("SPELL_CHECK", default=True)


def spell_check_segments(
    segments: Sequence[Any],
    *,
    vocab_path: str | Path | None = None,
    enabled: bool | None = None,
) -> SpellCheckResult:
    """Apply vocabulary-only corrections while preserving segment identity/timing.

    Exact case-insensitive matches are canonicalized at every term length,
    including explicit garble-alias lines ('sout => SEWP'). Single-word terms
    of at least four characters also accept a unique Damerau-Levenshtein
    distance-one near miss. Term-boundary garbles are fixed by joining or
    splitting adjacent tokens when the combined span matches a vocabulary
    term exactly or by a unique distance-one edit (multi-word joins require a
    term of at least six characters). Repeated single-character token runs
    (the measured 'c c c cuture' hallucination pattern) are collapsed. No
    text absent from the vocabulary is introduced or modified.
    """
    copied_segments = list(segments)
    if enabled is None:
        enabled = spell_check_enabled()
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
    """Load valid canonical terms (with optional garble aliases), warning rather
    than failing on bad input.

    Line forms: ``Canonical term`` or ``garble1, garble2 => Canonical term``.
    Lines starting with ``#`` are comments and are skipped silently.
    """
    path = Path(vocab_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        LOGGER.warning("Vocabulary file %s is unavailable; spell-check skipped: %s", path, error)
        return []

    index: dict[tuple[str, ...], _VocabularyTerm] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        term = raw_line.strip()
        if term.startswith("#"):
            continue
        aliases: tuple[tuple[str, ...], ...] = ()
        if _ALIAS_SEPARATOR in term:
            alias_part, _, canonical_part = term.partition(_ALIAS_SEPARATOR)
            if _ALIAS_SEPARATOR in canonical_part:
                LOGGER.warning("Malformed vocabulary line %d in %s: %r ignored", line_number, path, raw_line)
                continue
            canonical_part = canonical_part.strip()
            parsed_aliases: list[tuple[str, ...]] = []
            bad_alias = False
            for alias in alias_part.split(","):
                alias_words = _parse_term_words(alias.strip())
                if alias_words is None:
                    LOGGER.warning(
                        "Malformed vocabulary line %d in %s: alias %r ignored",
                        line_number, path, alias.strip() or "<empty>",
                    )
                    bad_alias = True
                    break
                parsed_aliases.append(alias_words)
            if bad_alias:
                continue
            words = _parse_term_words(canonical_part)
            if not canonical_part or words is None:
                LOGGER.warning("Malformed vocabulary line %d in %s: %r ignored", line_number, path, raw_line)
                continue
            aliases = tuple(parsed_aliases)
        elif not term:
            LOGGER.warning("Malformed vocabulary line %d in %s: blank term ignored", line_number, path)
            continue
        else:
            words = _parse_term_words(term)
            if words is None:
                LOGGER.warning("Malformed vocabulary line %d in %s: %r ignored", line_number, path, raw_line)
                continue

        normalized = tuple(word.casefold() for word in words)
        existing = index.get(normalized)
        if existing is not None:
            merged_aliases = existing.aliases
            for alias in aliases:
                if alias not in merged_aliases:
                    merged_aliases = merged_aliases + (alias,)
            if merged_aliases != existing.aliases:
                index[normalized] = replace(existing, aliases=merged_aliases)
            continue
        index[normalized] = _VocabularyTerm(
            canonical_words=words, normalized_words=normalized, aliases=aliases
        )
    return list(index.values())


def _parse_term_words(term: str) -> tuple[str, ...] | None:
    """Validate one canonical or alias phrase; None when malformed."""
    words = tuple(_WORD_RE.findall(term))
    if not words or " ".join(words) != term:
        return None
    return words


def _correct_text(text: str, terms: Sequence[_VocabularyTerm]) -> tuple[str, list[str]]:
    tokens = list(_WORD_RE.finditer(text))
    if not tokens:
        return text, []

    phrases, single_words = _build_phrase_index(terms)
    phrase_sizes = sorted({len(key) for key in phrases}, reverse=True)
    replacements: list[tuple[int, int, str, str]] = []
    index = 0
    while index < len(tokens):
        exact = _exact_term_match(tokens, index, phrases, phrase_sizes)
        if exact is not None:
            term, size = exact
            source = text[tokens[index].start() : tokens[index + size - 1].end()]
            replacement = _replacement_text(term, source)
            if source != replacement:
                replacements.append((tokens[index].start(), tokens[index + size - 1].end(), replacement, f"{source} -> {replacement}"))
            index += size
            continue

        stutter = _stutter_collapse(text, tokens, index)
        if stutter is not None:
            start, end, replacement, example, consumed = stutter
            replacements.append((start, end, replacement, example))
            index += consumed
            continue

        joined = _term_join_match(tokens, index, terms, single_words)
        if joined is not None:
            term, size = joined
            source = text[tokens[index].start() : tokens[index + size - 1].end()]
            replacement = _replacement_text(term, source)
            if source != replacement:
                replacements.append((tokens[index].start(), tokens[index + size - 1].end(), replacement, f"{source} -> {replacement}"))
            index += size
            continue

        token = tokens[index]
        source = token.group(0)
        fuzzy = _unique_fuzzy_term(source, terms)
        if fuzzy is not None:
            replacement = _replacement_text(fuzzy, source)
            if source != replacement:
                replacements.append((token.start(), token.end(), replacement, f"{source} -> {replacement}"))
        index += 1

    if not replacements:
        return text, []
    corrected = text
    for start, end, replacement, _ in reversed(replacements):
        corrected = corrected[:start] + replacement + corrected[end:]
    return corrected, [example for _, _, _, example in replacements]


def _build_phrase_index(
    terms: Sequence[_VocabularyTerm],
) -> tuple[dict[tuple[str, ...], _VocabularyTerm], set[str]]:
    """Casefolded exact-phrase index (canonical wins over alias collisions)
    plus the set of single-word forms for join/split safety guards."""
    phrases: dict[tuple[str, ...], _VocabularyTerm] = {}
    single_words: set[str] = set()
    for term in terms:
        phrases.setdefault(term.normalized_words, term)
    for term in terms:
        for alias in term.aliases:
            phrases.setdefault(alias, term)
    for term in terms:
        if term.word_count == 1:
            single_words.add(term.normalized_words[0])
        for alias in term.aliases:
            if len(alias) == 1:
                single_words.add(alias[0])
    return phrases, single_words


def _exact_term_match(
    tokens: Sequence[re.Match[str]],
    index: int,
    phrases: dict[tuple[str, ...], _VocabularyTerm],
    phrase_sizes: Sequence[int],
) -> tuple[_VocabularyTerm, int] | None:
    for size in phrase_sizes:
        if index + size > len(tokens):
            continue
        actual = tuple(tokens[index + offset].group(0).casefold() for offset in range(size))
        term = phrases.get(actual)
        if term is not None:
            return term, size
    return None


def _stutter_collapse(
    text: str, tokens: Sequence[re.Match[str]], index: int
) -> tuple[int, int, str, str, int] | None:
    """Collapse repeated single-character token runs (the measured
    'c c c c c c c c c cuture' hallucination). A run of three or more
    identical single alphabetic characters collapses to one token, or is
    absorbed entirely when the following token starts with the same
    character (the stutter was the engine emitting the letter before the
    word it then mangled). Multi-character repeats ('no no no') are natural
    language and are never touched. Returns the replacement span, the
    replacement text, a log example, and how many tokens were consumed."""
    token = tokens[index]
    source_word = token.group(0)
    normalized = source_word.casefold()
    if len(source_word) != 1 or not normalized.isalpha():
        return None
    run_end = index
    while run_end + 1 < len(tokens) and tokens[run_end + 1].group(0).casefold() == normalized:
        run_end += 1
    if run_end - index + 1 < _STUTTER_MIN_RUN:
        return None
    run_source = text[token.start() : tokens[run_end].end()]
    consumed = run_end - index + 1
    next_token = tokens[run_end + 1] if run_end + 1 < len(tokens) else None
    if next_token is not None and next_token.group(0).casefold().startswith(normalized):
        # The span stops at the following token, which survives; that token
        # still gets its own pass through the rules afterwards.
        return (
            token.start(),
            next_token.start(),
            "",
            f"{run_source} -> (collapsed into {next_token.group(0)!r})",
            consumed,
        )
    return token.start(), tokens[run_end].end(), source_word, f"{run_source} -> {source_word}", consumed


def _term_join_match(
    tokens: Sequence[re.Match[str]],
    index: int,
    terms: Sequence[_VocabularyTerm],
    single_words: set[str],
) -> tuple[_VocabularyTerm, int] | None:
    """Match a vocabulary term across token boundaries: split one garbled
    token ('contractholders' -> 'contract holders') or join adjacent tokens
    ('its av' -> 'ITCAV'). Window spans one to three tokens. Exact joined
    matches always qualify; distance-one joins are restricted to same-length
    spans (substitution/transposition only — insertions would let 'a
    contract holder' or 'soup contract' be mangled), require a term of at
    least six characters when the term is multi-word (four when single-word),
    and are refused when every window token is an ordinary English word.
    Distance-zero beats distance-one, zero token-count change beats a
    change, and any tie between distinct terms is treated as ambiguous and
    skipped."""
    candidates: list[tuple[int, int, int, _VocabularyTerm]] = []
    max_size = min(_JOIN_MAX_WINDOW_TOKENS, len(tokens) - index)
    for size in range(1, max_size + 1):
        window = [tokens[index + offset].group(0).casefold() for offset in range(size)]
        if any(word in single_words for word in window):
            continue
        joined = "".join(window)
        all_common = all(word in _COMMON_WORDS for word in window)
        for term in terms:
            if term.word_count == 1 and size == 1:
                continue
            if len(joined) != len(term.joined):
                if joined != term.joined:
                    continue  # unequal lengths only ever qualify as exact joins
            elif all_common:
                continue  # prose like 'was a'/'it can' is never fuzzy-joined
            distance = _damerau_levenshtein_distance_at_most_one(joined, term.joined)
            if distance is None:
                continue
            if distance == 1 and len(term.joined) < (
                _JOIN_MIN_TERM_LEN if term.word_count >= 2 else _NEAR_MISS_MIN_TERM_LEN
            ):
                continue
            candidates.append((distance, abs(size - term.word_count), size, term))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    best = candidates[0]
    tied = {item[3] for item in candidates if item[:3] == best[:3]}
    if len(tied) > 1:
        return None
    term = best[3]
    return term, best[2]


def _unique_fuzzy_term(word: str, terms: Sequence[_VocabularyTerm]) -> _VocabularyTerm | None:
    normalized = word.casefold()
    if normalized in _COMMON_WORDS:
        return None  # 'seep' is a word; near-miss correction is for garbles only
    candidates = [
        term
        for term in terms
        if len(term.normalized_words) == 1
        and len(term.normalized_words[0]) >= _NEAR_MISS_MIN_TERM_LEN
        and _damerau_levenshtein_at_most_one(normalized, term.normalized_words[0])
    ]
    return candidates[0] if len(candidates) == 1 else None


def _damerau_levenshtein_at_most_one(left: str, right: str) -> bool:
    """Return whether two words have edit distance at most one, transposition included."""
    return _damerau_levenshtein_distance_at_most_one(left, right) is not None


def _damerau_levenshtein_distance_at_most_one(left: str, right: str) -> int | None:
    """Return 0 (equal), 1 (one edit or transposition) or None (>1)."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return None
    if len(left) == len(right):
        differing = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(differing) == 1:
            return 1
        if (
            len(differing) == 2
            and differing[1] == differing[0] + 1
            and left[differing[0]] == right[differing[1]]
            and left[differing[1]] == right[differing[0]]
        ):
            return 1
        return None

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    offset = 0
    while offset < len(shorter) and shorter[offset] == longer[offset]:
        offset += 1
    return 1 if shorter[offset:] == longer[offset + 1 :] else None


def _replacement_text(term: _VocabularyTerm, source: str) -> str:
    """Canonical text with case preserved word-by-word for all-lowercase
    vocabulary terms ('Contract holder' stays capitalized, 'CONTRACT HOLDERS'
    stays shouted — forcing the lowercase canonical would damage correct
    text). Canonicals that already carry capitals (acronyms and garble fixes
    like SEWP/NASA/ITCAV) are emitted exactly as spelled."""
    canonical = term.canonical_words
    if any(word != word.casefold() for word in canonical):
        return " ".join(canonical)
    source_words = source.split()
    corrected: list[str] = []
    for position, word in enumerate(canonical):
        sample = source_words[position] if position < len(source_words) else ""
        if sample.isupper() and any(c.isalpha() for c in sample):
            corrected.append(word.upper())
        elif sample[:1].isupper():
            corrected.append(word[:1].upper() + word[1:])
        else:
            corrected.append(word)
    return " ".join(corrected)


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
    env_value = os.environ.get("VOCAB_PATH")
    if env_value:
        return Path(env_value)
    return DEFAULT_VOCAB_PATH


def _env_enabled(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}
