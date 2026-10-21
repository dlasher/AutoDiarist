"""Opt-in LLM cleanup pass over a diarized transcript (V2, card t_48c2efae).

The LLM may only fix spelling, punctuation, and capitalization/formatting.
It is never allowed to alter source content: wording, sentence boundaries,
names/identifiers, speaker labels, timestamps, or line count are all
protected by a mandatory mechanical verification step. A chunk whose
response violates the invariants is REJECTED and the original text is kept.

Interface discipline mirrors the v2 seam precedent (``SELF_ID_LLM`` /
``extra_evidence`` in :mod:`pipeline.self_identification`):

* the model callable is **injected** by the caller
  (``run_llm_cleanup(segments, llm=...)``) — this module performs no
  network I/O and ships no HTTP client;
* the feature is **off unless a caller passes a callable** — no
  environment toggle hardwires a network call;
* everything tunable lives in :class:`CleanupSettings` (chunk budget,
  overlap, timeout, retries, model/endpoint description). Generation
  temperature is fixed at ``0.0`` and enforced as an assertion rather
  than configured.

Pipeline: segments -> canonical cleanup lines -> turn-aware chunks with a
seam overlap -> preface-guarded prompt -> injected LLM -> verification
(line count, order, numbers, exact ``speaker [t0-t1] —`` prefix, token-level
spelling-only diff) -> accepted text replaces the segment text; rejected
chunks keep the original. A per-chunk accept/reject diff report accompanies
the result, and (when enabled) quoted speaker-name evidence is returned in
``Evidence``-shaped mappings for the existing ``extra_evidence`` seam.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PREFACE_VERSION = 1
PREFACE_PATH = Path(__file__).with_name("llm_preface.md")

#: Canonical line format: everything up to and including SEPARATOR is the
#: immutable prefix (line number is added separately in the prompt).
SEPARATOR = " — "
_LINE_RE = re.compile(r"^(?P<label>.+?) \[(?P<t0>-?\d+\.\d+)-(?P<t1>-?\d+\.\d+)\] — (?P<text>.*)$")

CONTEXT_HEADER = "--- READ-ONLY CONTEXT"
LINES_HEADER = "--- TRANSCRIPT LINES TO CLEANUP"
EVIDENCE_MARKER = "--- NAME EVIDENCE ---"

_NAME_REVIEW_ADDENDUM = (
    "\nOPTIONAL SPEAKER-NAME REVIEW: after your numbered cleanup lines, you may\n"
    f"append the single marker line '{EVIDENCE_MARKER}' followed by at most one evidence\n"
    "line per claim, formatted as '<speaker label> | <full name> | <sentence\n"
    "quoted verbatim from the transcript>'. This section is advisory only: it\n"
    "must never change any transcript line and never alters the speaker labels\n"
    "or timestamps themselves. If you have no name evidence, output nothing\n"
    "after the numbered lines."
)


class CleanupConfigError(ValueError):
    """Cleanup settings are invalid (mirrors pipeline.chunking.ConfigError)."""


def load_preface(path: str | Path | None = None) -> str:
    """Return the versioned prompt preface stating the cleanup rules."""
    preface_path = Path(path) if path is not None else PREFACE_PATH
    text = preface_path.read_text(encoding="utf-8")
    if not text.startswith("PREFACE VERSION:"):
        raise CleanupConfigError(f"{preface_path} must start with 'PREFACE VERSION: n'")
    return text


@dataclass(frozen=True)
class CleanupSettings:
    """Documented settings object for the cleanup pass (no network wiring).

    Environment-tunable knobs (budgets only — never a network toggle):
    ``LLM_CLEANUP_CHUNK_CHARS``, ``LLM_CLEANUP_OVERLAP_LINES``,
    ``LLM_CLEANUP_TIMEOUT_S``, ``LLM_CLEANUP_MAX_RETRIES``,
    ``LLM_CLEANUP_MODEL`` (opaque description string) and
    ``LLM_CLEANUP_REVIEW_NAMES=1``. Generation temperature is deliberately
    NOT environment-configurable: the verification guard assumes a
    deterministic pass and pins it to 0.
    """

    chunk_char_budget: int = 6000
    overlap_lines: int = 1
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    spelling_similarity_threshold: float = 0.6
    review_names: bool = False
    model: str = ""

    def __post_init__(self) -> None:
        if self.chunk_char_budget <= 0:
            raise CleanupConfigError("chunk_char_budget must be positive")
        if self.overlap_lines < 0:
            raise CleanupConfigError("overlap_lines must be non-negative")
        if self.timeout_seconds <= 0:
            raise CleanupConfigError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise CleanupConfigError("max_retries must be non-negative")
        if not 0.0 < self.spelling_similarity_threshold <= 1.0:
            raise CleanupConfigError("spelling_similarity_threshold must be in (0, 1]")
        if self.temperature != 0.0:
            raise CleanupConfigError(
                "LLM cleanup temperature is fixed at 0 (deterministic pass); "
                f"got {self.temperature!r}"
            )

    @classmethod
    def from_env(cls) -> CleanupSettings:
        kwargs: dict[str, Any] = {}
        if os.environ.get("LLM_CLEANUP_CHUNK_CHARS"):
            kwargs["chunk_char_budget"] = int(os.environ["LLM_CLEANUP_CHUNK_CHARS"])
        if os.environ.get("LLM_CLEANUP_OVERLAP_LINES"):
            kwargs["overlap_lines"] = int(os.environ["LLM_CLEANUP_OVERLAP_LINES"])
        if os.environ.get("LLM_CLEANUP_TIMEOUT_S"):
            kwargs["timeout_seconds"] = float(os.environ["LLM_CLEANUP_TIMEOUT_S"])
        if os.environ.get("LLM_CLEANUP_MAX_RETRIES"):
            kwargs["max_retries"] = int(os.environ["LLM_CLEANUP_MAX_RETRIES"])
        if os.environ.get("LLM_CLEANUP_MODEL"):
            kwargs["model"] = os.environ["LLM_CLEANUP_MODEL"]
        if os.environ.get("LLM_CLEANUP_REVIEW_NAMES"):
            kwargs["review_names"] = os.environ["LLM_CLEANUP_REVIEW_NAMES"] == "1"
        return cls(**kwargs)


@dataclass(frozen=True)
class CleanupChunk:
    """One authoritative slice of the transcript plus its read-only seam."""

    index: int
    start: int  # first owned line index (inclusive)
    end: int  # last owned line index (exclusive)
    context_start: int  # seam context slice (owned by earlier chunks)
    context_end: int


@dataclass(frozen=True)
class ChunkOutcome:
    """Verification verdict for one chunk, for the accept/reject report."""

    index: int
    start: int
    end: int
    accepted: bool
    reason: str
    attempts: int
    changed: bool
    failed: bool = False
    diffs: tuple[tuple[int, str, str], ...] = ()  # (line_index, before, after)


@dataclass
class CleanupResult:
    segments: list[Any]
    outcomes: list[ChunkOutcome] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    changed_lines: int = 0
    name_evidence: list[dict[str, Any]] = field(default_factory=list)
    report_markdown: str = ""


# --- Line helpers -----------------------------------------------------------


def build_cleanup_line(segment: Any) -> str:
    """Canonical cleanup-pass line: ``Speaker 1 [0.00-3.50] — text``."""
    record = _segment_dict(segment)
    return f"{record['speaker']} [{float(record['t0']):.2f}-{float(record['t1']):.2f}]{SEPARATOR}{record['text']}"


def line_prefix(line: str) -> str:
    """The immutable part of a cleanup line (label + timestamps + separator)."""
    idx = line.index(SEPARATOR)
    return line[: idx + len(SEPARATOR)]


def _segment_dict(segment: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(segment) and not isinstance(segment, type):
        return dataclasses.asdict(segment)
    if isinstance(segment, Mapping):
        return dict(segment)
    raise TypeError(f"unsupported segment type: {type(segment)!r}")


def _replace_text(segment: Any, text: str) -> Any:
    if dataclasses.is_dataclass(segment) and not isinstance(segment, type):
        return dataclasses.replace(segment, text=text)
    if isinstance(segment, Mapping):
        record = dict(segment)
        record["text"] = text
        return record
    raise TypeError(f"unsupported segment type: {type(segment)!r}")


# --- Chunking (turn-aware, seam overlap) ------------------------------------


def plan_cleanup_chunks(lines: Sequence[str], settings: CleanupSettings) -> list[CleanupChunk]:
    """Pack whole speaker-turn lines into budget-sized chunks.

    Boundaries only ever fall between lines (a line is one speaker turn, so
    a single speaker's sentence is never split across chunks). A line longer
    than the budget gets its own chunk instead of being cut. Every chunk
    after the first carries the previous chunk's trailing ``overlap_lines``
    lines as read-only context.
    """
    chunks: list[CleanupChunk] = []
    start = 0
    used = 0
    for i, line in enumerate(lines):
        cost = len(line) + 1
        if i > start and used + cost > settings.chunk_char_budget:
            chunks.append(_make_chunk(len(chunks), start, i, lines, settings, chunks))
            start = i
            used = 0
        used += cost
    if lines:
        chunks.append(_make_chunk(len(chunks), start, len(lines), lines, settings, chunks))
    return chunks


def _make_chunk(
    index: int,
    start: int,
    end: int,
    lines: Sequence[str],
    settings: CleanupSettings,
    previous: Sequence[CleanupChunk],
) -> CleanupChunk:
    if not previous:
        return CleanupChunk(index=index, start=start, end=end, context_start=0, context_end=0)
    prev = previous[-1]
    context_start = max(prev.start, start - settings.overlap_lines)
    return CleanupChunk(index=index, start=start, end=end,
                        context_start=context_start, context_end=start)


# --- Prompt assembly ---------------------------------------------------------


def build_prompt(preface: str, context_lines: Sequence[str], lines: Sequence[str],
                 settings: CleanupSettings) -> str:
    parts: list[str] = [preface.strip(), ""]
    parts.append(
        f"{CONTEXT_HEADER} ({len(context_lines)} line(s); already-cleaned transcript seam "
        "from the previous chunk — read for context only, do NOT output these)"
    )
    parts.extend(context_lines if context_lines else ["(none)"])
    parts.append("")
    parts.append(
        f"{LINES_HEADER} — output exactly {len(lines)} line(s), same order, same numbers; "
        "keep each line's speaker label and timestamps unchanged"
    )
    for i, line in enumerate(lines, 1):
        parts.append(f"{i}|{line}")
    if settings.review_names:
        parts.append(_NAME_REVIEW_ADDENDUM)
    return "\n".join(parts)


def authoritative_lines(prompt: str) -> list[str]:
    """Extract the transcript line texts (without ``N|`` numbers) from a prompt."""
    marker = prompt.index(LINES_HEADER)
    tail = prompt[marker + len(LINES_HEADER):]
    out: list[str] = []
    for raw in tail.splitlines():
        num, sep, rest = raw.partition("|")
        if sep and num.strip().isdigit():
            out.append(rest)
    return out


# --- Verification (mandatory safety net) --------------------------------------


def _normalize_token(token: str) -> str:
    return token.strip(string.punctuation).casefold()


def _damerau_levenshtein(left: str, right: str) -> int:
    size_a, size_b = len(left), len(right)
    if size_a == 0 or size_b == 0:
        return max(size_a, size_b)
    table = [[0] * (size_b + 1) for _ in range(size_a + 1)]
    for i in range(size_a + 1):
        table[i][0] = i
    for j in range(size_b + 1):
        table[0][j] = j
    for i in range(1, size_a + 1):
        for j in range(1, size_b + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + cost)
    return table[size_a][size_b]


def _token_similarity(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - _damerau_levenshtein(left, right) / longest


def _verify_body(original_text: str, cleaned_text: str, threshold: float) -> str | None:
    """Return a rejection reason if the LLM went beyond permitted edits."""
    before = original_text.split()
    after = cleaned_text.split()
    if len(before) != len(after):
        return (
            "wording change: token count differs (sentences may not be merged, "
            "split, inserted, or deleted)"
        )
    for k, (a, b) in enumerate(zip(before, after)):
        if a == b:
            continue
        if _normalize_token(a) == _normalize_token(b):
            continue  # capitalization/punctuation-only change
        if _token_similarity(_normalize_token(a), _normalize_token(b)) >= threshold:
            continue  # near-miss: accepted as a spelling fix
        return f"wording change: word '{a}' was replaced with '{b}' (token {k + 1})"
    return None


def _parse_line_label(line: str) -> tuple[str, float]:
    match = _LINE_RE.match(line)
    if not match:
        raise ValueError(f"malformed cleanup line: {line!r}")
    return match.group("label"), float(match.group("t0"))


def _harvest_name_evidence(block: str, lines: Sequence[str]) -> list[dict[str, Any]]:
    """Parse ``Label | Name | snippet`` lines; only chunk-known labels survive."""
    known: dict[str, list[tuple[float, str]]] = {}
    for line in lines:
        label, t0 = _parse_line_label(line)
        known.setdefault(label, []).append((t0, line))
    entries: list[dict[str, Any]] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        pieces = [piece.strip() for piece in raw.split("|")]
        if len(pieces) < 2:
            continue
        label, name = pieces[0], pieces[1]
        snippet = pieces[2] if len(pieces) > 2 else ""
        if not name or label not in known:
            LOGGER.info("llm_cleanup: dropped name evidence for unknown label %r", label)
            continue
        anchor = next(
            (t0 for (t0, line) in known[label] if snippet and snippet in line),
            known[label][0][0],
        )
        entries.append(
            {"label": label, "claimed_name": name, "snippet": snippet,
             "source": "llm_cleanup", "t0": anchor}
        )
    return entries


def verify_chunk_response(
    response: str, lines: Sequence[str], settings: CleanupSettings
) -> tuple[bool, str, list[str], list[dict[str, Any]]]:
    """Verify one chunk response against its original lines.

    Returns ``(ok, reason, cleaned_lines, name_evidence)``. Invariants:
    exact line count, exact order, exact ``N|`` numbers, byte-identical
    speaker-label/timestamp prefixes, and token-level spelling-only body
    edits. Anything else rejects the whole chunk.
    """
    evidence: list[dict[str, Any]] = []
    body = response
    if settings.review_names and EVIDENCE_MARKER in response:
        body, _marker, tail = response.partition(EVIDENCE_MARKER)
        evidence = _harvest_name_evidence(tail, lines)
    returned = [raw for raw in body.splitlines() if raw.strip()]
    if len(returned) != len(lines):
        return (False, f"line count changed: got {len(returned)}, expected {len(lines)}",
                [], evidence)
    cleaned: list[str] = []
    for i, (raw, original) in enumerate(zip(returned, lines), 1):
        num, sep, rest = raw.partition("|")
        if not sep or num.strip() != str(i):
            return (False, f"line order/numbering changed at line {i}", [], evidence)
        prefix = line_prefix(original)
        if not rest.startswith(prefix):
            return (False, ("speaker label or timestamp altered on line "
                            f"{i}: {rest[:40]!r}"), [], evidence)
        reason = _verify_body(original[len(prefix):], rest[len(prefix):],
                              settings.spelling_similarity_threshold)
        if reason:
            return (False, f"line {i} rejected ({reason})", [], evidence)
        cleaned.append(rest)
    return (True, "ok", cleaned, evidence)


# --- Runner -------------------------------------------------------------------


def _call_with_retry(llm: Callable[[str], str], prompt: str,
                     settings: CleanupSettings) -> tuple[str | None, str, int]:
    last_error = ""
    total_attempts = settings.max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return _call_with_timeout(llm, prompt, settings.timeout_seconds), "", attempt
        except Exception as exc:  # noqa: BLE001 - any client failure rejects the chunk
            last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("llm_cleanup: attempt %d/%d failed: %s",
                           attempt, total_attempts, last_error)
    return None, last_error, total_attempts


def _call_with_timeout(llm: Callable[[str], str], prompt: str, timeout: float) -> str:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(llm, prompt)
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_llm_cleanup(
    segments: Sequence[Any],
    *,
    llm: Callable[[str], str] | None,
    settings: CleanupSettings | None = None,
    preface: str | None = None,
) -> CleanupResult:
    """Run the opt-in cleanup pass over ``segments`` with an injected LLM.

    ``llm`` is a caller-supplied ``prompt -> response`` callable (this
    module never talks to a model itself). Chunks that fail verification or
    the client are kept verbatim; only accepted chunks replace segment
    text, and only via permitted spelling/punctuation/formatting edits.
    """
    if llm is None:
        raise TypeError(
            "run_llm_cleanup requires an injected llm callable "
            "(opt-in seam; the service must not hardwire a network call)"
        )
    settings = settings or CleanupSettings()
    preface = preface if preface is not None else load_preface()
    record_list = list(segments)
    out_segments = list(record_list)
    result = CleanupResult(segments=out_segments)
    if not record_list:
        result.report_markdown = render_cleanup_report(result, settings)
        return result

    lines = [build_cleanup_line(segment) for segment in record_list]
    current = list(lines)
    for chunk in plan_cleanup_chunks(lines, settings):
        owned = current[chunk.start:chunk.end]
        context = current[chunk.context_start:chunk.context_end] if chunk.index else []
        prompt = build_prompt(preface, context, owned, settings)
        response, error, attempts = _call_with_retry(llm, prompt, settings)
        if response is None:
            reason = f"LLM call failed after {attempts} attempt(s): {error}"
            outcome = ChunkOutcome(chunk.index, chunk.start, chunk.end, False,
                                   reason, attempts, False, failed=True)
            result.failed += 1
            result.rejected += 1
            result.outcomes.append(outcome)
            continue
        ok, verify_reason, cleaned, evidence = verify_chunk_response(response, owned, settings)
        if not ok:
            outcome = ChunkOutcome(chunk.index, chunk.start, chunk.end, False,
                                   verify_reason, attempts, False)
            result.rejected += 1
            result.outcomes.append(outcome)
            continue
        diffs: list[tuple[int, str, str]] = []
        for offset, (before, after) in enumerate(zip(owned, cleaned)):
            if after != before:
                line_idx = chunk.start + offset
                current[line_idx] = after
                prefix = line_prefix(before)
                result.segments[line_idx] = _replace_text(
                    record_list[line_idx], after[len(prefix):]
                )
                diffs.append((line_idx, before, after))
        result.accepted += 1
        result.changed_lines += len(diffs)
        result.name_evidence.extend(evidence)
        result.outcomes.append(ChunkOutcome(chunk.index, chunk.start, chunk.end, True,
                                            verify_reason, attempts, bool(diffs),
                                            diffs=tuple(diffs)))
    result.report_markdown = render_cleanup_report(result, settings)
    return result


def render_cleanup_report(result: CleanupResult, settings: CleanupSettings) -> str:
    """Per-chunk accept/reject diff report (counts + what changed)."""
    lines = [
        "# LLM Cleanup Report",
        "",
        f"- Preface: {load_preface().splitlines()[0]}",
        f"- Model/endpoint: {settings.model or '(opaque, caller-supplied client)'}",
        (f"- Chunks: {len(result.outcomes)} | accepted: {result.accepted} | "
         f"rejected: {result.rejected} (failed: {result.failed}) | "
         f"permitted line edits: {result.changed_lines}"),
        "",
    ]
    for outcome in result.outcomes:
        status = "ACCEPTED" if outcome.accepted else ("FAILED" if outcome.failed else "REJECTED")
        header = (f"## Chunk {outcome.index + 1} (lines {outcome.start + 1}-"
                  f"{outcome.end}) — {status}")
        lines.append(header)
        if outcome.accepted:
            if not outcome.changed:
                lines.append("- no changes (verbatim echo)")
            for (line_idx, before, after) in outcome.diffs:
                lines.append(f"- line {line_idx + 1}: `{before}` → `{after}`")
        else:
            lines.append(f"- reason: {outcome.reason}")
            lines.append("- action: chunk rejected; original text kept verbatim")
        lines.append("")
    lines.append(f"## Speaker-name evidence harvested: {len(result.name_evidence)}")
    for entry in result.name_evidence:
        lines.append(
            f"- {entry['label']} → “{entry['claimed_name']}” "
            f"(quote: {entry.get('snippet') or 'n/a'})"
        )
    return "\n".join(lines) + "\n"
