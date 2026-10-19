"""Bounded-memory audio chunking and cross-chunk speaker reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from collections.abc import Callable, Sequence

CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "600"))
OVERLAP_SECONDS = float(os.environ.get("OVERLAP_SECONDS", "30"))
DEFAULT_SIMILARITY_THRESHOLD = 0.8


@dataclass(frozen=True)
class ChunkWindow:
    chunk_id: int
    t0: float
    t1: float


@dataclass(frozen=True)
class Segment:
    text: str
    t0: float
    t1: float
    speaker: str
    chunk_id: int | None = None
    local_idx: str | None = None
    match_confidence: float | None = None


class ChunkProcessingError(RuntimeError):
    """A chunk failed and therefore the containing job must be marked failed."""


class ConfigError(ValueError):
    """Chunk configuration cannot produce bounded, overlapping windows."""


def chunk_settings() -> tuple[float, float]:
    """Read and validate runtime chunk configuration at job start."""
    try:
        chunk_seconds = float(os.environ.get("CHUNK_SECONDS", "600"))
        overlap_seconds = float(os.environ.get("OVERLAP_SECONDS", "30"))
    except ValueError as exc:
        raise ConfigError("CHUNK_SECONDS and OVERLAP_SECONDS must be numeric") from exc
    if chunk_seconds <= 0:
        raise ConfigError("CHUNK_SECONDS must be positive")
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ConfigError("OVERLAP_SECONDS must be non-negative and smaller than CHUNK_SECONDS")
    return chunk_seconds, overlap_seconds


def build_chunk_windows(
    duration: float,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[ChunkWindow]:
    """Return contiguous coverage windows with overlap only at chunk boundaries."""
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap_seconds must be non-negative and smaller than chunk_seconds")
    if duration == 0:
        return []

    step = chunk_seconds - overlap_seconds
    windows: list[ChunkWindow] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        windows.append(ChunkWindow(len(windows), start, end))
        if end == duration:
            break
        start += step
    return windows


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _overlap(left: ChunkWindow, right: ChunkWindow) -> tuple[float, float]:
    return max(left.t0, right.t0), min(left.t1, right.t1)


def reconcile_chunks(
    chunks: Sequence[Sequence[Segment]],
    *,
    windows: Sequence[ChunkWindow],
    embedding_for: Callable[[int, str, tuple[float, float]], Sequence[float]],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[Segment]:
    """Assign stable global speaker labels using embeddings from chunk overlaps.

    A local speaker only joins an existing global label when its best overlap
    cosine similarity clears the supplied threshold. Otherwise it intentionally
    receives a new label, preserving uncertainty in its provenance confidence.
    """
    if len(chunks) != len(windows):
        raise ValueError("chunks and windows must have the same length")
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")

    global_embeddings: dict[str, list[float]] = {}
    next_label = 1
    reconciled: list[Segment] = []

    for chunk_id, local_segments in enumerate(chunks):
        window = windows[chunk_id]
        overlap = _overlap(windows[chunk_id - 1], window) if chunk_id else (window.t0, window.t1)
        local_indices = sorted({str(segment.speaker) for segment in local_segments})
        labels_for_local: dict[str, tuple[str, float]] = {}

        for local_idx in local_indices:
            embedding = list(embedding_for(chunk_id, local_idx, overlap))
            candidates = [
                (_cosine(embedding, global_embedding), label)
                for label, global_embedding in global_embeddings.items()
            ]
            confidence, label = max(candidates, default=(0.0, ""))
            if confidence < similarity_threshold:
                label = f"g{next_label}"
                next_label += 1
                confidence = 1.0 if not global_embeddings else max(confidence, 0.0)
            labels_for_local[local_idx] = (label, confidence)
            global_embeddings[label] = embedding

        for segment in local_segments:
            local_idx = str(segment.speaker)
            label, confidence = labels_for_local[local_idx]
            reconciled.append(
                replace(
                    segment,
                    speaker=label,
                    chunk_id=chunk_id,
                    local_idx=local_idx,
                    match_confidence=confidence,
                )
            )

    return sorted(reconciled, key=lambda segment: (segment.t0, segment.t1, segment.chunk_id or 0))


def process_chunked(
    *,
    duration: float,
    process_chunk: Callable[[ChunkWindow], Sequence[Segment]],
    embedding_for: Callable[[int, str, tuple[float, float]], Sequence[float]],
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[Segment]:
    """Process chunks sequentially so only one bounded window is in memory."""
    windows = build_chunk_windows(duration, chunk_seconds, overlap_seconds)
    chunks: list[Sequence[Segment]] = []
    for window in windows:
        try:
            chunks.append(process_chunk(window))
        except Exception as exc:
            raise ChunkProcessingError(f"chunk {window.chunk_id} failed: {exc}") from exc
    return reconcile_chunks(
        chunks,
        windows=windows,
        embedding_for=embedding_for,
        similarity_threshold=similarity_threshold,
    )
