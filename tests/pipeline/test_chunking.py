from __future__ import annotations

import math

import pytest

from pipeline.chunking import (
    ChunkProcessingError,
    ConfigError,
    Segment,
    build_chunk_windows,
    chunk_settings,
    process_chunked,
    reconcile_chunks,
)


def test_windows_cover_25_minutes_with_only_configured_overlap() -> None:
    windows = build_chunk_windows(duration=25 * 60, chunk_seconds=600, overlap_seconds=30)

    assert [(window.t0, window.t1) for window in windows] == [
        (0.0, 600.0),
        (570.0, 1170.0),
        (1140.0, 1500.0),
    ]
    assert windows[0].t0 == 0.0
    assert windows[-1].t1 == 1500.0
    assert all(left.t1 - right.t0 == 30.0 for left, right in zip(windows, windows[1:]))
    assert all(window.t0 < window.t1 for window in windows)


def test_same_overlap_voice_maps_to_one_global_speaker() -> None:
    chunks = [
        [Segment(text="first", t0=560, t1=590, speaker="0")],
        [Segment(text="second", t0=580, t1=610, speaker="4")],
    ]
    embeddings = {
        (0, "0"): [1.0, 0.0],
        (1, "4"): [0.99, 0.01],
    }

    result = reconcile_chunks(
        chunks,
        windows=build_chunk_windows(610, 600, 30),
        embedding_for=lambda chunk_id, local_idx, overlap: embeddings[(chunk_id, local_idx)],
        similarity_threshold=0.8,
    )

    assert [segment.speaker for segment in result] == ["g1", "g1"]
    assert [segment.local_idx for segment in result] == ["0", "4"]
    assert result[1].match_confidence > 0.8


def test_ambiguous_overlap_gets_stable_new_label_and_provenance() -> None:
    chunks = [
        [Segment(text="one", t0=560, t1=590, speaker="0")],
        [Segment(text="two", t0=580, t1=610, speaker="1")],
    ]
    embeddings = {
        (0, "0"): [1.0, 0.0],
        (1, "1"): [0.0, 1.0],
    }

    result = reconcile_chunks(
        chunks,
        windows=build_chunk_windows(610, 600, 30),
        embedding_for=lambda chunk_id, local_idx, overlap: embeddings[(chunk_id, local_idx)],
        similarity_threshold=0.8,
    )

    assert [segment.speaker for segment in result] == ["g1", "g2"]
    assert result[1].chunk_id == 1
    assert result[1].local_idx == "1"
    assert result[1].match_confidence == 0.0


def test_invalid_environment_settings_fail_fast(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SECONDS", "30")
    monkeypatch.setenv("OVERLAP_SECONDS", "30")

    with pytest.raises(ConfigError, match="OVERLAP_SECONDS"):
        chunk_settings()


def test_chunk_failure_is_reported_with_its_index() -> None:
    def process(window):
        if window.chunk_id == 1:
            raise RuntimeError("fake diarizer error")
        return [Segment(text="ok", t0=window.t0, t1=window.t0 + 1, speaker="0")]

    with pytest.raises(ChunkProcessingError, match=r"chunk 1.*fake diarizer error"):
        process_chunked(
            duration=1500,
            process_chunk=process,
            embedding_for=lambda chunk_id, local_idx, overlap: [1.0, 0.0],
        )


def test_chunked_processing_returns_time_ordered_global_segments() -> None:
    def process(window):
        return [Segment(text=str(window.chunk_id), t0=window.t0, t1=window.t0 + 1, speaker="0")]

    result = process_chunked(
        duration=1500,
        process_chunk=process,
        embedding_for=lambda chunk_id, local_idx, overlap: [1.0, 0.0],
    )

    assert [(segment.text, segment.speaker, segment.chunk_id) for segment in result] == [
        ("0", "g1", 0),
        ("1", "g1", 1),
        ("2", "g1", 2),
    ]
    assert result == sorted(result, key=lambda segment: (segment.t0, segment.t1))
