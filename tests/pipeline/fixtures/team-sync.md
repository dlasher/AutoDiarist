# team-sync

| Field | Value |
|---|---|
| Source | team-sync.mp3 |
| Duration | 47:12 |
| Language | en-US |
| Processed | 2026-10-16T11:30:00-07:00 |
| Speakers | 2 (Speaker 1 (Jane Wu); Speaker 2) |
| Pipeline | nvidia-asr v1 (CrispASR nemotron Q4_K @v0.8.43) |

## Speaker mapping

| Speaker | Name |
|---|---|
| Speaker 1 | Jane Wu |

## Transcript

**Jane Wu** [00:00:03] — Welcome everyone. Let's start.

**Speaker 2** [00:00:05] — Thanks, Jane.

## Speaker identification

Resolved 1 of 2 speakers by name (0 with lower confidence, 1 unidentified).

- **Jane Wu** — Speaker 1 (`g1`), confidence=n/a; sources: user-supplied.
- **Speaker 2** (`g2`) UNIDENTIFIED (no evidence).

### Speaker identification request

Names are advisory evidence, not source truth: transcript labels in the appendix remain the authoritative speaker keys. Set `SELF_ID_SECTION=0` to omit this section.

## Appendix — Raw Diarizer Segments

```json
[
  {
    "speaker": "g1",
    "t0": 3.0,
    "t1": 4.0,
    "text": "Welcome everyone.",
    "chunk_id": 0,
    "local_idx": "0",
    "match_confidence": 1.0
  },
  {
    "speaker": "g1",
    "t0": 4.2,
    "t1": 5.0,
    "text": "Let's start.",
    "chunk_id": 0,
    "local_idx": "0",
    "match_confidence": 1.0
  },
  {
    "speaker": "g2",
    "t0": 5.4,
    "t1": 6.4,
    "text": "Thanks, Jane.",
    "chunk_id": 0,
    "local_idx": "1",
    "match_confidence": 1.0
  }
]
```

### Suggestion Provenance

```json
{
  "g1": "Jane Wu"
}
```
