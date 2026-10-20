# Pipeline Reference

Stage order inside the worker (`api/main.py process_job_background`, one job
at a time, FIFO by `seq`):

| Stage | Module | Progress band | Gate |
| --- | --- | --- | --- |
| decode | `pipeline.transcribe.decode_audio` | 0.1 | skipped entirely in mock mode |
| transcribe + diarize | `pipeline.transcribe` (CrispASR adapter) | 0.4 → 0.9 | always |
| spell check | `pipeline.spellcheck` | 0.92 | only when `SPELL_CHECK=1` (exact `1`) and not mock |
| speaker self-ID | `pipeline.self_identification` | 0.95 | always (advisory) |
| render | `pipeline.renderer` | 0.98 | always |

Progress values in `status.json` are strictly monotonic; the `asr`/`diar`
progress callbacks map onto the 0.4–0.9 band so a poller can show real
movement during the single whole-file engine invocation.

## CrispASR adapter (`pipeline/transcribe.py`)

Backend verdict (spike card t_7e17589e, GO): the engine runs as a **CLI
subprocess**, one invocation per job over the whole 16 kHz WAV. There is no
in-process NeMo/torch path anymore; chunk-window VRAM math from the NeMo era
applies only to the `pipeline.chunking` utility, which the real path no
longer uses (the engine auto-slices into ~30 s windows internally; measured
1085 MiB peak over the full 39-minute fixture, well under the 3.5 GiB cap).

### Invocation

`build_crispasr_command` composes:

```text
crispasr -f <wav> --backend nemotron \
         -m <GGUF_MODEL or 'auto' + --auto-download> \
         -l <CRISPASR_LANGUAGE, default en-US> \
         --gpu-backend <CRISPASR_GPU_BACKEND, default cuda> \
         --suppress-nst \
         -oj -of <workdir>/crispasr_out \
         [--cache-dir <CRISPASR_CACHE_DIR>] \
         [--diarize --diarize-method <DIAR_METHOD> \
          (--diarize-model auto for sortformer | --diarize-embedder auto otherwise)]
```

`DIAR_METHOD=none` omits the diarization flags; the segment mapping then
collapses everything to a single speaker. The subprocess environment is
`os.environ` with the binary's own directory and `CRISPASR_LIB_DIR`
prepended to `LD_LIBRARY_PATH`, so the bundled/CUDA-13 libraries resolve
before system copies.

### Backend assertion (the CPU-fallback guard)

`probe_crispasr_binary` runs `crispasr --version` once per binary per
process (cached in-process), parses the `ggml backends : ...` line from its
output, and raises `CrispasrError` when the pinned `CRISPASR_GPU_BACKEND`
(default `cuda`) is not in that list. This is deliberate: the CLI **silently** drops to
CPU (~1.7x wall on short clips, ~11.6x on production-length files; card
t_18ee2e43, no error) when `libcudart.so.13`/`libcublas.so.13` are
missing, so a missing CUDA runtime must fail the job fast instead of spending
minutes per job. The probe runs when the worker starts a real-mode job (via
`_validate_runtime_config`); an assertion failure fails that job with the
typed error text persisted in `status.json`.

### Segment mapping (`parse_crispasr_output`)

Engine JSON `transcription` entries map to pipeline Segment dicts
`{text, t0, t1, speaker}`:

- Times: `timestamps.from`/`to` as `HH:MM:SS.mmm` parsed to seconds; items
  without usable times are dropped.
- Text: language-tag stripping (below); rows that are empty after stripping
  are dropped.
- Speaker extraction: `_speaker_raw` searches the entry's `speaker` field
  for `Speaker <N>` (case-insensitive); the value feeds label normalization
  (below). Missing values mean the engine ran without diarization labels.
- Ordering: items are sorted by `(t0, t1)`, then `_clamp_monotonic` enforces
  `t0 <= t1` and non-decreasing `t0` across the run.
- Labels: `_normalize_speakers` maps engine speaker ids to
  `Speaker 1..Speaker N` in order of first appearance — the stable contract
  the renderer and self-ID pass consume.
- Zero surviving rows raises `CrispasrError` (an empty transcript is a job
  failure, never a silent fallback). A zero exit without the `.json` output
  file is likewise a hard `CrispasrError`.

### Fragment merge

Nemotron's fixed 30 s auto-chunks orphan tail words into micro-segments whose
too-short embedding window mis-clusters them to a spurious extra speaker
(spike finding: 15/16 of sub-2 s fragments on the 39-minute fixture
mis-clustered). `MIN_FRAGMENT_SECONDS` (default 2.0, `0` disables) merges any
segment shorter than the threshold into its preceding turn: text is appended
with a single space and the neighbor's speaker label wins; the time envelope
extends to cover both. The first fragment has no predecessor and keeps its
own slot.

### `<en-US>` suppression

The engine streams can embed the inline language token (`<en-US> …`) into
transcript text. Suppression is belt-and-braces:

1. `--suppress-nst` on the command line (engine-side non-speech token
   stripping).
2. `_strip_lang_tags` in the parser: regex
   `\s*<\s*[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})?\s*>` removes any remaining
   `<lang>` / `<lang-REGION>` tag from segment text before the row is kept.

Neither mechanism touches timing or labels; both run before fragment merge.

### VRAM instrumentation

A sampler thread polls
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` (period
`CRISPASR_VRAM_SAMPLE_SECONDS`, default 1 s, floor 0.1 s; logs
`VRAM sampling disabled` and runs unobserved where `nvidia-smi` is absent).
After each run the adapter logs
`VRAM stage=… allocated=…MiB peak=…MiB samples=… cap=…MiB status=ok|EXCEEDED|unobserved`
at INFO (visible via `docker logs`) and raises `TranscriptionOOM`
(`peak VRAM N MiB exceeded cap M MiB`) when the sampled peak crossed
`VRAM_LIMIT_MIB` (default 3584). Wall-clock overrun of
`CRISPASR_TIMEOUT_SECONDS` is enforced by `subprocess.run` (kills the child)
and raises `CrispasrError`.
`LAST_RUN_STATS` holds the last run's `engine_seconds`, `vram_peak_mib`,
`vram_samples`, `vram_limit_mib`, and `segments` for operators and
`scripts/vm_integration.py`.

### Mock path

`MOCK_PIPELINE=1` (or `mock=True`) returns two deterministic segments
(Speaker 1 greeting + Speaker 2 reply, 0.0–7.2 s) without touching ffmpeg,
the binary, or the GPU, so the full queue/render path is exercisable on
GPU-free CI.

## Spell check (`pipeline/spellcheck.py`)

Vocabulary-only correction over Segment records; text absent from the
vocabulary is never modified. Exact case-insensitive matches are canonicalized
at every term length; for single-word vocabulary terms of six or more
characters, a unique Damerau-Levenshtein distance-1 near miss is also
canonicalized. Timing and speaker fields are preserved by construction. The
worker runs this stage only when `SPELL_CHECK=1` exactly — see CONFIG.md for
the module-level default difference.

## Weighted-evidence speaker self-ID (`pipeline/self_identification.py`)

Advisory pass: it returns provenance-rich suggestions per diar label and
**never mutates** segments, timing, or labels. Evidence sources and weights:

| Source | Base weight | Attribution target |
| --- | --- | --- |
| `self_intro` ("my name is X", "this is X", "X here", …) | 0.7 | the speaking label |
| `third_party_intro` ("joining me today is X", role apposition, …) | 0.5 | the *next different speaker* |
| `handoff` ("handing it over to X", "the floor goes to X", …) | 0.4 | the *next different speaker* |
| `extra_evidence` (v2 seam, e.g. an LLM review pass) | caller-supplied (default 0.0) | caller-supplied label |

Next-speaker attribution is gated by timing from the start of the saying
segment: within 5 s the tight bonus `+0.3` applies, within 30 s the loose
bonus `+0.1`; beyond that the attribution is dropped (logged).

Aggregation per `(label, name)` sums weights — stacking the same name adds
weight; competing names with total above `CONFLICT_MIN_WEIGHT` (0.3) are
listed in `conflicts` and cost the winner `CONFLICT_PENALTY` (0.2).
Confidence is capped at `CONFIDENCE_CAP` (0.95). Thresholds: below
`SELF_ID_ANNOTATE_THRESHOLD` (0.4) the suggestion is dropped entirely; at or
above `SELF_ID_RESOLVE_THRESHOLD` (0.7) `action` is `"resolved"`, otherwise
`"suggested"`. A plausibility validator (ASCII name grammar, function-word
rejects) plus the configurable false-positive blocklist filters phrases that
look like names but are not. Cross-label first-name collisions are
deduplicated: only the highest-confidence label keeps the name.

The E4 LLM review pass is wired as the pure `extra_evidence` seam, not as an
environment switch (see CONFIG.md, "v2 seams"), so model output never enters
the service's request context implicitly.

## Render contract (`pipeline/renderer.py`)

Per SPEC 08, section "Render contract — speaker naming" — this **supersedes**
the original annotate-only label design (SPEC "Output markdown" keeps the
field table, "Transcript" the line shape, "Appendix" the raw segment dump):

- **Metadata table.** `# <title>` then `| Field | Value |` rows: Source,
  Duration (`mm:ss`, `h:mm:ss` past an hour), Language (`en-US` unless
  `meta['language']`), Processed (ISO-8601), Speakers, Pipeline.
- **Speaker mapping table.** When at least one speaker resolves, a
  `## Speaker mapping` table (`| Speaker | Name |`) appears **above** the
  transcript, one row per resolved speaker, so downstream search/replace of
  names stays trivial.
- **Direct-name transcript.** For runs crossing the resolve gate the body
  uses the name directly — `**Astro Bob** [00:01:23] — text…` — instead of
  the annotated-label form. Unresolved speakers stay plain
  `**Speaker N** [hh:mm:ss] — …`. Continuation paragraphs of the same
  speaker run (split on gaps greater than `SILENCE_SPLIT_SECONDS`, default
  1.5 s, or label changes) render as bare `[hh:mm:ss] — …`; the label prints
  once per run.
- **Speaker identification evidence section.** `## Speaker identification`
  follows the transcript (toggle: `SELF_ID_SECTION=1`, exact `1`) with a
  summary line (`Resolved X of N speakers by name (Y with lower confidence,
  Z unidentified).`), then one bullet per label in speaker-number order:
  resolved speakers as `- **Name** — Speaker N (\`label\`), confidence=…,
  sources: …`; suggestions below the resolve gate as `- \`label\` likely
  **Name** (confidence=…; sources: …) — kept as \`Speaker N\` in the
  transcript`; speakers with no evidence as `- **Speaker N** (\`label\`)
  UNIDENTIFIED (no evidence)`. Each named bullet lists its evidence lines
  (\`timestamp\` source (w=weight) “snippet”) and any
  `competing names:` conflict entries. Whenever not all speakers resolved,
  a `### Speaker identification request` block closes the section:
  "Names are advisory evidence, not source truth: transcript labels in the
  appendix remain the authoritative speaker keys." with the
  `SELF_ID_SECTION=0` opt-out printed alongside.
- **Speakers header row.** The metadata `Speakers` cell counts distinct
  post-reconciliation speakers and annotates them: `Speaker N (Name)` for
  resolved, `Speaker N (suggested: Name)` otherwise.
- **Appendix.** The raw normalized segment list (`speaker, t0, t1, text`)
  is embedded after the evidence section as the as-built record —
  the labels there remain the canonical join key even after naming.

Promotion rules the renderer applies to `suggest_speaker_names` output
(precedence order): an explicit boolean `resolved` flag wins; else an
`action` string (`"resolved"` → name in body, `"suggested"` → evidence
section only); else a numeric `confidence` is compared against
`SELF_ID_RESOLVE_THRESHOLD` as a fallback gate; plain-string suggestions
carry no provenance and are treated as resolved.

### Pipeline stamp

The Pipeline metadata row renders the actual backend identity
(card t_3275e056): `autodiarist v1 (CrispASR nemotron Q4_K @<version>)`.
`<version>` is parsed from the same `crispasr --version` output the
CUDA guard consumes — `probe_crispasr_binary` caches it in
`_BACKEND_PROBE_CACHE` and `pipeline.transcribe.backend_identity()`
exposes the label, rendering `@unknown` only until a probe has run in
the process (which the real path always does before rendering). A caller
may override the label via `meta['backend_label']`. Because the probe
never runs in mock mode, `MOCK_PIPELINE=1` renders the deterministic
`autodiarist v1 (mock)`. The pre-pivot `NeMo Speech @<sha>` label and the
legacy meta keys `nemo_speech_sha`/`nemo_sha` are retired: the renderer
ignores them.
