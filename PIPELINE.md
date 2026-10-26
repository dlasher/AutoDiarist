# Pipeline Reference

Stage order inside the worker (`api/main.py process_job_background`, each
worker runs one job at a time; the pool size is `MAX_PROCESSING_GLOBAL`,
default 1 — serial, card t_dbad2235; jobs are claimed fair-share:
round-robin across sessions, FIFO by `seq`
within a session — card t_4b40acfd). Stages are the public `stage` enum (V2, card
t_14ac6b49); bands are the `[start, end)` slice of `[0, 1]` each stage owns:

| Stage | Module | Progress band | Gate |
| --- | --- | --- | --- |
| `queued` | API queue (`seq`-ordered) | 0.00 | initial; also re-set on boot reclaim |
| `decoding` | `pipeline.transcribe.decode_audio` | 0.05–0.30 | skipped entirely in mock mode |
| `transcribing` | `pipeline.transcribe` (CrispASR adapter) | 0.30–0.60 | always; refined per reported ASR chunk |
| `diarizing` | same engine call, second pass | 0.60–0.75 | always; refined per diarized-chunk boundary |
| `spellcheck` | `pipeline.spellcheck` | 0.75–0.80 | ON by default (card t_72405565); skipped when `SPELL_CHECK` opts out (`0/false/no/off`) and in mock mode |
| `llm_cleanup` | `pipeline.llm_cleanup` | 0.80–0.85 | only when the caller injects an `llm` callable into `process_job_background` (V2 seam; the HTTP path never does) |
| `self_identification` | `pipeline.self_identification` | 0.85–0.92 | always (advisory; also folds in cleanup-pass name evidence when present) |
| `rendering` | `pipeline.renderer` | 0.92–1.00 | always |
| `done` | — | 1.00 | terminal; `progress == 1.0` iff done |

Progress values in `status.json` are **monotonic non-decreasing** and
published only at real work boundaries — never from timers. Within a band,
engine-reported per-pass/per-slice counts refine the value (card
t_1eb108e0, and card t_2e901945: the adapter consumes the engine's `-pp`
stderr slice stream plus its stdout chunk lines, mapping `asr`/`diar`
reports onto the `transcribing`/`diarizing` bands via
`api/main.py::_ENGINE_STAGES`); a requeue-then-reclaim never rewinds
progress (monotonicity clamp, card t_14ac6b49 review rework).

Every published status is also append-logged to
`<job_dir>/status_journal.jsonl` (card t_14ac6b49): an audit trail of
transitions — including the boot-reclaim requeue line — that survives the
mutable `status.json` snapshot. Retention/eviction removals are logged to
the service log instead (`reason=age-expiry` / `reason=disk-pressure`).

Finished job dirs are reclaimed by the retention sweep (`RETENTION_DAYS`,
default 30, boot + hourly) and disk-pressure eviction
(`DISK_MIN_FREE_MIB`/`DISK_CHECK_INTERVAL_SECONDS`) — see CONFIG.md; the
`queued`/`processing` pair is never swept or evicted mid-flight (card
t_5e6200fc).

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
crispasr -f <wav> --backend nemotron -pp \
         -m <GGUF_MODEL or 'auto' + --auto-download> \
         -l <CRISPASR_LANGUAGE, default en-US> \
         --gpu-backend <CRISPASR_GPU_BACKEND, default cuda> \
         --suppress-nst \
         -oj -of <workdir>/crispasr_out \
         [--cache-dir <CRISPASR_CACHE_DIR>] \
         [--diarize --diarize-method <DIAR_METHOD> \
          (--diarize-model auto for sortformer | --diarize-embedder auto otherwise) \
          (--sherpa-segment-model <SHERPA_SEGMENT_MODEL> \
           --sherpa-embedding-model <SHERPA_EMBEDDING_MODEL> \
           [--sherpa-bin <SHERPA_BIN>] for sherpa/ecapa)]
```

`DIAR_METHOD=none` omits the diarization flags; the segment mapping then
collapses everything to a single speaker. For `sherpa`/`ecapa` the engine
shells out to the sherpa-onnx CLI (`SHERPA_BIN`, a wrapper that injects
`--clustering.cluster-threshold=<SHERPA_CLUSTER_THRESHOLD>` and threads
both its networks with `--segmentation.num-threads`/`--embedding.num-threads`
= `min(max(1, cores // MAX_PROCESSING_GLOBAL), SHERPA_MAX_THREADS default 4)` —
the CLI's own default is 1 thread and diarization is ~80% of a job's wall
time; cards t_3cb8f97c + t_71a5768f, measured on BOTH shipped hosts: the
optimum is 4 threads everywhere — 2.85x on the 4-core NVIDIA box then a
CLIFF (6+ threads oversubscribe 4 cores; 8 threads is worse than 1), and
2.66x on the 12-core AMD box then a PLATEAU (memory-bandwidth bound;
6-12 threads within ~12% of the optimum) — with identical results at every
thread count); if either
ONNX model is missing the engine emits speaker-less segments — the
adapter refuses that at startup (`CrispasrConfigError`) and rejects it
at runtime (`CrispasrError`) instead of normalizing to one fake speaker
(card t_5a287979 AC4). The subprocess environment is
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

### Live progress on long jobs (card t_2e901945)

What moves `GET /v1/jobs/{token}` during a long transcription, and why it
can be trusted:

* **The engine's `-pp` stderr stream is the in-run signal.** The adapter
  always passes `-pp`; crispasr then writes one line per completed audio
  slice — `crispasr: progress =  X% (k/N slices)` — to stderr, which is
  unbuffered at the C level and lands in the job's `crispasr_stderr.log`
  the moment a slice finishes. The adapter tails that file during the run
  and publishes `("asr", k, N)` reports, so a 39-minute input produced 86
  distinct progress events at real cadence (measured live on the NV host
  against the pinned 0.8.43 binary).
* **Why not stdout, the previously documented signal.** The diarized
  transcript crispasr prints to stdout is app-held until the final merge
  stage. Measured against the pinned binary: a piped stdout yields zero
  bytes for ~80% of the run, and NEITHER `stdbuf -oL` NOR a pty changes
  that — the hold is not stdio buffering, so no shim can fix it. Stdout
  chunk lines still refine progress when the burst finally arrives, which
  is why short jobs always looked healthy before this fix.
* **Never a timer.** No heartbeat, no wall-clock interpolation: a
  progress value that moves without engine evidence is a lie, and a
  progress value that lies about work done is worse than a frozen one.
  Where the engine itself runs a silent blocking stage (the global sherpa
  timeline pre-pass), progress honestly stays at the band floor until the
  next real boundary.

## Spell check (`pipeline/spellcheck.py`)

Vocabulary-only correction over Segment records; text absent from the
vocabulary is never modified. The shipped `vocab.txt` (baked at
`/app/vocab.txt`; the module default resolves to it by absolute path, `VOCAB_PATH`
overrides) carries the NASA/SEWP domain terms measured mangled at Q4_K —
SEWP, NASA, NASA SEWP, GWAC, OMB, ITCAV, contract holder(s), soup contracts,
exceptional — plus explicit garble-alias lines for the measured error forms
(`sout, soupes => SEWP`, `soubcontracts => soup contracts`,
`acceptional => exceptional`).

Matching rules (all unique-match-only): exact case-insensitive phrases at
every term length including aliases; single-word terms of four or more
characters accept a distance-one near miss when the candidate is not an
ordinary English word; term-boundary garbles are split
('contractholders' -> 'contract holders') or joined ('its av' -> 'ITCAV')
against the same concatenated term, distance-one joins restricted to
same-length spans and refused when every window token is an ordinary word;
runs of three or more identical single-character tokens (the measured
`c c c c c c c c c cuture` hallucination) collapse. All-lowercase canonicals
preserve source casing; acronyms are emitted as spelled. Corrections may
change a segment's word count (join/split/collapse) but never segment count,
order or timing — the word-count promise in the module docstring was
revisited deliberately for the measured error classes (card t_72405565 AC5).

The worker runs this stage ON by default; `SPELL_CHECK` in
`{0,false,no,off}` opts out (shared `spell_check_enabled()` rule — the old
API gate on exact `1` and the two-way default mismatch are retired;
card t_72405565).

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
`"suggested"`. A plausibility validator filters phrases that look like names but are not:
self-intro captures use a two-tier ASCII name grammar (issue #1 option (b)) —
the lead phrase matches case-insensitively on every phrase; the ambiguous
phrases (`I'm X`, `I am X`, `X here`) require a capitalized capture, which is
how "I'm Shelley Festing" works but also how ASR case-restores prose decoys
("I'm sorry" arrives as "I'm Sorry"), so capitalized prose reaches the
validator rather than the capture; the
unambiguous phrases (`my name is X`, `this is X`) tolerate lowercase ASR
output ("my name is jane wu" yields `jane wu`), but a lowercase-dependent
capture must have the two-word given+surname shape, so "my name is wrong"
and "this is strictly about ability one" never surface. Trailing function
words shorten the capture from the tail (one fewer word, down to a
single-word floor) instead of discarding the whole match, on both tiers;
prose that trims down to a function word logs `non-name function words`.
The configurable false-positive blocklist
then rejects phrases that look like names but are not. Finally, a
conservative conversational-word set (`_CONVERSATIONAL_WORDS`, card
t_ac7889e4) rejects any candidate containing a conversational word at any
position — the single word ("I'm Sorry" -> `Sorry`), the surname ("Smith
Sorry"), or the given word ("Sorry Smith") — on every tier, because
capitalization alone cannot separate case-restored prose after "I'm"/"I am"/
"this is" from a name; the set deliberately excludes plausible surnames (May,
Young, Cook, Rich), so real introductions keep resolving. Rejections log
`contains conversational word(s)`. Cross-label
first-name collisions are deduplicated: only the highest-confidence label
keeps the name.

The E4 LLM review pass is wired as the pure `extra_evidence` seam, not as an
environment switch (see CONFIG.md, "v2 seams"), so model output never enters
the service's request context implicitly.

## Opt-in LLM cleanup pass (`pipeline/llm_cleanup.py`, V2)

Rewrites **only spelling, punctuation, and capitalization** across the
transcript, under a mandatory mechanical guard. Interface discipline is the
same as the `extra_evidence` seam above: the model client is a
caller-injected `prompt -> response` callable
(`run_llm_cleanup(segments, llm=...)`), the module contains no HTTP client,
and no environment toggle hardwires a network call — the pass is off unless
an embedding caller injects a client.

- **Chunking** (`plan_cleanup_chunks`): whole speaker-turn lines are packed
  to a character budget; a single oversized turn gets its own chunk instead
  of being split. Every chunk after the first carries the previous chunk's
  trailing lines as a read-only seam context, so the boundary text is in
  the prompt (not a bare chunk index) and thoughts are never severed.
- **Preface** (`pipeline/llm_preface.md`, `PREFACE VERSION: 1`): the AGENT
  rules block prepended to every prompt — permitted changes (spelling,
  punctuation, capitalization/formatting) and absolute prohibitions
  (wording, merges/splits, invented content, names/identifiers, speaker
  labels, timestamps, summarization), plus the same-lines/same-order output
  contract so a diff shows only permitted edits.
- **Verification** (mandatory safety net, `verify_chunk_response`): the
  response must have the exact line count, order, and `N|` numbering; the
  `Speaker N [t0-t1] —` prefix must be byte-identical; and each token may
  only change by case/punctuation normalization or a Damerau-Levenshtein
  near-miss (`spelling_similarity_threshold`, default 0.6) — any other
  token difference is a wording change and rejects the whole chunk.
  Rejected/failed chunks keep the original text verbatim.
- **Report**: `CleanupResult.report_markdown` is a per-chunk
  accept/reject/fail diff report (counts, per-changed-line before/after,
  rejection reasons). Through the service seam it is written to
  `llm_cleanup_report.md` in the job dir and summarized via
  `meta["cleanup_summary"]` into a `## LLM transcript cleanup` section of
  the rendered report.
- **Name review (optional, `LLM_CLEANUP_REVIEW_NAMES=1`)**: the prompt
  invites `Label | Full Name | quoted sentence` evidence after an
  `--- NAME EVIDENCE ---` marker (advisory only; transcript lines and their
  labels/timestamps stay immutable). Entries survive only from ACCEPTED
  chunks and are returned as `Evidence`-shaped mappings
  (`source: "llm_cleanup"`) for the existing
  `suggest_speaker_names(..., extra_evidence=[...])` seam — callers supply
  the weight, as with any injected evidence.
- **Settings**: `CleanupSettings` (frozen dataclass) — chunk budget, seam
  overlap, per-attempt timeout, retries, model/endpoint description string.
  Generation `temperature` is pinned to `0.0` by assertion (no env
  override exists), because the verification guard assumes a deterministic
  pass.

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
