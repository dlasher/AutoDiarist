# Changelog

## Unreleased (card t_48c2efae — V2 opt-in LLM cleanup pass)

- **`pipeline/llm_cleanup.py`** — opt-in, caller-injected cleanup pass over
  the transcript: fixes spelling/punctuation/capitalization only, never
  source content. Chunks whole speaker-turn lines to a character budget
  (oversized turns get their own chunk, never split) and carries a
  read-only seam overlap into each subsequent prompt so chunk boundaries
  contain the previous seam's text, not just indices.
- **Mandatory verification guard**: each chunk response must preserve the
  exact line count, order, `N|` numbering, and byte-identical
  `Speaker N [t0-t1] —` prefixes; token-level checks allow only
  case/punctuation normalization and Damerau-Levenshtein near-misses
  (spelling fixes) — any wording change rejects the whole chunk and the
  original text is kept. Temperature is pinned to `0.0` by assertion.
- **`pipeline/llm_preface.md`** — versioned AGENT rules preface
  (`PREFACE VERSION: 1`) stating exactly what may and may not change.
- **Per-chunk accept/reject diff report**: counts plus before/after for
  every permitted edit and rejection reasons; through the service seam it
  lands in `llm_cleanup_report.md` and the rendered report gains a
  `## LLM transcript cleanup` summary section (`meta['cleanup_summary']`).
- **Service seam (AC4)**: `process_job_background(..., llm=...,
  cleanup_settings=...)` — the HTTP/queue path never injects a client; the
  module performs no network I/O; no env toggle hardwires a call
  (`SELF_ID_LLM`/`extra_evidence` precedent). `CleanupSettings.from_env()`
  documents budget knobs only (`LLM_CLEANUP_CHUNK_CHARS`,
  `LLM_CLEANUP_OVERLAP_LINES`, `LLM_CLEANUP_TIMEOUT_S`,
  `LLM_CLEANUP_MAX_RETRIES`, `LLM_CLEANUP_MODEL`,
  `LLM_CLEANUP_REVIEW_NAMES`).
- **Optional speaker-name review** (default off): quoted verbatim evidence
  lines are harvested only from ACCEPTED chunks and returned as
  `Evidence`-shaped mappings for the existing
  `suggest_speaker_names(..., extra_evidence=[...])` seam.
- **Tests**: 25 new offline tests (`tests/pipeline/test_llm_cleanup.py`,
  `tests/api/test_llm_cleanup_seam.py`) covering turn-aware chunking,
  seam-overlap prompts, preface rules, spelling-accept/wording-reject,
  label/timestamp immutability, line-count/order enforcement,
  retries/timeout, temperature assertion, evidence→`extra_evidence` flow,
  and an end-to-end rendered-report diff showing only the permitted change
  (suite: 150 passed, no network).

## 1.0.2 - 2026-10-20 (docs final — no code changes)

- CPU-only validation of the pinned `cuda13` v0.8.43 asset (card
  t_18ee2e43, README commit 2bffe0e): with `libggml-cuda.so` quarantined —
  the documented `GGML_BACKEND_DL` fallback — the 60 s spike source ran on
  4 vCPUs at ~1.7x CUDA wall (ASR-only 13.7 s median vs 8 s; diarized
  24.4 s vs 13 s; peak RSS ~1,065 MiB) with 97.7% word parity, 3/3
  sections and identical speaker labels; the full 39-minute fixture took
  142.9 s vs 12.3 s (~11.6x) at 71.2% parity. `nvidia-smi` showed 0
  compute apps / 0 MiB GPU throughout; caveat documented:
  `--gpu-backend cpu` alone still dlopens the CUDA plugin. This
  measurement scopes the older `~11x` estimate in the guard message
  (`pipeline/transcribe.py`): it holds at production-file lengths, not on
  short clips.
- Live fresh-clone replication proof (card t_99d78ff4): the public repo
  (`github.com/dlasher/AutoDiarist`) was cloned at 88dee5d to
  `/root/replication-test` on the T400 VM and `docker compose build` +
  `up` succeeded from the clone alone — both `sha256sum -c` verifications
  passed (model + engine), the backend assertion logged
  `crispasr probe: version-ok backends=['cpu', 'cuda'] pinned=cuda`, and
  2 E2E jobs ran on the 39-minute fixture (252/252 segments, 3 speakers,
  wall 437 s; peak `allocated=1083 MiB / peak=1085 MiB`), zero ERROR
  lines. Host driver 580.178.04 vs image CUDA userland 595.71.05
  compatibility confirmed.
- VM cleanup complete (card t_413e2b7b): ~48 GB reclaimed on the
  validation box (`df -h` free 25 GB -> 73 GB) — homemade
  `nvidia-asr:local` image (18.5 GB), docker build cache (25.0 GB),
  legacy image layers/volumes (4.7 GB), spike/adapter dirs (10.1 GB),
  host `cuda13` apt runtime (~1 GB, image-bundled now), stale job
  artifacts (0.14 GB). All reports/patches were copied first to
  `~/CODE/nvidia-asr-planning/archive/vm-cleanup-2026-10-20/` (sealed
  tarball sha256 `f73eb3d4…`). Deployment survived the purge: post-cleanup
  GPU job `745556b4-…` completed in the kept
  `/root/replication-test` stack; `/opt/nvidia-asr` (131 MB) retained as
  reviewer-approved fallback.

## 1.0.1 - 2026-10-19

- Project rebrand `nvidia-asr` -> `AutoDiarist` (commit 1dc36af).
- Renderer pipeline stamp now reports the actual backend identity
  (card t_3275e056, follow-up to the docs rewrite):
  `autodiarist v1 (CrispASR nemotron Q4_K @<version>)`, with `<version>`
  parsed from the `crispasr --version` probe output and cached alongside
  the backend registration; `@unknown` only until a probe has run.
  `MOCK_PIPELINE=1` renders the deterministic `autodiarist v1 (mock)`.
  The legacy `NeMo Speech @<sha>` label and the `nemo_speech_sha`/
  `nemo_sha` meta keys are retired from `pipeline/renderer.py`; golden
  fixture, `tests/pipeline/test_renderer.py`, and the PIPELINE.md
  "As-built quirk" note updated to the new contract.

## 1.0.0 - 2026-10-19 (post-pivot release)

- Backend pivot: transcription + diarization now runs through the
  **CrispASR subprocess backend** (`crispasr` v0.8.43 CUDA-13 CLI, Nemotron
  3.5 ASR streaming 0.6B Q4_K GGUF, sha256 `22480dd2…`, engine default
  ~30 s internal windowing). The in-process NeMo/torch path and its
  torch-era environment knobs (`DEVICE_DTYPE`, `PYTORCH_*`, `HF_HOME`) are
  retired; no `nemo`/`torch` import remains on the real path.
- Diarization default is `DIAR_METHOD=foxnose` (globally-stable speaker
  IDs); `ecapa`, `sherpa`, `sortformer`, `none` selectable.
- Silent-CPU-fallback guard: the adapter pins `--gpu-backend cuda` and
  asserts backend registration from `crispasr --version` before each
  first job per process; missing CUDA-13 runtime libraries fail jobs fast
  instead of silently running ~11x slower on CPU.
- VRAM discipline: per-stage `nvidia-smi` sampling with `allocated=/peak=`
  INFO lines, `TranscriptionOOM` on breach of `VRAM_LIMIT_MIB` (3584),
  `CRISPASR_TIMEOUT_SECONDS` (1800) wall-clock budget.
- Post-merge pipeline hardening: sub-`MIN_FRAGMENT_SECONDS` (2.0 s)
  fragment merge against 30 s-window mis-clustering, `<en-US>`/NST
  suppression (`--suppress-nst` + parser-side tag stripping), hard failure
  on empty/missing engine output (never a silent fallback transcript).
- Queue hardening (card t_daebf369 lineage): disk-persisted FIFO queue,
  `MAX_QUEUE_DEPTH`/`MAX_JOBS_PER_SESSION` admission (503/429 +
  `Retry-After`), queued-job cancellation with restart revalidation,
  `queue_position` in status.
- Self-identification v2: weighted-evidence suggestions
  (`self_intro` 0.7 / `third_party_intro` 0.5 / `handoff` 0.4, tight/loose
  attribution bonuses, conflict penalty, name-plausibility blocklist,
  cross-label collision dedup) with the `extra_evidence` seam reserved for
  the (default-off) E4 LLM review pass.
- Render contract per SPEC 08 "Render contract — speaker naming"
  (supersedes annotate-only): direct-name transcript body for resolved
  speakers, `## Speaker mapping` table, `## Speaker identification`
  evidence section, raw-segment appendix.
- Language fixed to `en-US` (SPEC 08 "Languages").
- Docker deliverable rebuilt: crispasr prebuilt + GGUF baked by verified
  sha256, apt CUDA-13 runtime libraries, foxnose embedder + job state on
  named volumes, GPU reservation pinned.
- Documentation set (README/CONFIG/PIPELINE/TROUBLESHOOTING) rewritten to
  the shipped post-pivot reality; CONFIG.md cross-checked line-by-line
  against every environment lookup in `api/main.py` and `pipeline/*.py`.

### QA traceability

Accepted by real end-to-end verification (kanban task t_c17fd4fb,
"QA verification: real E2E run of /root/audio.wav on VM, no mock",
verdict PASS): job token **`decb8249-bfd4-43ee-a7c9-b9ac05ed02a9`** —
`POST /v1/jobs` 202, polling reached `done` on `MOCK_PIPELINE=0`,
downloaded Markdown contains 107 segments of genuine transcribed speech
(zero "Mock transcript"/"fallback segment" occurrences), metadata duration
2340.01 s (~39 min) matching the source, speaker attribution and the raw
diarizer appendix verified. Evidence comment retained on the QA card.

## 0.x (pre-pivot development, 2026-10-16)

- Added the HTTP job API, deterministic mock-pipeline integration path,
  and pipeline helper modules for chunking/reconciliation, vocabulary-only
  spelling correction, advisory speaker self-identification, and Markdown
  rendering (NeMo/torch in-process experiments superseded by the CrispASR
  backend above; the spike report carries the pivot evidence, card t_7e17589e).
