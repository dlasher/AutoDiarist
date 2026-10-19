# Changelog

## Unreleased

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
