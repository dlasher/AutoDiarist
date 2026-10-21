# Configuration Reference

Every environment variable the service reads, grouped by owning module. The
tables below were produced by cross-checking each `os.environ` lookup in
`api/main.py` and `pipeline/*.py` (the settings surface; there is no separate
settings object) against this document. Read at request/boot time unless
noted "import time". `.env.example` in the repository root mirrors these with
GPU-default values.

## Engine contract — `pipeline/transcribe.py` (CrispASR adapter)

| Variable | Default | Effect and validation |
| --- | --- | --- |
| `CRISPASR_BIN` | unset (falls back to bare `crispasr` on `PATH`) | Path to the engine CLI. In the image this is `/opt/crispasr/crispasr` (v0.8.43 CUDA prebuilt). |
| `GGUF_MODEL` | unset | Path to the Nemotron 3.5 ASR streaming 0.6B Q4_K GGUF; the image bakes it at `/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf` (sha256 `22480dd2…`, see Dockerfile). Set **empty** to switch the adapter to `-m auto --auto-download`, letting the engine fetch the canonical `cstr/nemotron-3.5-asr-streaming-0.6b-GGUF` file into `CRISPASR_CACHE_DIR`. |
| `DIAR_METHOD` | `foxnose` | Diarization method passed as `--diarize-method`. Accepted: `foxnose` (default, globally-stable speaker IDs), `ecapa`, `sherpa`, `sortformer`, `none`. Anything else raises `CrispasrConfigError` before the job runs. `sortformer` adds `--diarize-model auto`; `foxnose`/`ecapa`/`sherpa` add `--diarize-embedder auto`; `none` omits all diarization flags. |
| `CRISPASR_GPU_BACKEND` | `cuda` | Pinned via `--gpu-backend`. Accepted: `cuda`, `cpu`, `metal`, `vulkan`. The adapter probes `crispasr --version` once per binary per process and fails fast (typed `CrispasrError`) when this value is missing from the build's `ggml backends` list — this is the guard against the silent CPU fallback (see TROUBLESHOOTING.md). Setting it to `cpu` skips the assertion. Change only to force CPU debugging. |
| `CRISPASR_LANGUAGE` | `en-US` | `-l` language qualifier. SPEC 08 fixes v1 to `en-US`; override is only for local per-locale experimentation. |
| `CRISPASR_CACHE_DIR` | unset | Passed as `--cache-dir`. Holds the foxnose embedder (`--diarize-embedder auto`) and any auto-downloaded artifacts; a named volume in compose. |
| `CRISPASR_LIB_DIR` | unset | Directory containing `libcudart.so.13` / `libcublas.so.13` (image: `/usr/local/cuda-13.0/targets/x86_64-linux/lib`, apt `cuda-cudart-13-0` + `libcublas-13-0`). Prepended to the subprocess `LD_LIBRARY_PATH`. Without it the engine silently drops to CPU. |
| `MIN_FRAGMENT_SECONDS` | `2.0` | Sub-second fragment merge threshold (spike: ~94% of sub-2 s fragments mis-cluster). Fragments shorter than this join the previous turn. `0` disables. |
| `VRAM_LIMIT_MIB` | `3584` (3.5 GiB) | Per-job GPU-allocated ceiling enforced by the sampler thread. Breaching it kills the engine process and fails the job with `TranscriptionOOM`. `0` disables the guard. |
| `CRISPASR_TIMEOUT_SECONDS` | `1800` | Wall-clock budget per engine invocation; overrun kills the process and fails the job. |
| `CRISPASR_VRAM_SAMPLE_SECONDS` | `1.0` | `nvidia-smi` sampler period (floor 0.1 s). |
| `MOCK_PIPELINE` | unset | Exact `1` returns two deterministic synthetic segments from `transcribe_and_diarize` without touching the binary. Also read by `api.main` (see below). |

## HTTP API and queue — `api/main.py`

| Variable | Default | Effect and validation |
| --- | --- | --- |
| `JOBS_DIR` | `/tmp/autodiarist_jobs` | Job-state root. Created at import time; the compose file sets `/data/jobs` on a named volume. Holds `<token>/status.json` plus the decoded/transcribed artifacts. |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MiB) | Upload cap enforced while streaming; oversize uploads get `413` and leave no job dir. Parsed `int` at request time. |
| `MOCK_PIPELINE` | unset | Exact `1` makes the worker short-circuit decode/ASR and write a deterministic mock result; admission, limits, cancellation, and download semantics are unchanged. |
| `SPELL_CHECK` | unset | Exact `1` enables the spell-check stage in the worker. Note the two-way mismatch documented in PIPELINE.md: the API stage requires `==1`, while `pipeline.spellcheck` on its own defaults enabled unless the value is in `{0,false,no,off}`. |
| `QUEUE_WORKER_ENABLED` | `1` | Exact `1` starts the in-process FIFO worker at startup; `0` admits jobs into `queued` and holds them (ops valve / tests). Jobs admitted while disabled are still claimed if the flag is flipped on later. |
| `MAX_QUEUE_DEPTH` | `50` | System backpressure. At or beyond this many `queued` jobs, `POST /v1/jobs` answers `503` + `Retry-After`. |
| `MAX_JOBS_PER_SESSION` | `2` | Per-session quota over active (`queued` + `processing`) jobs; breach answers `429`. |
| `RETRY_AFTER_SECONDS` | `30` | Value of the `Retry-After` header on `429`/`503` admissions. |
| `QUEUE_POLL_SECONDS` | `0.2` | Worker idle-poll period (float; invalid values fall back to the default). |
| `SESSION_HEADER` | `Authorization` | Header carrying session identity for quota accounting. Missing header falls back to client IP, then to a shared `anonymous` bucket. Identities are stored as SHA-256 digests (first 32 hex chars), never raw header values. |

## Job retention & disk pressure — `api/main.py`

| Variable | Default | Effect |
| --- | --- | --- |
| `RETENTION_DAYS` | `30` | Age expiry for finished jobs. `done`/`failed`/`cancelled` dirs whose `status.json` mtime exceeds the window are deleted; the sweep runs synchronously at service boot and hourly after that, and every removal logs the token with `reason=age-expiry`. `0` disables age-based expiry entirely. The live pair `queued`/`processing` is never age-swept (requeued jobs must survive restarts). |
| `DISK_MIN_FREE_MIB` | `8192` | Disk-pressure watermark on the filesystem backing `JOBS_DIR` (`shutil.disk_usage`). While free space is below it, job dirs are evicted oldest-mtime-first until the watermark is recovered. A `processing` job is never evicted; `queued` jobs are evictable (the next POST admission reuses their `seq` slots). `0` disables eviction. The mounted CrispASR cache volume is never touched. |
| `DISK_CHECK_INTERVAL_SECONDS` | `300` | Interval between disk-pressure checks; the check also runs once at boot. |

## Chunking utility — `pipeline/chunking.py`

Kept for the standalone utility and its tests. The shipped CrispASR backend
runs whole-file in one invocation (the engine does its own ~30 s slicing), so
these no longer bound VRAM.

| Variable | Default | Effect |
| --- | --- | --- |
| `CHUNK_SECONDS` | `600` | Chunk length for `chunk_audio` / `chunk_settings`. `.env.example` ships `180` as a legacy sample value; it has no effect on the CrispASR path. |
| `OVERLAP_SECONDS` | `30` | Overlap between chunks; must be `< CHUNK_SECONDS`. |

## Spell-check — `pipeline/spellcheck.py`

| Variable | Default | Effect |
| --- | --- | --- |
| `SPELL_CHECK` | enabled | Line above shared with the API; read here via `_env_enabled`: absent → on, otherwise off only for `{0,false,no,off}` (case-insensitive). |
| `VOCAB_PATH` | `vocab.txt` | Custom vocabulary file. Missing or unreadable files warn and degrade to no correction. |

## Speaker self-identification — `pipeline/self_identification.py`

| Variable | Default | Effect |
| --- | --- | --- |
| `SELF_ID_RESOLVE_THRESHOLD` | `0.7` | Confidence at or above which the renderer may use the name directly in the transcript body. Also read by `pipeline/renderer.py` as its own fallback gate. |
| `SELF_ID_ANNOTATE_THRESHOLD` | `0.4` | Confidence at or above which a suggestion is reported as `suggested` (evidence section only). |
| `SELF_ID_PATTERNS` | built-ins | Pipe-delimited regex overrides for the self-introduction evidence patterns. Compiled with `re.IGNORECASE` for the lead phrase; every `(?P<name>...)` capture group is auto-wrapped in a case-sensitive scope (`(?-i:...)`, issue #1), so name grammars written as `[A-Z][a-z]+` keep rejecting lowercase prose. |
| `SELF_ID_HANDOFF_PATTERNS` | built-ins | Pipe-delimited overrides for floor-handoff patterns (attributed to the *next different speaker*). |
| `SELF_ID_INTRO_PATTERNS` | built-ins | Pipe-delimited overrides for third-party-introduction patterns. |
| `SELF_ID_BLOCKLIST_PATH` | `self_id_blocklist.txt` | Optional line-per-phrase false-positive blocklist file; missing files warn and are ignored, `#` starts a comment. |
| `SELF_ID_NAME_BLOCKLIST` | unset | Extra blocked phrases as a comma-separated string or JSON array; malformed values fail open with a warning. |

## Renderer — `pipeline/renderer.py`

| Variable | Default | Effect |
| --- | --- | --- |
| `SELF_ID_RESOLVE_THRESHOLD` | `0.7` | Used only for suggestions that carry no explicit `action` field (see PIPELINE.md precedence). |
| `SELF_ID_SECTION` | `1` | Exact `1` emits the `## Speaker identification` evidence section; any other value omits it. |
| `SILENCE_SPLIT_SECONDS` | `1.5` | Same-speaker gap that starts a new transcript paragraph (label rendered once per speaker run). Invalid/non-positive values raise at render time. |

## Reserved / retired (no live reader)

| Name | Status |
| --- | --- |
| `DEVICE_DTYPE` | **Removed with the NeMo pivot.** There is no engine device/dtype selector to configure: the CrispASR path chooses compute via `CRISPASR_GPU_BACKEND`, and the Q4_K GGUF quantization is fixed by the model asset. `tests/test_docker_packaging.py` asserts it is absent from `.env.example`. |
| `PYTORCH_*`, `HF_HOME` | Torch-era knobs retired with the NeMo image; compose assertions keep them out. |

## v2 seams (not configured via environment)

- **`SELF_ID_LLM` (E4 LLM review pass, default off).** Deliberately not an
  env read in v2. External evidence — e.g. an LLM review pass over the
  transcript — is injected through the pure function parameter
  `suggest_speaker_names(segments, extra_evidence=[...])`. Entries are
  `Evidence`-shaped mappings (`label`, `claimed_name`, optional `source`,
  `weight`, `snippet`/`text`, `t0`/`timestamp`) folded into the same weighted
  aggregation; the default weight of an injected entry is `0.0`, so callers
  must supply their own. Injected evidence alone still only reaches
  `resolved` when the weighted total crosses `SELF_ID_RESOLVE_THRESHOLD`.
  Keeping this a caller-supplied seam avoids the API context ever holding
  model output; a future LLM pass must be wired by the service owner, not
  toggled by env.
- **LLM transcript cleanup pass (V2, default off; card t_48c2efae).** The
  cleanup client is injected as a `prompt -> response` callable —
  `pipeline.llm_cleanup.run_llm_cleanup(segments, llm=..., settings=...)`;
  the service seam is the keyword parameter
  `api.main.process_job_background(..., llm=..., cleanup_settings=...)`,
  which the HTTP/queue path never passes. No env toggle hardwires a
  network call; the module ships no HTTP client and the service reads no
  model output. When enabled by an embedding caller, the pass rewrites
  only spelling/punctuation/capitalization inside chunked, preface-guarded
  prompts (`pipeline/llm_preface.md`, "PREFACE VERSION: 1") and mechanically
  verifies every response (exact line count/order/numbers, byte-identical
  speaker-label+timestamp prefixes, token-level spelling-only edits). A
  chunk that violates any invariant is REJECTED and the original text is
  kept; a per-chunk accept/reject diff report is written to
  `llm_cleanup_report.md` in the job dir and summarized in the rendered
  report. Optional speaker-name evidence (`LLM_CLEANUP_REVIEW_NAMES=1`)
  is returned only for accepted chunks and folded into the
  `extra_evidence` seam above. Settings live in the
  `CleanupSettings` dataclass; environment knobs (read only when a caller
  has injected a client):

  | Name | Default | Meaning |
  | --- | --- | --- |
  | `LLM_CLEANUP_CHUNK_CHARS` | `6000` | Authoritative transcript characters per LLM chunk |
  | `LLM_CLEANUP_OVERLAP_LINES` | `1` | Read-only seam lines carried into the next chunk's prompt |
  | `LLM_CLEANUP_TIMEOUT_S` | `60.0` | Per-attempt wall-clock timeout for the injected callable |
  | `LLM_CLEANUP_MAX_RETRIES` | `2` | Extra attempts after a client failure (failed chunk → rejected, original kept) |
  | `LLM_CLEANUP_MODEL` | `` (empty) | Opaque model/endpoint description; report header only, never used to connect |
  | `LLM_CLEANUP_REVIEW_NAMES` | off | Exact `1` enables the optional speaker-name review section |

  Generation temperature is **not configurable**: `CleanupSettings` asserts
  `temperature == 0.0` and raises otherwise — there is deliberately no
  environment variable for it, and the guard is documented with that
  assertion so a future editor cannot loosen it silently.
- E3a/E3b (slide/panelist-strip OCR), E5 (cross-reference inference), and
  hard label remapping of transcript segments are also v2 hooks,
  intentionally unimplemented (audio-only scope, advisory contract).

## Cross-check procedure (AC1)

```bash
# Every environment lookup in the shipped code (direct reads plus the
# _env_str/_env_int/_env_float and _configured_patterns indirections):
grep -rhoE '(environ\.get\(|environ\[|_env_(str|int|float)\(|_configured_patterns\()["'"'"'][A-Z_]+["'"'"']' \
    api pipeline | grep -oE '[A-Z_]{3,}' | sort -u
```

Yields 33 names at original authoring time, 42 as of the V2 cleanup pass and
retention sweeper (the six `LLM_CLEANUP_*` knobs above plus later additions);
each is documented in the tables above (a
value read only as a comparison variable, like the `_env_*` helpers' `name`
parameter, is not a name). Verified with zero undocumented reads;
`RETENTION_DAYS` was long the single name present in `.env.example` without
a code reader — since the V2 retention sweep (card t_5e6200fc) it reads live
in `api/main.py`, together with its two new disk-pressure siblings
(`DISK_MIN_FREE_MIB`, `DISK_CHECK_INTERVAL_SECONDS`), all documented in the
retention table above.
