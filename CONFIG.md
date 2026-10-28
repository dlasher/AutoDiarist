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
| `DIAR_METHOD` | `sherpa` | Diarization method passed as `--diarize-method`. Accepted: `sherpa` (default since card t_5a287979 — external sherpa-onnx subprocess with segmentation + embedding + clustering; the engine's documented reliable path), `ecapa` (same subprocess), `foxnose` (fast global pass; collapses 5-speaker panel audio to 1 label via its BIC single-speaker veto — see README), `sortformer`, `none`. Anything else raises `CrispasrConfigError` before the job runs. `sortformer` adds `--diarize-model auto`; `foxnose`/`ecapa`/`sherpa` add `--diarize-embedder auto`; `sherpa`/`ecapa` additionally add `--sherpa-segment-model`/`--sherpa-embedding-model` (and `--sherpa-bin` when set) and REQUIRE the two ONNX models on disk — their absence is a startup `CrispasrConfigError`, never a silent single-speaker transcript (AC4). `none` omits all diarization flags. |
| `SHERPA_SEGMENT_MODEL` | `/opt/models/sherpa-onnx-reverb-diarization-v1.onnx` | Segmentation ONNX for `sherpa`/`ecapa` (baked in the image; README "Diarization models" pins the release-asset URL + sha256). |
| `SHERPA_EMBEDDING_MODEL` | `/opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | Speaker-embedding ONNX for `sherpa`/`ecapa` (same provenance). |
| `SHERPA_BIN` | `/opt/sherpa/sherpa-diarization-wrapper` | Executable the engine shells out to for `sherpa`/`ecapa`. Shipped as a wrapper that injects `--clustering.cluster-threshold` (the engine has no flag for it); validated executable-or-startup-error. Empty = PATH lookup of `sherpa-onnx-offline-speaker-diarization`. |
| `SHERPA_CLUSTER_THRESHOLD` | `0.40` | Read by the wrapper at run time. sherpa-onnx FastClustering distance threshold: LOWER resolves more speakers, HIGHER merges. NV-host calibration table in README "Diarization models". |
| `DIARIZATION_MAX_SPEAKERS` | `8` | Card t_38e52d89. Ceiling on distinct diarization speaker labels, enforced by the wrapper's CONDITIONAL second clustering pass: the auto pass runs first and is only rewritten when its label count exceeds the ceiling (the engine's own `--clustering.num-clusters=0` is stripped before injection, else the setting is a silent no-op). Compliant audio replays byte-identical; a bare fixed count over-splits (panel 2 -> 4 at NC=8). DIARIZATION_MAX_SPEAKERS=0 disables (pre-card behaviour). See `autodiarist-speaker-count-ceiling.md`. |
| `SHERPA_NUM_THREADS` | empty = AUTO | Card t_3cb8f97c. Threads the wrapper passes as BOTH `--segmentation.num-threads` and `--embedding.num-threads` on every sherpa invocation (including the ceiling re-run pass). The CLI default is 1; diarization was ~80% of a job's wall time running on a single core. AUTO = `min(max(1, cores // MAX_PROCESSING_GLOBAL), SHERPA_MAX_THREADS)` where `cores` is the container-visible core count (nproc, else getconf; the containers see ALL host cores — no compose file sets CPU limits). Unset/`0`/negative/non-integer all mean AUTO; a valid explicit count is passed verbatim, clamped only down to the core count. The resolved value (and why) is logged to the job stderr AND appended to `$JOBS_DIR/sherpa-threads.log` — the service's engine-stderr temp file is deleted on success, so the log file is the record ops can actually read after a healthy run. Full mechanics and the measured curve: section "Diarization threading" below. |
| `SHERPA_MAX_THREADS` | `4` | Cards t_3cb8f97c + t_71a5768f. Ceiling on the AUTO value only (explicit `SHERPA_NUM_THREADS` is not capped by it; it is always `min(cap, cores)`). Invalid values (0/negative/non-integer) fall back to 4. 4 is the MEASURED optimum on both shipped hosts — the AMD sweep (2026-10-26) corrected the previously derived default of 8; on either shipped host raising this knob buys nothing (AMD plateau), so it exists for boxes whose thread curve has a different shape. |
| `CRISPASR_GPU_BACKEND` | `cuda` | Pinned via `--gpu-backend`. Accepted: `cuda`, `cpu`, `metal`, `vulkan`. The adapter probes `crispasr --version` once per binary per process and fails fast (typed `CrispasrError`) when this value is missing from the build's `ggml backends` list — this is the guard against the silent CPU fallback (see TROUBLESHOOTING.md). Setting it to `cpu` skips the assertion. Change only to force CPU debugging. |
| `CRISPASR_LANGUAGE` | `en-US` | `-l` language qualifier. SPEC 08 fixes v1 to `en-US`; override is only for local per-locale experimentation. |
| `CRISPASR_CACHE_DIR` | unset | Passed as `--cache-dir`. Holds the `--diarize-embedder auto` GGUF embedders (wespeaker/titanet) and any auto-downloaded artifacts; a named volume in compose. |
| `CRISPASR_LIB_DIR` | unset | Directory containing `libcudart.so.13` / `libcublas.so.13` (image: `/usr/local/cuda-13.0/targets/x86_64-linux/lib`, apt `cuda-cudart-13-0` + `libcublas-13-0`). Prepended to the subprocess `LD_LIBRARY_PATH`. Without it the engine silently drops to CPU. |
| `MIN_FRAGMENT_SECONDS` | `2.0` | Sub-second fragment merge threshold (spike: ~94% of sub-2 s fragments mis-cluster). Fragments shorter than this join the previous turn. `0` disables. |
| `VRAM_LIMIT_MIB` | `3584` (3.5 GiB) | Per-job GPU-allocated ceiling enforced by the sampler thread. Breaching it kills the engine process and fails the job with `TranscriptionOOM`. `0` disables the guard. The `nvidia-smi` sampler reads the CARD TOTAL, so under global concurrency (`MAX_PROCESSING_GLOBAL=N`) the effective ceiling is the FLEET cap `VRAM_LIMIT_MIB x N` (card t_dbad2235 planner note): one job is never failed for its siblings' memory, and the per-job meaning of this knob is preserved. On the AMD image there is no `nvidia-smi` at all — the sampler self-disables (`VRAM sampling disabled`) and this cap is inert, at any N. |
| `CRISPASR_TIMEOUT_SECONDS` | `1800` | Wall-clock budget per engine invocation; overrun kills the process and fails the job. |
| `CRISPASR_VRAM_SAMPLE_SECONDS` | `1.0` | `nvidia-smi` sampler period (floor 0.1 s). |
| `MOCK_PIPELINE` | unset | Exact `1` returns two deterministic synthetic segments from `transcribe_and_diarize` without touching the binary. Also read by `api.main` (see below). |

## HTTP API and queue — `api/main.py`

| Variable | Default | Effect and validation |
| --- | --- | --- |
| `JOBS_DIR` | `/tmp/autodiarist_jobs` | Job-state root. Created at import time; the compose file sets `/data/jobs` on a named volume. Holds `<token>/status.json` plus the decoded/transcribed artifacts. |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MiB) | Upload cap enforced while streaming; oversize uploads get `413` and leave no job dir. Parsed `int` at request time. |
| `MOCK_PIPELINE` | unset | Exact `1` makes the worker short-circuit decode/ASR and write a deterministic mock result; admission, limits, cancellation, and download semantics are unchanged. |
| `MOCK_TRANSCRIBE_SECONDS` | `0` | Mock-stage parking knob (card t_fef54150): with `MOCK_PIPELINE=1` the transcribing stage holds for this many seconds while polling the cancel flag, so mid-run cancellation is testable GPU-free. Non-numeric or negative values behave as `0`. Production (non-mock) runs ignore it entirely. |
| `SPELL_CHECK` | unset (enabled) | Spell-check layer ON by default; the API seam and `pipeline.spellcheck` share `spell_check_enabled()` (`pipeline/spellcheck.py`), so unset → enabled, and `{0,false,no,off}` (case-insensitive) disables. The old API-only hard gate (`== "1"`) that meant the shipped service never ran the pass is retired (card t_72405565). Set `SPELL_CHECK=0` or `off` to opt out. |
| `QUEUE_WORKER_ENABLED` | `1` | Exact `1` starts the in-process queue worker POOL (fair-share round-robin across sessions; pool size = `MAX_PROCESSING_GLOBAL`, default 1 — serial) at startup; `0` admits jobs into `queued` and holds them (ops valve / tests). Jobs admitted while disabled are still claimed if the flag is flipped on later. |
| `MAX_QUEUE_DEPTH` | `50` | System backpressure. At or beyond this many `queued` jobs, `POST /v1/jobs` answers `503` + `Retry-After` — the **only admission rejection** since V1.5 (card t_82bfbc15); the detail names both the depth and the Retry-After value. Accepted jobs are always queued, whatever the caller's session holds. |
| `MAX_JOBS_PER_SESSION` | *(deprecated)* | **No effect** since V1.5 (card t_82bfbc15): a session may hold any number of active jobs. The variable is still accepted so existing `.env` files load unchanged; when set, startup logs `config: MAX_JOBS_PER_SESSION=... is DEPRECATED and IGNORED` and the value is omitted from the boot config summary. Remove it from your env. |
| `MAX_PROCESSING_PER_SESSION` | `1` | Fair-share scheduling cap (V1.5, card t_4b40acfd): maximum jobs of ONE session that may be `processing` concurrently. Enforced by the scheduler at claim time — a session at the cap is skipped even when its queued jobs sit at the head of the queue, so the cap survives a future multi-worker scale-up. Read at every claim; values `< 1` or unparseable log a warning and clamp to 1 (a zero cap would freeze the queue). Sessions are served round-robin (FIFO within a session) using a persistent pointer in `JOBS_DIR/_scheduler.json`; `queue_position` reports the rank in that fair execution order. |
| `MAX_PROCESSING_GLOBAL` | `1` | Global concurrency (V1.5.1, card t_dbad2235): number of queue workers processing jobs SIMULTANEOUSLY. The loop was serial by construction (one task awaiting each job to completion); this spawns N workers running the same claim/process/wake cycle. The claim stays atomic under `_queue_lock` (N workers cannot double-claim) and the cap is enforced AT CLAIM TIME — `_select_next_queued` blocks every claim while total in-flight ≥ N, so the service never starts a job it cannot run. Read once at startup (pool size + dedicated per-job executor width) and again at every claim: lowering it acts immediately, raising it above the live worker count needs a restart. Unset/invalid/`< 1` warn and clamp to 1 — never freeze the queue. Default 1 keeps existing deployments byte-for-byte unchanged; opt in per host from MEASUREMENT (see below). Under N>1 the VRAM guard becomes a FLEET cap (`VRAM_LIMIT_MIB` × N — the `nvidia-smi` sampler reads the card total); fair-share selection and the per-session cap above are untouched at any N. `GET /v1/jobs` reports the live `processing_count`. |
| `RETRY_AFTER_SECONDS` | `30` | Value of the `Retry-After` header on `503` admissions (repeated in the detail body). |
| `PROCESSING_SECONDS_PER_AUDIO_SECOND` | `0.185` | Queue-aware wait-estimate rate (V1.5, card t_b3ae9366): estimated GPU seconds per second of audio. The default is **hardware-specific**, measured on the T400 4 GB integration rig: a 39-minute meeting (2340 s audio) ran 423–434 s wall on CUDA — 434/2340 ≈ 0.185. **Re-derive it on your own GPU** with `docker exec autodiarist-api ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 <job>/audio_source.wav` for a representative recording, time the job from `started_at` to `done` in its `status.json`, and set the quotient (wall_s ÷ audio_s); the CPU fallback path runs ~11x slower, so a CPU box needs roughly an order of magnitude larger value. Exact `0` disables `estimated_wait_seconds`/`estimated_completion_at` emission entirely — queue side and running side alike. Negative or non-numeric values log a boot warning and fall back to `0.185`. Estimates are linear-model ESTIMATES — never guarantees (diarization variance and chunk overhead are not modeled). The rate constant is the QUEUE-side model only; the running-phase ETA is derived from each job's own observed rate and this value contributes to it only as the emission kill-switch and the startup-cost gate (README "Running-phase ETA"). |
| `AUDIO_DURATION_FALLBACK_SECONDS` | `300` | Duration substituted into the ETA prefix sum for a queued job AHEAD whose admission ffprobe failed (`duration_seconds: null`) — one unprobeable file must not poison the estimates behind it. The polled job itself still shows no ETA when its own duration is unknown; the API never fabricates a duration at request time. |
| `RUNNING_ETA_MIN_PROGRESS` | `0.30` | Progress fraction a `processing` job must genuinely cross before the running-phase ETA emits (card t_d64bdf29). The estimate is derived ONLY from the job's own observed rate (elapsed since `attempt_started_at` ÷ published `progress`); this gate keeps warm-up out of that observation window — the default is the transcribing band's lower edge. Non-numeric or values outside `(0, 1)` warn and fall back to the default. Unknown-duration jobs are gated on this alone; known-duration jobs must ALSO outlive the startup-cost floor (duration × `PROCESSING_SECONDS_PER_AUDIO_SECOND` × 0.30) before emission. |
| `RUNNING_ETA_REANCHOR_SECONDS` | `120` | Freshness allowance for the running-phase ETA while `progress` freezes (stalled stage, or a band publishing only at its edges, e.g. spellcheck/llm_cleanup): the last valid value is held at most this long without a real advance, then goes `null` until progress moves again — fresh or null, never stale. A held value additionally nulls as soon as it passes its own promised completion time. Non-positive/non-numeric warn and fall back to the default. Published values never move upward between polls (ratchet); a reclaimed attempt re-derives from its own clock with fresh memory. |
| `QUEUE_POLL_SECONDS` | `0.2` | Worker idle-poll period (float; invalid values fall back to the default). |
| `SESSION_HEADER` | `X-Session` | Header carrying a client-supplied session identity; highest precedence in the identity chain. `Authorization` remains honored when set here explicitly. With no header, the server-issued `autodiarist_session` cookie (returned on `POST /v1/jobs` body + Set-Cookie) identifies the session; then client IP; then a shared `anonymous` bucket. All identities persist as SHA-256 digests (first 32 hex chars, domain-separated); raw values never hit disk. The issued id is additionally hashed under the job-state key `session_ident`. |
| `SESSION_COOKIE_NAME` | `autodiarist_session` | Name of the server-issued session cookie. Overridable for proxies that reserve the default name; an invalid (non-HTTP-token) value warns at boot and falls back to the default. Effective values of both session knobs, `MAX_QUEUE_DEPTH`, `MAX_PROCESSING_PER_SESSION`, `RETENTION_DAYS` and `PROCESSING_SECONDS_PER_AUDIO_SECOND` are validated and printed once at boot (`config: session header=... cookie=... max_queue_depth=... max_processing_per_session=... max_processing_global=... retention_days=... eta_rate=... eta=on|OFF`); a set-but-deprecated `MAX_JOBS_PER_SESSION` warns that it is ignored instead. |

## Global concurrency — `MAX_PROCESSING_GLOBAL` (V1.5.1, card t_dbad2235)

Until V1.5.1 the queue was serial BY CONSTRUCTION: one worker task awaited
each job to completion before claiming the next. `MAX_PROCESSING_GLOBAL=N`
spawns N workers over the same atomic claim; the cap is additionally checked
AT CLAIM TIME, so the service never starts a job it cannot run. Default 1 —
byte-for-byte today's behaviour until an operator opts in.

| Run (137.5 s clip, 3 waves, NVIDIA T400 4 GiB) | wall / wave | throughput | engines alive |
| --- | --- | --- | --- |
| N=1 (serial) | 72.7 s | 0.014 jobs/s | 1/1 |
| N=2 | 82.8 s | 0.024 jobs/s | 2/2 |
| N=3 | 94.1 s | 0.032 jobs/s | 3/3 at 3207 MiB of 4096 |
| N=4 | — | — | 3/4 every wave: the 4th aborts with `nemotron: sched alloc encoder graph failed`, the other 3 SURVIVE |

N=3 is the measured operating point on the T400 (the card ceiling is 4096
MiB; the per-job peak is ~1069 MiB). **The AMD RX 6600 value must come from
a MEASUREMENT of the same shape, never from these NVIDIA numbers** — until
then its shipped default stays 1.

Failure mode under oversubscription: a hard child-process abort, not a
Python exception. The service marks only that job `failed` with the real
stderr tail and every worker keeps serving. The VRAM guard scales with the
fleet: the `nvidia-smi` sampler reads the CARD TOTAL, so with N workers each
job is checked against `VRAM_LIMIT_MIB × N` (`VRAM_LIMIT_MIB` itself stays
the documented per-job share). On the AMD image the sampler has no
`nvidia-smi` and `VRAM_LIMIT_MIB` is inert at any N.

Operational notes:

* The knob must be interpolated inside the compose `environment:` block
  (`MAX_PROCESSING_GLOBAL: ${MAX_PROCESSING_GLOBAL:-1}`) — Docker Compose
  does NOT pass arbitrary host env vars into a container, so
  `MAX_PROCESSING_GLOBAL=3 docker compose up -d` only works because that
  line ships in both profiles.
* Lowering the cap at runtime acts on the next claim; raising it above the
  live worker count requires a restart (the pool and the dedicated per-job
  executor are sized once at boot; the executor is NOT asyncio's default
  `min(32, cpu+4)` pool, so the cap is never silently under-delivered by
  thread starvation).
* Fair-share round-robin and `MAX_PROCESSING_PER_SESSION` are unchanged at
  any N: 3 workers with 3 jobs of one session and cap 1 still run them one
  at a time, while three different sessions are each served immediately.
* Boot log: `concurrency: MAX_PROCESSING_GLOBAL=N — N worker(s) spawned,
  per-job executor width N`, plus the resolved value in the `config:`
  summary; the live in-flight count is on `GET /v1/jobs` as
  `processing_count`.

## Diarization threading — `SHERPA_NUM_THREADS` (cards t_3cb8f97c + t_71a5768f)

The engine invokes the sherpa-onnx diarization CLI through the
`SHERPA_BIN` wrapper (the same wrapper that injects the clustering
threshold and enforces the speaker ceiling). The CLI runs BOTH the
segmentation and the embedding network on ONE thread by default, and
diarization is ~80% of a job's wall time (measured 72.3 s job: ASR 14.0 s
+ sherpa 58.3 s) while the GPU idles through the whole phase.

The wrapper now passes `--segmentation.num-threads=T` and
`--embedding.num-threads=T` on every sherpa invocation: the
ceiling-disabled exec path, pass 1, AND the ceiling re-run pass (the
second clustering pass starts while the first pass's memory is still
resident, which is why the auto cap stays conservative).

**Measured curves — BOTH shipped hosts** (isolated sherpa stage, operator
sweeps 2026-10-26; speakers detected constant at EVERY thread count on
both — threading is a performance knob, not a quality knob). Raw walls
are NOT comparable across hosts (different clips), so each table also
gives s of diarization wall per s of audio.

AMD host — 12 cores, 249 s panel clip. Above 4 threads the curve is a
PLATEAU: no oversubscription on this box, and the work appears
MEMORY-BANDWIDTH bound — 4 threads already saturate it, so more threads
neither help (each row within ~12% of the optimum) nor hurt.

| threads | sherpa wall | vs 1 thread | s wall / s audio | speakers |
|---|---|---|---|---|
| 1 | 52.8 s | 1.00x | 0.212 | 5 |
| 2 | 30.8 s | 1.72x | 0.124 | 5 |
| **4** | **19.8 s** | **2.66x** | **0.080** | 5 | ← OPTIMUM
| 6 | 21.0 s | 2.52x | 0.084 | 5 |
| 8 | 21.2 s | 2.49x | 0.085 | 5 |
| 12 | 22.2 s | 2.38x | 0.089 | 5 |

NVIDIA T400 host — 4 cores, 137.5 s clip. Above 4 threads the curve is a
CLIFF: 6 threads on 4 cores OVERSUBSCRIBES — 8 threads is worse than one
and 12 threads is half the one-thread throughput.

| threads | sherpa wall | vs 1 thread | s wall / s audio | speakers |
|---|---|---|---|---|
| 1 | 58.3 s | 1.00x | 0.424 | 4 |
| 2 | 32.3 s | 1.81x | 0.235 | 4 |
| **4** | **20.4 s** | **2.85x** | **0.148** | 4 | ← OPTIMUM
| 6 | 53.4 s | 1.09x | 0.388 | 4 |
| 8 | 69.3 s | 0.84x — WORSE than 1 | 0.504 | 4 |
| 12 | 110.4 s | 0.53x | 0.803 | 4 |

**4 threads is the measured optimum on BOTH hosts.** On a 4-core box
"optimum == core count" and "optimum == a constant 4" make the SAME
prediction, so the NVIDIA sweep alone never discriminated; the 12-core
AMD sweep did, and the constant won — the default cap is now 4 (card
t_71a5768f), not the earlier derived 8.

Live on the same host after the change (service path, same 137.45 s clip
at threshold 0.40, raw CLI segment lines byte-identical at 1/2/4):
OLD build engine stage 80.6 s / job wall 85.4 s -> NEW build auto T=4
41.9 s / 45.3 s (output payload identical modulo the per-job token);
host override `SHERPA_NUM_THREADS=2` through compose -> 53.7 s and
`threads=2 (explicit: cores=4 workers=1 cap=4 requested=2 max=unset)`
recorded in `$JOBS_DIR/sherpa-threads.log` — knobs honored end-to-end.

The NVIDIA CLIFF above 4 threads (6+ threads oversubscribe a 4-core box:
8 threads there is WORSE than one) and the AMD PLATEAU above 4 (memory-
bandwidth bound; extra threads buy nothing) are why the constant default
is 4 — not 8, which would make the 4-core host ~2x slower than today and
the 12-core host 7% slower than the optimum. The auto default is computed
per invocation:

```
T = min( max(1, floor(C / N)), min(SHERPA_MAX_THREADS, C) )
    C = container core count (nproc, else getconf _NPROCESSORS_ONLN).
        Neither compose file sets CPU limits (deploy.resources carries GPU
        reservations only), so containers see ALL host cores: 4 on NVIDIA,
        12 on AMD. Cores unreportable -> C unknown -> T = 1 (== today).
    N = MAX_PROCESSING_GLOBAL (card t_dbad2235). N workers x T threads each
        is the TOTAL demand; C // N keeps it <= C by construction.
        Unset / non-integer / < 1 counts as 1 (same parse rule as
        api/main.py — the queue must never freeze on a bad value).
```

| host | C | N=1 (shipped) | N=2 | N=3 | N=4 |
|---|---|---|---|---|---|
| NVIDIA T400 | 4 | 4 (full 2.85x) | 2 | 1 | 1 |
| AMD (RX 6600) | 12 | 4 (measured cap; was 8) | 4 (cap trims 6) | 4 | 3 |

Two properties the card fixes as requirements:

* **Regression floor:** the default is NEVER worse than the old
  single-thread behaviour — at any N, T >= 1 and N x T <= C, so a job
  never oversubscribes the box. A default that sped up N=1 but slowed
  N=3 would be a net loss and does not ship.
* **Deterministic and logged:** the auto value is announced once per sherpa
  call — on the job stderr (`sherpa-wrapper: diarization threads=T (auto:
  cores=C workers=N cap=...)`) and appended to `$JOBS_DIR/sherpa-threads.log`
  (the engine's stderr temp file is deleted on successful runs, so the log
  file inside the jobs volume — next to the pre-existing `_scheduler.json` —
  is the durable per-box record of what resolved, including explicit
  overrides). Ops never need a rebuild to check it.

Overrides (no rebuild for either):

* `SHERPA_NUM_THREADS=<n>` — explicit count, wins over the auto formula
  and ignores `MAX_PROCESSING_GLOBAL`; clamped only down to the core
  count (explicit `9999` on 4 cores -> 4) and never below 1. Unset/`0`/
  negative/non-integer = auto.
* `SHERPA_MAX_THREADS=<n>` — raises/lowers the AUTO ceiling only (default
  4, always `min(cap, C)`; invalid -> 4). The AMD sweep landed (card
  t_71a5768f): 4 is the measured optimum on BOTH shipped hosts, the AMD
  curve PLATEAUS above it and the NVIDIA curve CLIFFS, so set NOTHING on
  either box — this knob is for hosts whose thread curve has a different
  shape (set it via compose: `SHERPA_MAX_THREADS=6 docker compose up -d`).
* Both work the same on a RUNNING container: `docker compose exec -e
  SHERPA_NUM_THREADS=2 autodiarist …` (the knob is read per sherpa
  invocation by the wrapper, so an exec-spawned job honors it without a
  restart; the service-spawned workers honor the compose-level value).

With concurrency the per-job diarization share shrinks (C // N), while
each of the N concurrent jobs still threads within its share — threading
COMPOSES with card t_dbad2235 instead of fighting it.

## Job retention & disk pressure — `api/main.py`

| Variable | Default | Effect |
| --- | --- | --- |
| `RETENTION_DAYS` | `30` | Age expiry for finished jobs. `done`/`failed`/`cancelled` dirs whose `status.json` mtime exceeds the window are deleted; the sweep runs synchronously at service boot and hourly after that, and every removal logs the token with `reason=age-expiry`. `0` disables age-based expiry entirely. The live pair `queued`/`processing` is never age-swept (requeued jobs must survive restarts). |
| `DISK_MIN_FREE_MIB` | `8192` | Disk-pressure watermark on the filesystem backing `JOBS_DIR` (`shutil.disk_usage`). While free space is below it, job dirs are evicted oldest-mtime-first until the watermark is recovered. A `processing` job is never evicted; `queued` jobs are evictable (the next POST admission reuses their `seq` slots). `0` disables eviction. The mounted CrispASR cache volume is never touched. |
| `DISK_CHECK_INTERVAL_SECONDS` | `300` | Interval between disk-pressure checks; the check also runs once at boot. |

## Transcript cache — two controls (card t_ac748379)

The cache serves a finished transcript to a client that re-uploads audio it
has already transcribed. Two controls with OPPOSITE polarity govern it, and
this section is the single place their interaction is documented:

- **`X-Cache` request header** (per-request opt-in, house style cf.
  `X-Session`): on `POST /v1/jobs`, values `1`, `true`, `allow`
  (case-insensitive, whitespace-trimmed) turn the cache ON for that request
  — BOTH directions with one flag: LOOKUP-if-present AND
  store-on-completion (a read-only mode is out of scope). Absent or empty
  means OFF and the request path is byte-for-byte the pre-cache behaviour.
  Any other value is a loud `400` naming the accepted list — a malformed
  value is never silently treated as off or on.
- **`CACHE_DISABLED` env kill switch**: default NOT disabled (clients who
  ask may use the cache). When truthy (`1`/`true`/`allow`, same
  normalisation as the header; junk values warn at boot and mean
  not-disabled) the cache is off for every request regardless of `X-Cache`
  — lookups skipped, nothing stored, `cache_hit` false — and the header is
  obeyed-as-absent, never an error. It is read at USE time per request, so
  an A/B test can flip it in-process without a restart.

Effect matrix (`H` = header `X-Cache` on a request; `K` = `CACHE_DISABLED`):

| | `CACHE_DISABLED` unset/false-ish (off) | `CACHE_DISABLED` truthy (on) |
|---|---|---|
| `X-Cache` absent/empty | today's behaviour exactly: full run, no lookup, no store | today's behaviour exactly |
| `X-Cache: 1/true/allow` | LOOKUP first; HIT answers `done` immediately (never queued, never a worker), MISS runs and STORES on completion | header obeyed-as-absent: full run, no lookup, no store, no error |

The cache key is the triple (`audio_sha256`, `config_fp`, `session_ident`):
a transcript is not a pure function of the audio. `config_fp` is
`sha256[:16]` over the output-affecting config (engine selection, resolved
model paths, vocab-file digest, target language, decode sample rate,
dialogue-split setting, diarization enablement + resolved thread budget,
speaker-count ceiling, spell-check state), computed at use time — so
changing one input can never silently serve a stale transcript.
`session_ident` in the key makes a cross-tenant hit structurally
impossible: the cached artifact may contain tenant-specific output (speaker
self-identification names, an opt-in LLM cleanup pass), so one tenant's
transcript is only ever servable to the same tenant. A cross-session cache
would require excluding those stages from the cached artifact and re-running
them per request — explicitly out of scope.

Storage is `JOBS_DIR/cache_index.sqlite` (`cache_entries` with the triple
as PRIMARY KEY plus `source_token`, `created_at`, `hit_count`,
`last_hit_at`); it survives restarts. A hit COPIES the source artifacts into
the new job dir (never symlinks), so the hit record outlives any later
eviction of the source. Cache rows are purged in the same pass as their
source job dir by BOTH the `RETENTION_DAYS` age sweep and
`DISK_MIN_FREE_MIB` eviction — the cache is never a reason to delete or to
keep a job — and a lookup that finds a row whose source is gone self-heals
by deleting the row and running fresh.

| Variable | Default | Effect and validation |
| --- | --- | --- |
| `CACHE_DISABLED` | unset (not disabled) | Transcript-cache kill switch, truthy = cache off service-wide regardless of `X-Cache` (see *Transcript cache — two controls* above). Accepted truthy spellings `1`, `true`, `allow` (case-insensitive, trimmed); a junk value warns at boot and keeps the documented default. Read at use time per request — flipping it needs no restart. |

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
| `SPELL_CHECK` | enabled | Shared with the API seam above via `spell_check_enabled()`: absent → on, otherwise off only for `{0,false,no,off}` (case-insensitive). Both layers now agree (card t_72405565 retired the old exact-`1` API gate). |
| `VOCAB_PATH` | `<repo>/vocab.txt` (absolute; `/app/vocab.txt` in the image) | Domain vocabulary the spell-check layer loads: NASA/SEWP terms (SEWP, NASA, GWAC, OMB, ITCAV, contract holder(s), soup contracts, exceptional) plus measured garble aliases (`sout, soupes => SEWP`, `soubcontracts => soup contracts`, `acceptional => exceptional`). One canonical term per line, `#` comments, optional `garble => Canonical` alias lines; extend by adding a line — the file is resolved by absolute path with no CWD dependence (card t_72405565). Missing or unreadable files warn and degrade to no correction. |

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
