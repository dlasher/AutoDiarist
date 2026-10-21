# AutoDiarist

Asynchronous HTTP job API that takes WAV/MP3/M4A/AAC/OGG/FLAC uploads and
returns a diarized Markdown transcript. Backend per the spike GO verdict
(card t_7e17589e): the `crispasr` CUDA CLI v0.8.43 running the Nemotron 3.5
ASR streaming 0.6B model as a Q4_K GGUF, with foxnose global-speaker
diarization by default. The retired in-process NeMo/torch path is gone; no
`nemo` or `torch` import remains on the real path.

## Architecture

```text
client
  | POST /v1/jobs (multipart audio; extension whitelist wav/mp3/m4a/aac/ogg/flac)
  v
FastAPI + persistent FIFO job queue on disk
  JOBS_DIR/<token>/status.json   atomic writes, monotonic `seq` = FIFO order
  JOBS_DIR/<token>/status_journal.jsonl   append-only publish history (stage audit)
  JOBS_DIR/<token>/audio_source.<ext>
  |
  | single in-process worker (GPU concurrency = 1), QUEUE_WORKER_ENABLED=1
  v
worker stages per job (api/main.py process_job_background):
  decode (ffmpeg -> 16 kHz mono WAV; skipped in mock mode)
  transcribe + diarize (pipeline.transcribe -> crispasr subprocess, whole file)
  spell check          (pipeline.spellcheck, only when SPELL_CHECK=1)
  speaker self-ID      (pipeline.self_identification, weighted evidence)
  render               (pipeline.renderer -> markdown_report.md)
  v
GET    /v1/jobs/{token}          -> status (+ queue_position while queued; stage + monotonic progress)
DELETE /v1/jobs/{token}          -> cancel a queued job (409 while processing)
GET    /v1/jobs/{token}/download -> output_result.json (default) or ?format=md
```

`MOCK_PIPELINE=1` short-circuits the ASR stage with two deterministic
segments end-to-end; everything else (queueing, limits, cancellation, files)
behaves identically, which is what the API integration tests run against.

## HTTP API

Five endpoints, all under `/v1/jobs`. Base URL in the examples below is the
quickstart default (`http://127.0.0.1:8000`). All error responses use the
FastAPI envelope `{"detail": "..."}` unless noted. Every status code here is
implemented in `api/main.py`.

### POST /v1/jobs — submit a job

Multipart form upload; the file field name is **`file`**. The server accepts
the extensions `wav`, `mp3`, `m4a`, `aac`, `ogg`, `flac` (case-insensitive,
from the uploaded filename). On success:

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
# -> 202 {"token": "6a7f…uuid4…", "status": "queued"}
```

Admission is checked in this order, each with its own failure mode:

| Code | When | Detail |
|---|---|---|
| 415 | extension not whitelisted (checked before the upload touches disk) | `Unsupported file type. Supported: wav, mp3, m4a, aac, ogg, flac` |
| 413 | upload exceeds `MAX_UPLOAD_BYTES` (enforced while streaming, not after) | `File exceeds maximum size of <bytes> bytes` |
| 503 | queue at `MAX_QUEUE_DEPTH` (count of `queued` jobs; default 50) | `Queue is full (n/50 jobs queued). Retry later.` + `Retry-After: <RETRY_AFTER_SECONDS>` header (default 30) |
| 429 | the caller's session already has `MAX_JOBS_PER_SESSION` active jobs (default 2; see *Session identity*) | `Session limit reached: 2/2 active jobs per session (queued+processing). Wait for a job to finish or cancel one.` |

Rejected uploads leave no trace on disk: the partially streamed job directory
is deleted before the error is returned.

### GET /v1/jobs/{token} — poll a job

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
```

The response is the job's persisted state. Common to every status: `token`,
`status`, `stage` (see *Progress stages* below), `seq` (FIFO position at
admission), `queued_at`, `progress`, `message`, `filename`, plus the
bookkeeping fields `session` (the hashed session id) and `audio_path`
(server-local path) — treat those as opaque.
`status` is one of `queued`, `processing`, `done`, `failed`, `cancelled`:

- `queued` — adds `queue_position`: the job's **1-based rank among currently
  queued jobs** against the single worker's pickup order; `queue_position: 1`
  means next.
- `processing` — adds `started_at`; `stage` moves through the pipeline in
  order while `progress` advances monotonically and `message` carries the
  stage text (e.g. `Claimed by worker`, `Transcribing pass 2/4`).
- `done` — `stage: "done"`, `progress: 1.0`, `message: "Transcription complete."`.
  The result itself is fetched from the download endpoint at
  `/v1/jobs/{token}/download` (the poll payload does not carry a URL field).
- `failed` — `stage` stays at the boundary where the work broke, `progress`
  holds its monotonic value **below 1.0** (a failure never looks done), plus
  `error` (the exception message from the worker).
- `cancelled` — `message: "Cancelled by client while queued"` plus
  `cancelled_at`; `stage`/`progress` freeze at their last published values.

#### Progress stages

`stage` is the machine-readable pipeline position — a stable enum, in order
of first appearance:

| Stage | Meaning | Progress band |
|-------|---------|---------------|
| `queued` | waiting in the FIFO queue (briefly also when a worker claims the job before decode starts) | 0.00 |
| `decoding` | audio decode / resample to 16 kHz | 0.05 |
| `transcribing` | ASR passes; engine-reported pass/chunk counts refine within the band (e.g. `Transcribing pass 2/4`) | 0.30–0.60 |
| `diarizing` | diarized assembly, likewise refined per reported boundary | 0.60–0.75 |
| `spellcheck` | optional; only runs when `SPELL_CHECK=1` | 0.75–0.85 |
| `self_identification` | speaker-name suggestions | 0.85–0.92 |
| `rendering` | Markdown report + result write-out | 0.92–1.00 |
| `done` | terminal success, `progress == 1.0` | 1.00 |

`progress` semantics: float in `[0, 1]`, **monotonic non-decreasing** across
the whole job, published only at real work boundaries (stage entries and the
engine's per-pass/per-chunk reports) — never from timers. It reaches exactly
**1.0 when and only when `status: "done"`**. Values are ordinal markers of
pipeline position, not percentage-accurate time estimates: a flat stretch
means one long boundary (e.g. a GPU pass), not a stuck job.

A fast stage tail (e.g. spellcheck→done in under a poll interval) can be
missed by a polling client — watch
`JOBS_DIR/<token>/status_journal.jsonl` instead: an append-only record of
every publish (`at`, `status`, `stage`, `progress`, `message`), so the full
ordered stage sequence is always auditable after the run.

Errors: `404` `Job not found` (unknown token), `422` `Invalid token format`
(token must be a canonical UUID4).

### GET /v1/jobs/{token}/download — fetch results

Only for `done` jobs. Default is JSON (`output_result.json` — token, status,
filename, duration, diarized segments); `?format=md` returns the rendered
report (`markdown_report.md`) as `text/markdown`:

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN/download -o result.json
# Content-Type: application/json; attachment; filename="<token>_result.json"
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md" -o report.md
# Content-Type: text/markdown; attachment; filename="<token>_report.md"
```

Errors: `409` `Cannot download results when status is <status>` for any job
that is not `done`; `422` `Unknown format requested. Use format=json or
format=md` for any other `format` value; `404` for unknown tokens (and, as a
consistency guard, `Result file missing` / `Markdown report missing` if a
finished job's artifact is gone from disk).

### DELETE /v1/jobs/{token} — cancel a queued job

```bash
curl -sS -X DELETE http://127.0.0.1:8000/v1/jobs/$TOKEN
# queued -> 202 {"token": "...", "status": "cancelled"}
```

Cancelling a `queued` job frees both queue depth and the session slot; the
source audio is deleted immediately and the state file is kept so `cancelled`
remains observable via GET. Repeat cancels are idempotent (`202`). A job that
is `processing` gets `409` (`cancellation of running jobs is not supported
yet` — GPU concurrency is 1 and the pipeline is mid-card in v1); `done` and
`failed` jobs also get `409` (`Job already finished with status '<x>'.`).

### Queueing and persistence

The queue lives entirely on disk under `JOBS_DIR`: one directory per token
holding `status.json` (written atomically via temp-file rename) and the
uploaded `audio_source.wav`. Admission assigns `seq = max(existing seq) + 1`
under a lock, so `seq` is a monotonic FIFO order — this is exactly the order
the single in-process worker (`GPU concurrency = 1`, `QUEUE_WORKER_ENABLED=1`)
claims jobs in, and exactly what `queue_position` reports. `queued` and
`processing` jobs count against limits; each job's `session` id is persisted
alongside it.

**Restart behavior:** the queue is rebuilt by scanning `JOBS_DIR`, so a
restart (crash or redeploy) does not lose queued work. A job caught
mid-`processing` is requeued on boot with its `seq` intact (message:
`Requeued after service restart`) and re-runs from the start — FIFO order is
preserved across restarts.

**Retention:** finished jobs (all statuses except the live pair
`queued`/`processing`) are never swept automatically. `RETENTION_DAYS` exists
in `.env.example` and is documented in `CONFIG.md` as *reserved for a future
retention sweep*; no shipped code path reads it yet. Until then, clean up
`JOBS_DIR/<token>/` directories yourself when disk matters — a cancelled
job's `status.json` (and a done job's outputs) persist indefinitely.

### Session identity

`MAX_JOBS_PER_SESSION` is enforced per *session*, resolved for every request
in this order (then hashed, sha256 → first 32 hex chars, so no raw token or
IP is ever stored on disk):

1. the value of the session header — `Authorization` by default, override
   with `SESSION_HEADER` (e.g. `-H 'Authorization: Bearer alice'` gives
   `alice` a stable quota across IP changes);
2. the client IP when no such header is present (`request.client.host`);
3. an anonymous bucket when neither is available.

A slot frees when the job reaches `done`, `failed`, or `cancelled`. Behind a
reverse proxy that replaces the peer address, all clients share one IP
bucket — set a real session header in that deployment.

## Quickstart (GPU host, Docker)

Prerequisites: Docker with the NVIDIA container toolkit and a CUDA 13-era
host driver. The reference GPU is the NVIDIA T400 4GB (Turing, compute
capability 7.5) on which the pinned engine build's real `sm_75` kernels were
validated during the spike; other architectures were not the release target.

```bash
git clone https://github.com/dlasher/AutoDiarist.git autodiarist
cd autodiarist
cp .env.example .env
docker compose up --build
```

In another terminal (see *Test fixture* below — `audio.wav` is not tracked
in this repo):

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
```

Save the token from the response, then poll and download:

```bash
TOKEN=<token-from-response>
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md" -o transcript.md
```

The compose file pins the backend contract (`CRISPASR_GPU_BACKEND=cuda`,
`DIAR_METHOD=foxnose`, `VRAM_LIMIT_MIB=3584`, `CRISPASR_TIMEOUT_SECONDS=1800`)
and persists the foxnose embedder cache (`crispasr-cache:/var/cache/crispasr`)
and job state (`jobs-data:/data/jobs`). `HOST_PORT` (default 8000) remaps the
published port.

## Quickstart (no GPU, mock pipeline)

Prerequisites: Python 3.10+ only. `ffmpeg` is required exclusively for real
transcription runs; mock mode never touches it.

```bash
git clone https://github.com/dlasher/AutoDiarist.git autodiarist
cd autodiarist
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
MOCK_PIPELINE=1 uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal (any `.wav` filename validates; mock mode does not read
the bytes):

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
TOKEN=<token-from-response>
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md"
```

## Test fixture

`audio.wav` (~74 MB, 39-minute meeting recording) is a **local untracked
test fixture** — `.gitignore` excludes `*.wav` so the repo stays under 5 MB.
Place any WAV file at `./audio.wav` in the repo root to follow the
curl examples above. Mock mode (`MOCK_PIPELINE=1`) never reads the bytes;
real runs need 16 kHz-capable content (any standard WAV is accepted —
`ffmpeg` normalizes it). To generate a stand-in tone locally:

```bash
ffmpeg -f lavfi -i sine=frequency=440:duration=5 -ac 1 -ar 16000 audio.wav
```

## Language

`en-US` is fixed for v1 (SPEC 08, "Languages" section): the adapter always
passes `-l en-US` (overridable via `CRISPASR_LANGUAGE` for local per-locale
experimentation) and the rendered metadata table reports `en-US`. No
`language` request parameter is accepted.

## Other backends

The image pins the `cuda13` Linux release asset, but the engine itself
carries a wider build matrix. The authoritative source for per-asset
detail is upstream: CrispASR
[docs/install.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/install.md)
("Prebuilt Linux tarballs") and
[docs/environment-variables.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/environment-variables.md).
**Aside from the CPU execution path (measured below), no row under `cuda13`
is verified by this project** — the only hardware we have here is NVIDIA, so
the remaining non-CUDA paths are documented as upstream facts for reference,
not as supported deployments of this service. The
adapter additionally accepts only `cpu`/`cuda`/`metal`/`vulkan` for
`CRISPASR_GPU_BACKEND` (CONFIG.md); there is no supported engine path here
for HIP, WASM, or CANN at all.

| Engine path | Upstream build/runtime status | Verified here? |
| --- | --- | --- |
| `crispasr-linux-x86_64-cuda13.tar.gz` (pinned) | CUDA 13; falls back to CPU when CUDA libs are absent | yes — T400, `sm_75` (spike t_7e17589e); CPU execution of this same asset tested separately (row below, measurements after the table) |
| CPU execution of the pinned `cuda13` asset (`libggml-cuda.so` absent → `GGML_BACKEND_DL` fallback) | tested v0.8.43 on the 60 s spike source; a pure-CPU host (no NVIDIA driver at all) is N/A on this GPU box | yes — ~1.7x slower wall than CUDA on the same 60 s input; results in kanban t_18ee2e43 (`REPORT_CPU.md`) |
| `crispasr-linux-x86_64-cuda.tar.gz` | CUDA 12; same CPU fallback | no — NOT verified |
| `crispasr-linux-x86_64` plain / `-cpu-legacy` / `-avx512` | CPU-only standalone builds; their inference path is the same CPU backend exercised in the fallback row above, but these exact assets were never downloaded on this host | no — NOT verified as separate assets |
| `crispasr-linux-x86_64-vulkan.tar.gz` | Vulkan GPU; **no** CPU fallback | no — NOT verified |
| `crispasr-linux-x86_64-hip.tar.gz` | AMD ROCm; **no** CPU fallback | no — NOT verified |
| Metal (`-DGGML_METAL=ON` source build) | Apple GPU; ggml prefers CUDA > Metal > SYCL > Vulkan > CPU at runtime | no — NOT verified |
| WASM (`build-wasm.sh`) | all backends compile to WebAssembly; runs multithreaded in the browser | no — NOT verified |
| CANN (`-DGGML_CANN=ON`, `--gpu-backend cann`) | Huawei Ascend; see the CANN caveat below | no — NOT verified |

### CPU execution — measured (v0.8.43, kanban t_18ee2e43)

Protocol: the pinned `crispasr-linux-x86_64-cuda13.tar.gz` (sha256
`fea397a3…4fdf54`, `ggml backends: cpu,cuda`) re-extracted to a clean dir on
the T400 VM, `libggml-cuda.so` quarantined out of the extraction (the
documented `GGML_BACKEND_DL` fallback — no CUDA runtime can load), and the
60 s spike source (`excerpt60.wav`, sha256 `b5559cf7…afd46ce`, 1,920,078 B)
piped via ffmpeg exactly as the GPU runs. No GPU flags were passed.
Results on 4 vCPUs: exit 0; ASR-only wall 13.7 s median (3 repeats,
±0.02 s; first cold-cache run ~20 s) vs 8 s on CUDA for the same input —
**~1.7x slower wall** (engine-reported 13.67 s vs 6.16 s, ~2.2x); diarized
wall 24.4 s vs 13 s on CUDA; peak RSS ~937 MiB ASR-only, ~1,065 MiB
diarized. Transcript parity vs the CUDA runs on the same clip: 3/3 sections,
identical speaker labels, 97.7% word agreement, ±400 ms segment-boundary
drift. Guards: `nvidia-smi` showed 0 compute apps and 0 MiB used throughout;
`audio.wav` and the spike/VRAM artifacts verified byte-identical pre/post.
**Caveat found:** `--gpu-backend cpu` alone is *not* "no CUDA runtime
loaded" — the plugin still dlopens and `ggml_cuda_init` still enumerates the
GPU before compute falls to CPU; true CPU-only execution requires the runtime
or the backend `.so` to be absent (or a CPU-only build asset).

Fallback rule (per upstream install.md): the Linux `-cuda` and `-cuda13`
tarballs use dynamic backend loading (`GGML_BACKEND_DL`) since v0.8.30 —
if the NVIDIA driver or CUDA runtime is missing, the CUDA backend simply
does not load and the process silently runs on CPU and exits 0. That is
exactly the failure mode the adapter guards against by pinning
`--gpu-backend cuda` and probing the build's registered backends before
the first job (see the silent-CPU-fallback note in *Replication notes*
and TROUBLESHOOTING.md). The `-vulkan` and `-hip` tarballs are statically
linked against their respective runtimes and do **not** fall back to CPU:
without the matching driver they fail loudly instead of quietly
transcribing slowly. Upstream keeps those two release legs static until
each has been A/B-tested on real AMD and Vulkan hardware.

CANN is the mirror-image caveat: the manual `cann-build.yml` workflow
compiles CLI, shared library, diff and quantizer with the official CANN
SDK images on x86_64/910B (ACL graphs enabled) and arm64/310P (ACL graphs
disabled), but GitHub hosted runners carry no Ascend NPU — a green run is
compile proof, not inference or performance proof. Treat CANN as
unvalidated upstream as well as here.

To check what a given prebuilt asset actually requires before deploying
it:

```bash
readelf -d crispasr | grep NEEDED
```

Engine environment variables follow upstream's
`CRISPASR_<BACKEND>_<FEATURE>` convention (per-backend switches such as
`CRISPASR_OMNIVOICE_CODEC_GPU`; process-global variables drop the backend
segment — `docs/environment-variables.md` is the canonical list). The
adapter's own `CRISPASR_*` names share that prefix but are this service's
settings, documented in CONFIG.md; the adapter hands values to the engine
as explicit CLI flags (e.g. `CRISPASR_CACHE_DIR` -> `--cache-dir`).

## Replication notes

- Engine: crispasr prebuilt `v0.8.43`, `cuda13` x86-64 release asset
  (archive root dir stripped with `--strip-components=1`), extracted to
  `/opt/crispasr`. The download is validated by `sha256sum -c` against the
  GitHub release asset digest at build time; the `crispasr --version`
  backend assertion runs at job time (see PIPELINE.md), not at build.
- Model: canonical Q4_K GGUF `nemotron-3.5-asr-streaming-0.6b-q4_k.gguf`
  (~408.6 MB) from the `cstr/nemotron-3.5-asr-streaming-0.6b-GGUF` Hugging
  Face repo (the engine's own auto-download target; published LFS OID
  re-verified on the VM), baked at `/opt/models` and pinned by content hash:
  the build asserts
  `sha256 22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd`
  with `sha256sum -c` before accepting the download (Dockerfile lines 99-100;
  the crispasr release tarball is likewise pinned,
  `fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54`).
  Do not swap in another repo's `q4_k` bytes: different quantization means
  different WER/VRAM behavior (spike verdict A3).
- Uploads are decoded to 16 kHz mono PCM s16 by `ffmpeg` before the engine
  sees them (engine accepts 24 kHz natively; the adapter standardizes).
- CUDA 13 runtime libraries are an explicit image prerequisite: exactly
  `cuda-cudart-13-0` + `libcublas-13-0` from the NVIDIA ubuntu2404 repo
  keyring — no CUDA toolkit, no devel headers, no `nvidia/cuda` base image.
  Their directory is exported as `CRISPASR_LIB_DIR` and prepended (with the
  binary's own directory) to the subprocess `LD_LIBRARY_PATH`.
- Silent-CPU-fallback warning: without those libraries the crispasr CLI
  silently runs on CPU and exits 0 — no error surfaces. Measured slowdown
  is workload-sized: ~1.7x wall on the 60 s clip but ~11.6x on the 39-minute
  fixture (card t_18ee2e43; see "CPU execution — measured" above). The adapter
  therefore pins `--gpu-backend cuda` and probes `crispasr --version`
  before every first job per process, asserting that the pinned backend is
  in the build's `Registered backends` list. A `cuda` build that lacks the
  runtime libs fails that assertion fast with a typed `CrispasrConfigError`
  instead of burning minutes of CPU inference per job.
- VRAM: one whole-file invocation per job (the engine auto-slices audio into
  ~30 s windows internally; measured 1085 MiB peak over the full 39-minute
  fixture), an `nvidia-smi` sampler thread records per-stage allocated/peak
  memory (visible in `docker logs` as `allocated=/peak=` INFO lines), and a
  peak above `VRAM_LIMIT_MIB` raises `TranscriptionOOM` rather than
  thrashing the card.
- Final GPU-box state (post-cleanup, card t_413e2b7b, 2026-10-20): ~48 GB
  reclaimed — `df -h` free went from 25 GB to 73 GB. Purged: the homemade
  `nvidia-asr:local` image (18.5 GB) and leftover build bases, the docker
  build cache (25.0 GB), legacy volumes (4.7 GB), the spike/adapter host
  dirs (`/opt/nvidia-asr-spike`, `/root/spike-crispasr`,
  `/root/nvidia-asr-adapter` — all reports and scripts were archived first
  under `~/CODE/nvidia-asr-planning/archive/vm-cleanup-2026-10-20/`), the
  host `cuda13` apt runtime libs (~1 GB; the final image bundles its own,
  proven by a post-purge GPU job), and stale job artifacts (0.14 GB).
  Still running: the fresh-clone deployment at `/root/replication-test`
  (image `autodiarist:local`; volumes `replication-test_asr-cache`,
  `replication-test_queue-state`, `crispasr-cache`), GPU driver 580.178.04.
  A post-cleanup GPU job (`745556b4…`, 60 s clip, done < 20 s) logged
  `crispasr probe: version-ok backends=['cpu', 'cuda'] pinned=cuda`.

## Tests

```bash
python3 -m pytest tests
```

The API tests (including every endpoint example shape above) run in-process
against `MOCK_PIPELINE=1`, no GPU required. The Docker packaging tests
(`tests/test_docker_packaging.py`) assert the backend pins, runtime-library
prerequisites, GGUF hash verification, and that the torch-era knobs stay
removed. `scripts/vm_integration.py` is the on-VM acceptance harness for the
real CrispASR path.

## Documentation map

| File | Covers |
| --- | --- |
| `CONFIG.md` | Every environment variable, defaults, validation, retired knobs |
| `PIPELINE.md` | CrispASR adapter, segment mapping, self-ID, render contract |
| `TROUBLESHOOTING.md` | Backend assertion, queue limits, failure post-mortems, restart behavior |
| `CHANGELOG.md` | Release history and QA traceability |
