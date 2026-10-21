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
GET    /v1/jobs/{token}          -> status (+ queue_position while queued)
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
`status`, `seq` (FIFO position at admission), `queued_at`, `progress`,
`message`, `filename`, plus the bookkeeping fields `session` (the hashed
session id) and `audio_path` (server-local path) — treat those as opaque.
`status` is one of `queued`, `processing`, `done`, `failed`, `cancelled`:

- `queued` — adds `queue_position`: the job's **1-based rank among currently
  queued jobs** against the single worker's pickup order; `queue_position: 1`
  means next.
- `processing` — adds `started_at`; `progress` advances through the pipeline
  stages (0.05 claimed → 0.1 decode → 0.4+ transcribe with per-chunk
  increments → 0.95/0.98 self-ID/render) and `message` carries the stage text
  (e.g. `Claimed by worker`).
- `done` — `progress: 1.0`, `message: "Transcription complete."`. The result
  itself is fetched from the download endpoint at
  `/v1/jobs/{token}/download` (the poll payload does not carry a URL field).
- `failed` — `progress: 1.0` plus `error` (the exception message from the
  worker).
- `cancelled` — `message: "Cancelled by client while queued"` plus
  `cancelled_at`.

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
for HIP, WASM, or CANN at all. Operator details for the unverified rows —
per-row URLs, prerequisites, selection flags, a container-asset-swap
recipe, and a "convert a row to verified" checklist — follow in the
subsections below the measured-CPU material.

| Engine path (v0.8.43 asset) | Prerequisites | Selection flag | Differences that matter operationally | Verified here? |
| --- | --- | --- | --- | --- |
| `crispasr-linux-x86_64-cuda13.tar.gz` (pinned) | NVIDIA driver + CUDA 13 runtime (`libcudart.so.13`, `libcublas.so.13`; image apt-pins `cuda-cudart-13-0` + `libcublas-13-0`) | image contract: `CRISPASR_GPU_BACKEND=cuda` → `--gpu-backend cuda` | CUDA 13; falls back to CPU when CUDA libs are absent (fallback rule below) | yes — T400, `sm_75` (spike t_7e17589e); CPU execution of this same asset tested separately (row below, measurements after the table) |
| CPU execution of the pinned `cuda13` asset (`libggml-cuda.so` absent → `GGML_BACKEND_DL` fallback) | same asset; fires when the NVIDIA driver / CUDA runtime is missing | automatic once the CUDA module fails to load; `--gpu-backend cpu` alone is *not* CPU-only (fallback rule below) | v0.8.43 on the 60 s spike source; a pure-CPU host (no NVIDIA driver at all) is N/A on this GPU box | yes — ~1.7x slower wall than CUDA on the same 60 s input; results in kanban t_18ee2e43 (`REPORT_CPU.md`) |
| `crispasr-linux-x86_64-cuda.tar.gz` | NVIDIA driver + CUDA 12 runtime (`libcudart.so.12`, CUDA-12 `libcublas`) | `--gpu-backend cuda`; service-side pin unchanged | CUDA 12; same `GGML_BACKEND_DL` silent-CPU fallback as cuda13 | no — NOT verified on this project's hardware |
| `crispasr-linux-x86_64.tar.gz` (plain) | none (CPU-only build); AVX2 baseline — "it fails fast with an explanatory error on older CPUs since #380/#405" (upstream install.md) | plain binary; `--gpu-backend cpu` | CPU-only standalone asset; the asset GitHub-release installers default to | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-cpu-legacy.tar.gz` | none (CPU-only); generic x86-64/SSE2 baseline (upstream install.md table) | `--gpu-backend cpu` | CPU-only; targets pre-AVX2 hosts | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-avx512.tar.gz` | none (CPU-only); AVX-512-capable CPU (upstream install.md table) | `--gpu-backend cpu` | CPU-only; AVX-512 vectorized build | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-vulkan.tar.gz` | Vulkan-capable GPU plus its vendor ICD/runtime — upstream: the tarball "require[s] their respective runtimes at link time (no CPU fallback)" | `--gpu-backend vulkan` (+ `-dev N` to pick the device); service-side `CRISPASR_GPU_BACKEND=vulkan` is an accepted value | statically linked: **no** CPU fallback — fails loudly without the runtime | no — NOT verified on this project's hardware |
| `crispasr-linux-x86_64-hip.tar.gz` | AMD GPU + ROCm runtime (same upstream sentence as Vulkan) | the asset *is* the ROCm build; upstream documents no `--gpu-backend hip` example — and there is **no service-side path**: `CRISPASR_GPU_BACKEND` accepts only `cuda`/`cpu`/`metal`/`vulkan` (CONFIG.md) | statically linked: **no** CPU fallback | no — NOT verified on this project's hardware |
| Metal — `-DGGML_METAL=ON` source build on macOS/Apple Silicon (the v0.8.43 page also carries `crispasr-macos-arm64.tar.gz`; never opened here) | macOS on Apple Silicon (upstream marks `-DGGML_METAL=ON` as the Apple Silicon flag) | `--gpu-backend metal`; service-side `CRISPASR_GPU_BACKEND=metal` is accepted for the pip-install dev path, not the Linux container | there is no Metal release asset for Linux at all — this project's image is Linux | no — NOT verified on this project's hardware |
| WASM — `crispasr-wasm.tar.gz` (prebuilt) or `build-wasm.sh` source build | browser with threading (upstream README: "Multithreaded, runs entirely client-side with COOP/COEP headers"); emsdk toolchain for source builds | n/a for the CLI/service — browser client-side artifact; this service spawns the native CLI | browser-oriented path, not a server deployment | no — NOT verified on this project's hardware |
| CANN — `-DGGML_CANN=ON` source build via the manual workflow | Huawei Ascend NPU + CANN SDK; **no CANN release asset exists in v0.8.43** (checked against the release's asset list) | build with `-DGGML_CANN=ON`, then `--gpu-backend cann -dev N` | see the CANN caveat below: a green `cann-build.yml` run is compile proof only | no — NOT verified on this project's hardware |

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
transcribing slowly. Upstream keeps those two release legs static because
flipping them "needs an A/B on real AMD and Vulkan hardware first"
(upstream install.md; quoted — that A/B has not happened upstream).

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

### ggml's auto-selection order — and why this service pins the backend

An engine binary that was compiled with several GPU backends picks one
itself at startup. Upstream states the order verbatim (install.md,
"GPU backends"): ggml "will pick the highest-priority compiled backend at
runtime, in ggml's registration order (CUDA > Metal > SYCL > Vulkan > CPU —
MUSA registers through the CUDA slot, so it shares CUDA's position)". This
project never relies on that order: the adapter always passes
`--gpu-backend $CRISPASR_GPU_BACKEND` (pinned `cuda` in the image — see
`pipeline/transcribe.py` and `docker-compose.yml`), and asserts before the
first job of each process that the pinned backend appears in
`crispasr --version`'s registered-backends list — so there is no silent
fallback surprise: a build or host that cannot run the pinned backend
fails fast at the probe instead of quietly computing on something else,
and no preference-order pick can select a backend other than the one
declared.
The adapter additionally refuses any `CRISPASR_GPU_BACKEND` outside
`cuda`/`cpu`/`metal`/`vulkan` (CONFIG.md), so a pinned backend is either
explicitly one of those or the service fails fast with a typed
`CrispasrConfigError` at startup. The order above only matters if you run
the raw engine CLI without this service.

### Operator how-to for the unverified rows (v0.8.43 assets)

Every bullet below is upstream-documented or an observation about our own
adapter/container — NOT verified on this project's hardware. Nothing here
is a claim that any of these paths run correctly; that is exactly what the
recipe further down establishes for one backend at a time. Exact asset
names and CPU-feature baselines are quoted from upstream install.md's
prebuilt-tarball table; release-asset existence was checked against the
v0.8.43 release listing (names only — no tarballs were downloaded here,
per the docs-only decision).

- **Vulkan** — asset
  [`crispasr-linux-x86_64-vulkan.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-vulkan.tar.gz).
  Prerequisite: a Vulkan-capable GPU with its vendor ICD/runtime —
  upstream: "The HIP and Vulkan tarballs still require their respective
  runtimes at link time (no CPU fallback)." Select with
  `crispasr --gpu-backend vulkan -dev N -m <gguf> -f <wav>` (`-dev N` picks
  the device on hybrid-GPU hosts); service-side, `CRISPASR_GPU_BACKEND=vulkan`
  is an accepted pin value. Difference that matters: statically linked —
  without the Vulkan runtime the binary fails loudly at load instead of
  degrading to CPU.
- **HIP / ROCm** — asset
  [`crispasr-linux-x86_64-hip.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-hip.tar.gz).
  Prerequisite: AMD GPU + ROCm runtime (same upstream sentence as Vulkan).
  The asset itself is the ROCm build; upstream install.md documents no
  `--gpu-backend hip` example. Service-side there is **no supported path
  yet**: `CRISPASR_GPU_BACKEND` accepts only `cuda`/`cpu`/`metal`/`vulkan`
  (our adapter's whitelist, CONFIG.md), so end-to-end deployment through
  this service needs an adapter change first — a limitation of this repo,
  not of the engine. Difference that matters: statically linked, no CPU
  fallback.
- **CUDA 12** — asset
  [`crispasr-linux-x86_64-cuda.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cuda.tar.gz).
  Prerequisite: NVIDIA driver + CUDA 12 runtime (`libcudart.so.12` and
  CUDA-12 `libcublas`; the image's apt-pinned `-13-0` packages do not
  satisfy it). Select with `--gpu-backend cuda` — the service-side pin is
  already `cuda`. Difference that matters: same `GGML_BACKEND_DL`
  silent-CPU-fallback behavior as the pinned cuda13 asset (upstream): with
  the CUDA 12 runtime absent it "simply runs on CPU" quietly — the same
  trap, same guard (see the fallback rule above).
- **CPU plain / legacy / AVX-512** — assets
  [`crispasr-linux-x86_64.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64.tar.gz),
  [`crispasr-linux-x86_64-cpu-legacy.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cpu-legacy.tar.gz),
  [`crispasr-linux-x86_64-avx512.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-avx512.tar.gz).
  Prerequisites: none beyond a CPU of the right feature baseline — plain is
  an "AVX2 CPU build" that "fails fast with an explanatory error on older
  CPUs since #380/#405", `-cpu-legacy` targets "generic x86-64/SSE2",
  `-avx512` targets AVX-512-capable hosts (all three quotes: upstream
  install.md). Select: run the binary directly, optionally
  `--gpu-backend cpu`. Difference that matters: there is no GPU module to
  miss-load, so the silent-CPU-fallback trap of the `-cuda` assets does not
  apply; throughput differs per ISA. These exact tarballs were never
  downloaded on this host — the CPU *code path* itself is what t_18ee2e43
  measured (rows above), not these assets.
- **Metal** — source build `-DGGML_METAL=ON` on macOS/Apple Silicon
  (upstream install.md flags Metal as the Apple Silicon target; the default
  source build recipe in install.md covers macOS). The v0.8.43 page also
  carries `crispasr-macos-arm64.tar.gz` / `crispasr-macos.tar.gz`, but
  their contents were never opened here, so their Metal status is not
  established by this README. Selection: `--gpu-backend metal`;
  service-side `CRISPASR_GPU_BACKEND=metal` is an accepted pin value, but
  note the difference that matters operationally: **there is no Metal
  release asset for Linux at all** — the containerized path stays
  `cuda`/`cpu`, and Metal is only reachable outside this image (a macOS dev
  machine running the pip-installed service). NOT verified on this
  project's hardware — we own no Apple Silicon.
- **WASM** — prebuilt asset
  [`crispasr-wasm.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-wasm.tar.gz)
  or a source build via `build-wasm.sh`. Upstream README: "All backends
  compile to WebAssembly (4.3 MB) via build-wasm.sh. Multithreaded, runs
  entirely client-side with COOP/COEP headers." Prerequisites: a browser
  that serves the demo page with COOP/COEP headers for threading; emsdk for
  the source build. Selection: none — this is a browser client-side
  artifact; the service spawns the native CLI, so there is no service path
  to select. Difference that matters: WASM is browser-oriented, not a
  server deployment option for this stack.
- **CANN** — no release asset: the v0.8.43 asset list contains no CANN
  build (checked against the release listing; upstream keeps it a manual
  workflow precisely because "Hosted runners have no Ascend NPU, so a
  successful build does not establish inference or performance" — install.md).
  Build path: `cmake -DGGML_CANN=ON` (or the manual `cann-build.yml`
  workflow: x86_64/910B with ACL graphs enabled, arm64/310P with them
  disabled — upstream install.md, restated in the CANN caveat above).
  Prerequisite: Huawei Ascend NPU + CANN SDK. Select at run time:
  `crispasr --gpu-backend cann -dev N`. Difference that matters: this
  service cannot pin it (`CRISPASR_GPU_BACKEND` rejects `cann`), so the
  engine is used standalone; a green workflow run proves compilation only.

### Pointing the service at a different engine asset

The adapter does not hard-code engine locations. The variable names below
are the real contract — cross-checked against CONFIG.md,
`pipeline/transcribe.py`, `.env.example`, and `docker-compose.yml` at HEAD
(`1c003a8`):

- `CRISPASR_BIN` — path to the crispasr CLI the adapter spawns (image
  default `/opt/crispasr/crispasr`; unset falls back to a bare `crispasr`
  on `PATH`). The adapter probes this exact file with `--version` and
  fails fast when the pinned backend is not in its registered-backends
  list.
- `GGUF_MODEL` — path to the canonical Nemotron Q4_K GGUF (image default
  `/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf`). Set it **empty**
  to switch the adapter to `-m auto --auto-download` instead.
- `CRISPASR_GPU_BACKEND` — the pin described above.
- `CRISPASR_LIB_DIR` — directory prepended to the subprocess
  `LD_LIBRARY_PATH` (image default: the apt CUDA-13 lib dir; unset it when
  your replacement asset carries its own runtime libs).

Baked into the image are the engine at `/opt/crispasr` and the model at
`/opt/models`; to run a different prebuilt asset against the same service
without rebuilding, mount over those paths. Example — pointing the
container at the Vulkan tarball (NOT verified on this project's hardware;
the adapter whitelist accepts `vulkan`, and the engine must be a Vulkan
build because the container's baked CUDA-13 libs do not affect it):

```bash
# host: extract the tarball into ./vulkan-engine/ (upstream layout: a
# single root dir inside — strip it like the Dockerfile does), and place
# the canonical Q4_K GGUF (or your own) in ./models/
docker run --rm \
  -e CRISPASR_BIN=/opt/crispasr/crispasr \
  -e GGUF_MODEL=/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
  -e CRISPASR_GPU_BACKEND=vulkan \
  -e CRISPASR_LIB_DIR= \
  -v "$PWD/vulkan-engine:/opt/crispasr:ro" \
  -v "$PWD/models:/opt/models:ro" \
  autodiarist:local
```

Container-side differences to expect (our own observation of this repo's
files, not upstream): the shipped compose file reserves NVIDIA devices
(`driver: nvidia` in `docker-compose.yml`) — a Vulkan/AMD host needs that
`deploy.resources` block overridden and the vendor ICD/ROCm runtime
visible inside the container; the image's apt CUDA-13 packages are inert
for non-CUDA assets but still present. Expect to validate the swap
yourself with the recipe below before trusting it in production.

### Converting a row from NOT-verified to verified

How to verify a backend yourself — run this for the one you own hardware
for, capture everything, and record the result; do not mark a row `yes`
from inference or upstream promises:

1. Download the exact asset from its URL above and record its digest:
   `curl -LO https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/<asset>.tar.gz`,
   `sha256sum <asset>.tar.gz`, extract with `--strip-components=1`.
2. Confirm what it links against before running:
   `readelf -d crispasr | grep NEEDED` (see the readelf tip above), and
   `crispasr --version` — capture the full `Registered backends` list.
3. Run inference directly on a real audio file (any WAV the ffmpeg decode
   step accepts — this project uses the 60 s excerpt of its meeting
   fixture), pinning the backend explicitly:
   `crispasr -f <wav> -m <gguf> -l en-US --gpu-backend <backend> -dev 0 -oj`.
4. Capture: wall time, the process's device utilization during the run
   (`nvidia-smi dmon`, `rocm-smi`, `vkcube --gpu N` / `vainfo`, as
   applicable), and the transcript output file.
5. Then exercise it through the service: start the container with the
   `CRISPASR_BIN`/`GGUF_MODEL`/`CRISPASR_GPU_BACKEND` overrides shown
   above, POST one real job, and download the result as both JSON and MD.
   Success requires a non-empty transcript AND the pinned backend
   appearing in the device-utilization capture (the same evidence standard
   t_18ee2e43 used for the CPU row).
6. Record the outcome two places, in the same PR: create a kanban card on
   the `nvidia-asr` board with the captures (model the protocol on the
   "CPU execution — measured" subsection above) **and** add a CHANGELOG.md
   line. Then update the table row: `no — NOT verified on this project's
   hardware` becomes `yes — <hardware>, <date> (kanban <card id>)`, and
   delete that row's entry from the how-to list.

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
