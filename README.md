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
**No row below `cuda13` is verified by this project** — the only hardware
we have here is NVIDIA, so the non-CUDA paths are documented as upstream
facts for reference, not as supported deployments of this service. The
adapter additionally accepts only `cpu`/`cuda`/`metal`/`vulkan` for
`CRISPASR_GPU_BACKEND` (CONFIG.md); there is no supported engine path here
for HIP, WASM, or CANN at all.

| Engine path | Upstream build/runtime status | Verified here? |
| --- | --- | --- |
| `crispasr-linux-x86_64-cuda13.tar.gz` (pinned) | CUDA 13; falls back to CPU when CUDA libs are absent | yes — T400, `sm_75` (spike t_7e17589e) |
| `crispasr-linux-x86_64-cuda.tar.gz` | CUDA 12; same CPU fallback | no — NOT verified |
| `crispasr-linux-x86_64` plain / `-cpu-legacy` / `-avx512` | CPU-only builds | no — NOT verified |
| `crispasr-linux-x86_64-vulkan.tar.gz` | Vulkan GPU; **no** CPU fallback | no — NOT verified |
| `crispasr-linux-x86_64-hip.tar.gz` | AMD ROCm; **no** CPU fallback | no — NOT verified |
| Metal (`-DGGML_METAL=ON` source build) | Apple GPU; ggml prefers CUDA > Metal > SYCL > Vulkan > CPU at runtime | no — NOT verified |
| WASM (`build-wasm.sh`) | all backends compile to WebAssembly; runs multithreaded in the browser | no — NOT verified |
| CANN (`-DGGML_CANN=ON`, `--gpu-backend cann`) | Huawei Ascend; see the CANN caveat below | no — NOT verified |

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
- Silent-CPU-fallback warning: without those libraries the crispasr CLI runs
  on CPU (~11x slower) and exits 0 — no error surfaces. The adapter
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
