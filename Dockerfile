# syntax=docker/dockerfile:1

# CrispASR deployment image (card t_423d1ee4; spike t_7e17589e GO verdict,
# adapter t_4d5f7d08). The retired in-process ASR toolkit is gone: the engine
# is the crispasr CUDA prebuilt CLI (v0.8.43), the model is the canonical
# cstr Nemotron 3.5 ASR streaming 0.6B Q4_K GGUF, and diarization runs the
# foxnose global-speaker pass by default. Target layout per spike REPORT.md;
# pipeline/transcribe.py probes `crispasr --version` and fails fast if the
# CUDA backend is not registered.
#
# Turing requirement (AC from t_95726260 line): the cuda13 build ships
# sm_75-real kernels, verified on the NVIDIA T400 4GB (CC 7.5).

# --- app wheel builder ------------------------------------------------------
# python:3.12-slim matches the runtime Ubuntu 24.04 system interpreter (3.12)
# so the wheels install without any other toolchain at runtime.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY api ./api
COPY pipeline ./pipeline
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip wheel --no-cache-dir --wheel-dir /wheels .

# --- runtime ----------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

# Env contract consumed by pipeline/transcribe.py (adapter t_4d5f7d08):
#   CRISPASR_BIN            engine CLI baked at /opt/crispasr
#   GGUF_MODEL              canonical Q4_K GGUF baked at /opt/models;
#                           set empty to fall back to engine auto-download
#                           into CRISPASR_CACHE_DIR
#   CRISPASR_LIB_DIR        apt CUDA-13 lib dir, prepended to the subprocess
#                           LD_LIBRARY_PATH by _crispasr_env()
#   CRISPASR_GPU_BACKEND    pinned to cuda so a missing runtime FAILS the
#                           --version probe instead of silently CPU-falling
#                           back (~11x slower; spike first-run trap)
#   DIAR_METHOD             foxnose default (sortformer selectable later)
#   VRAM_LIMIT_MIB          3.5 GiB ceiling; spike measured 1085 MiB peak
#                           over a 39-min file, engine auto-slices ~30 s
#   CRISPASR_TIMEOUT_SECONDS 30 min per-job budget
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JOBS_DIR=/data/jobs \
    PORT=8000 \
    CRISPASR_BIN=/opt/crispasr/crispasr \
    GGUF_MODEL=/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
    CRISPASR_CACHE_DIR=/var/cache/crispasr \
    CRISPASR_LIB_DIR=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
    CRISPASR_GPU_BACKEND=cuda \
    DIAR_METHOD=foxnose \
    VRAM_LIMIT_MIB=3584 \
    CRISPASR_TIMEOUT_SECONDS=1800

# CUDA-13 runtime libraries. The crispasr tarball deliberately does NOT
# bundle libcudart.so.13 / libcublas.so.13; without them the CLI runs on CPU
# without an error. Install exactly the two packages the spike pinned
# (cuda-cudart-13-0, libcublas-13-0) from the NVIDIA ubuntu2404 repo
# keyring — no toolkit, no devel headers, no nvidia/cuda base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL -o /tmp/cuda-keyring.deb \
       https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-cudart-13-0 libcublas-13-0 \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg decodes uploads to 16 kHz mono before the engine sees them
# (unchanged from the retired path; 24 kHz passthrough verified on VM).
# soundfile ships its own libsndfile in the wheel — no apt copy needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# crispasr v0.8.43 CUDA prebuilt. The sha256 is the GitHub release asset
# digest (spike AC1); the archive wraps everything in one top-level
# directory crispasr-linux-x86_64-cuda13/, hence --strip-components=1.
RUN curl -fsSL -o /tmp/crispasr.tar.gz \
       https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cuda13.tar.gz \
    && echo "fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54  /tmp/crispasr.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/crispasr \
    && tar -xzf /tmp/crispasr.tar.gz -C /opt/crispasr --strip-components=1 \
    && rm /tmp/crispasr.tar.gz

# Canonical Nemotron 3.5 ASR streaming 0.6B Q4_K GGUF (408.6 MB, engine's own
# auto-download target and published LFS OID re-verified on the VM). Do NOT
# swap in another repo's q4_k bytes: different quantization, different
# WER/VRAM behavior (spike A3 verdict).
RUN mkdir -p /opt/models \
    && curl -fsSL -o /opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
       https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
    && echo "22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd  /opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf" | sha256sum -c -

# API service itself.
COPY --from=builder /wheels /wheels
RUN python3 -m pip install --no-cache-dir --break-system-packages /wheels/*.whl \
    && rm -rf /wheels

# /var/cache/crispasr receives the foxnose embedder auto-download on first
# run; /data/jobs holds uploads + results. Named volumes inherit ownership
# from these pre-created, chowned directories.
RUN mkdir -p /var/cache/crispasr /data/jobs \
    && chown -R app:app /var/cache/crispasr /data/jobs

WORKDIR /app
COPY api ./api
COPY pipeline ./pipeline

USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
