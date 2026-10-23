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

# ONE Dockerfile, two hardware targets (packaging card t_f82ad22a): the
# GPU_BACKEND build ARG (cuda | vulkan) selects ONLY the host-runtime apt
# layer and the CrispASR engine release asset. The app layer — python
# wheels, ffmpeg, the canonical Nemotron Q4_K GGUF, the sherpa diarization
# stack, the entrypoint, the non-root `app` user — is one source of truth
# shared byte-for-byte by both images and cannot diverge between backends.
# Compose files pin the choice (the user never edits a tracked file):
#   docker-compose.yml       GPU_BACKEND=cuda   NVIDIA path — CUDA-13 apt
#                            libs + deploy.resources nvidia reservation,
#                            exactly the behavior shipped since t_423d1ee4
#   docker-compose.amd.yml   GPU_BACKEND=vulkan ubuntu:24.04 + libvulkan1 +
#                            mesa-vulkan-drivers (Mesa RADV = hardware
#                            Vulkan on amdgpu, no ROCm needed); /dev/dri +
#                            /dev/kfd passed as devices; the entrypoint
#                            auto-selects the AMD Vulkan ICD.
ARG GPU_BACKEND=cuda

# Env contract consumed by pipeline/transcribe.py (adapter t_4d5f7d08):
#   CRISPASR_BIN            engine CLI baked at /opt/crispasr
#   GGUF_MODEL              canonical Q4_K GGUF baked at /opt/models;
#                           set empty to fall back to engine auto-download
#                           into CRISPASR_CACHE_DIR
#   CRISPASR_LIB_DIR        apt CUDA-13 lib dir, prepended to the subprocess
#                           LD_LIBRARY_PATH by _crispasr_env(); on the
#                           vulkan image docker-compose.amd.yml overrides it
#                           to "" (the Vulkan loader resolves normally)
#   CRISPASR_GPU_BACKEND    defaults to the GPU_BACKEND build ARG so the
#                           image is self-describing; the compose file for
#                           each hardware pins the literal (cuda|vulkan) so
#                           a missing runtime FAILS the --version probe
#                           instead of silently CPU-falling back
#                           (~11x slower; spike first-run trap)
#   DIAR_METHOD             sherpa default since card t_5a287979 (foxnose's
#                           BIC counter collapses 5-speaker panels to one
#                           label; sherpa = external sherpa-onnx subprocess
#                           with segmentation + embedding + clustering).
#                           Accepted: sherpa, ecapa, foxnose, sortformer,
#                           none. sherpa/ecapa REQUIRE the two ONNX models
#                           below; their absence fails startup (AC4 guard
#                           in pipeline/transcribe.py) instead of silently
#                           emitting a one-speaker transcript.
#   SHERPA_SEGMENT_MODEL / SHERPA_EMBEDDING_MODEL
#                           diarization ONNX models baked at /opt/models
#   SHERPA_BIN              wrapper injecting --clustering.cluster-threshold
#                           (baked at /opt/sherpa; the engine has no flag
#                           for the threshold, and it is the knob that
#                           decides the speaker count)
#   SHERPA_CLUSTER_THRESHOLD 0.40 default (NV-host calibration; larger ->
#                           fewer clusters). Tunable at deploy time.
#   DIARIZATION_MAX_SPEAKERS 8 default (card t_38e52d89): ceiling on
#                           distinct diarization labels, enforced by the
#                           wrapper's conditional second clustering pass
#                           (only when the auto pass exceeds it — clean
#                           audio replays byte-identical). 0 disables.
#                           See autodiarist-speaker-count-ceiling.md.
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
    CRISPASR_GPU_BACKEND=${GPU_BACKEND} \
    DIAR_METHOD=sherpa \
    SHERPA_SEGMENT_MODEL=/opt/models/sherpa-onnx-reverb-diarization-v1.onnx \
    SHERPA_EMBEDDING_MODEL=/opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
    SHERPA_BIN=/opt/sherpa/sherpa-diarization-wrapper \
    SHERPA_CLUSTER_THRESHOLD=0.40 \
    DIARIZATION_MAX_SPEAKERS=8 \
    VRAM_LIMIT_MIB=3584 \
    CRISPASR_TIMEOUT_SECONDS=1800

# Host GPU runtime, selected by the GPU_BACKEND build ARG — the ONLY
# apt difference between the two images.
#   cuda: the crispasr tarball deliberately does NOT bundle libcudart.so.13
#     / libcublas.so.13; without them the CLI runs on CPU without an error.
#     Install exactly the two packages the spike pinned (cuda-cudart-13-0,
#     libcublas-13-0) from the NVIDIA ubuntu2404 repo keyring — no toolkit,
#     no devel headers, no nvidia/cuda base image.
#   vulkan: libvulkan1 (loader) + mesa-vulkan-drivers (RADV = hardware
#     Vulkan on amdgpu). No ROCm stack needed — the proven AMD bring-up on
#     10.4.10.55 uses exactly this pairing (amd-bringup.sh v3, ollama's
#     working shape on that host).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && case "$GPU_BACKEND" in \
         cuda) \
           curl -fsSL -o /tmp/cuda-keyring.deb \
             https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
           && dpkg -i /tmp/cuda-keyring.deb \
           && rm /tmp/cuda-keyring.deb \
           && apt-get update \
           && apt-get install -y --no-install-recommends cuda-cudart-13-0 libcublas-13-0 ;; \
         vulkan) \
           apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers ;; \
         *) \
           echo "FATAL: GPU_BACKEND must be 'cuda' or 'vulkan', got '$GPU_BACKEND'" >&2; \
           exit 1 ;; \
       esac \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg decodes uploads to 16 kHz mono before the engine sees them
# (unchanged from the retired path; 24 kHz passthrough verified on VM).
# soundfile ships its own libsndfile in the wheel — no apt copy needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# crispasr v0.8.43 prebuilt — per-backend release asset, sha256-pinned.
# The cuda13 digest is the spike AC1 asset digest; the vulkan digest was
# independently sha-digested for card t_f82ad22a (AC2) against the
# v0.8.43 GitHub release asset. Both archives wrap everything in one
# top-level directory (crispasr-linux-x86_64-<backend>/), hence
# --strip-components=1. A digest mismatch fails the build loudly.
RUN set -eux; \
    case "$GPU_BACKEND" in \
        cuda) \
            url=https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cuda13.tar.gz; \
            digest=fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54 ;; \
        vulkan) \
            url=https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-vulkan.tar.gz; \
            digest=d5c676e023df6fac61971dcef06fb7cce22ff3de95367f8e01efb8743c11ca46 ;; \
        *) \
            echo "FATAL: GPU_BACKEND must be 'cuda' or 'vulkan', got '$GPU_BACKEND'" >&2; \
            exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/crispasr.tar.gz "$url" \
    && echo "$digest  /tmp/crispasr.tar.gz" | sha256sum -c - \
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

# --- diarization stack (card t_5a287979) -------------------------------------
# BACKEND-INDEPENDENT ON PURPOSE: sherpa-onnx runs clustering on CPU
# (onnxruntime provider=cpu), so the CUDA and Vulkan images built from this
# Dockerfile carry the SAME model files and the SAME DIAR_METHOD default
# (AC5/AC4, cards t_5a287979 + t_f82ad22a): the GPU_BACKEND build ARG
# exists (engine tarball + runtime-libs layers only). These layers
# deliberately do NOT reference that arg, so the CUDA and Vulkan images
# are provably identical from here down -- same CLI, same libs, same
# models, same wrapper.
#
# (1) sherpa-onnx CLI + shared libs. The crispasr engine shells out to
#     sherpa-onnx-offline-speaker-diarization for --diarize-method
#     sherpa/ecapa; the pip wheel does NOT ship the CLI, only the release
#     tarballs do. Third vendored artifact class (after the engine tarball
#     and the ASR GGUF); sha256 over the exact v1.13.8 asset bytes,
#     computed on the NV host 2026-10-22 (GitHub does not publish digests).
RUN curl -fsSL -o /tmp/sherpa.tar.bz2 \
       https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-linux-x64-shared.tar.bz2 \
    && echo "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4  /tmp/sherpa.tar.bz2" | sha256sum -c - \
    && mkdir -p /opt/sherpa \
    && tar -xjf /tmp/sherpa.tar.bz2 -C /opt/sherpa --strip-components=1 \
    && rm -rf /opt/sherpa/include /tmp/sherpa.tar.bz2

# (2) The two diarization ONNX models (k2-fsa/sherpa-onnx release assets;
#     filenames as advertised by crispasr --help: "Download them from
#     https://github.com/k2-fsa/sherpa-onnx"). Segmentation = REVERB
#     DIARIZATION v1, not plain pyannote-segmentation-3-0: NV-host A/B on
#     the card fixtures shows it halves the single-speaker over-split
#     (BestPractice 7 -> 3 labels at thr 0.35) while the 249s panel file
#     keeps separating 5 speakers. Inner model.onnx digest re-verified
#     after extraction.
RUN curl -fsSL -o /tmp/reverb.tar.bz2 \
       https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2 \
    && echo "615761e980be1688da0ef81618c056134d63aa55ea0a5f1494c47393b9398eab  /tmp/reverb.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/reverb.tar.bz2 --wildcards '*/model.onnx' -O > /opt/models/sherpa-onnx-reverb-diarization-v1.onnx \
    && rm /tmp/reverb.tar.bz2 \
    && echo "8249e2e323f8fb0566a387d94b40a73d0b75d54ee60d02dcd729a56a5ba8ecea  /opt/models/sherpa-onnx-reverb-diarization-v1.onnx" | sha256sum -c - \
    && curl -fsSL -o /opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
       https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
    && echo "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b  /opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" | sha256sum -c -

# (3) Threshold wrapper: the engine passes --clustering.num-clusters plus
#     the two model paths to the sherpa CLI but has NO flag for
#     --clustering.cluster-threshold -- the knob that decides how many
#     speakers FastClustering emits. SHERPA_BIN (ENV above) points the
#     engine at this script; it injects SHERPA_CLUSTER_THRESHOLD (runtime
#     env, default 0.40) and execs the real binary.
COPY scripts/sherpa-diarization-wrapper.sh /opt/sherpa/sherpa-diarization-wrapper
RUN chmod +x /opt/sherpa/sherpa-diarization-wrapper \
    && ls -l /opt/sherpa/bin/sherpa-onnx-offline-speaker-diarization

# Build-time self-check: the sherpa CLI is dynamically linked against
# libonnxruntime.so, so a wrong LD_LIBRARY_PATH resolves only at exec
# time (a runtime failure per job, far from its cause). Running the
# wrapper's --help here proves, at BUILD time, that (a) SHERPA_REAL
# resolves, (b) /opt/sherpa/lib is picked up by the wrapper's
# prepend (no "error while loading shared libraries"), and (c) the
# --clustering.cluster-threshold flag the wrapper injects actually
# exists in the vendored CLI version. Verified by hand on the NV host
# against the staged v1.13.8 assets 2026-10-22 (rc=0, flag listed).
RUN sh -c '/opt/sherpa/sherpa-diarization-wrapper --help 2>&1 \
        | grep -q "clustering.cluster-threshold"'

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

# Container entrypoint (card t_f82ad22a): auto-selects the AMD Vulkan ICD
# (radeon|amdgpu glob under /usr/share/vulkan/icd.d — selecting ALL jsons
# there made the engine die at init with vulkan:11 on the proven bring-up;
# no-op on the CUDA image) and execs uvicorn exactly as the previous inline
# CMD did.
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

USER app
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
