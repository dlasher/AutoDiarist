"""Docker packaging asserts the CrispASR CUDA layout (card t_423d1ee4).

Backend per spike t_7e17589e + adapter t_4d5f7d08: the crispasr CUDA prebuilt
(v0.8.43) plus the apt CUDA-13 runtime libs, the canonical cstr Q4_K GGUF, and
the env contract consumed by pipeline/transcribe.py. The NeMo/torch image path
is retired and must not reappear.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Spike-verified digests (t_7e17589e REPORT.md; re-checked against the GitHub
# release asset digest and the cstr HuggingFace LFS OID).
CRISPASR_CUDA13_SHA256 = (
    "fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54"
)
GGUF_Q4K_SHA256 = (
    "22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd"
)
GGUF_NAME = "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_fetches_pinned_crispasr_cuda13_tarball() -> None:
    dockerfile = read("Dockerfile")

    assert (
        "github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/"
        "crispasr-linux-x86_64-cuda13.tar.gz" in dockerfile
    )
    assert CRISPASR_CUDA13_SHA256 in dockerfile
    assert "sha256sum -c" in dockerfile


def test_dockerfile_installs_cuda13_runtime_libs_not_silent_cpu_fallback() -> None:
    dockerfile = read("Dockerfile")

    # Tarball does NOT bundle libcudart/libcublas; without them crispasr
    # silently drops to CPU (~11x slower). Pinned via apt, dir exported.
    assert "cuda-cudart-13-0" in dockerfile
    assert "libcublas-13-0" in dockerfile
    assert "/usr/local/cuda-13.0/targets/x86_64-linux/lib" in dockerfile


def test_dockerfile_pins_canonical_cstr_q4k_gguf() -> None:
    dockerfile = read("Dockerfile")

    assert "cstr/nemotron-3.5-asr-streaming-0.6b-GGUF" in dockerfile
    assert GGUF_NAME in dockerfile
    assert GGUF_Q4K_SHA256 in dockerfile


def test_dockerfile_keeps_turing_compute_capability_75_requirement() -> None:
    dockerfile = read("Dockerfile").lower()

    # The cuda13 build ships sm_75-real kernels (Turing T400, spike AC1).
    assert "sm_75" in dockerfile
    assert "t400" in dockerfile or "turing" in dockerfile


def test_dockerfile_retires_the_nemo_torch_path() -> None:
    dockerfile = read("Dockerfile").lower()

    # word-boundary: the retired toolkit is "nemo", the *model* is nemotron.
    assert re.search(r"\bnemo\b", dockerfile) is None
    for retired in ("torch", "lightning", "lhotse"):
        assert retired not in dockerfile
    assert "FROM nvidia/cuda:12" not in dockerfile


def test_dockerfile_sets_runtime_env_contract_and_non_root_user() -> None:
    dockerfile = read("Dockerfile")

    assert "ffmpeg" in dockerfile
    assert "useradd" in dockerfile
    assert "USER app" in dockerfile
    assert "uvicorn api.main:app" in dockerfile
    # pipeline/transcribe.py env contract (adapter t_4d5f7d08).
    for fragment in (
        "CRISPASR_BIN=/opt/crispasr/crispasr",
        f"GGUF_MODEL=/opt/models/{GGUF_NAME}",
        "CRISPASR_CACHE_DIR=/var/cache/crispasr",
        "CRISPASR_LIB_DIR=/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "CRISPASR_GPU_BACKEND=cuda",
        "DIAR_METHOD=sherpa",
        "VRAM_LIMIT_MIB=3584",
        "CRISPASR_TIMEOUT_SECONDS=1800",
    ):
        assert fragment in dockerfile


def test_dockerfile_bakes_sherpa_diarization_stack_backend_independent() -> None:
    """Card t_5a287979 (AC3/AC5/AC8): the sherpa method's THREE vendored
    artifacts -- the sherpa-onnx CLI tarball, the segmentation model, the
    embedding model -- plus the threshold wrapper are baked into the single
    shared runtime layer, so the CUDA and Vulkan images carry the SAME
    files and the same default DIAR_METHOD. URLs are k2-fsa/sherpa-onnx
    GitHub release assets (never guessed), pinned by sha256."""
    dockerfile = read("Dockerfile")

    assert "k2-fsa/sherpa-onnx/releases/download/v1.13.8/" \
        "sherpa-onnx-v1.13.8-linux-x64-shared.tar.bz2" in dockerfile
    assert "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4" in dockerfile
    assert "speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2" in dockerfile
    assert "615761e980be1688da0ef81618c056134d63aa55ea0a5f1494c47393b9398eab" in dockerfile
    assert "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" in dockerfile
    assert "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b" in dockerfile
    assert "sha256sum -c" in dockerfile
    # models land at the paths the ENV contract advertises
    assert "/opt/models/sherpa-onnx-reverb-diarization-v1.onnx" in dockerfile
    assert "/opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" in dockerfile
    # wrapper installed at exactly the SHERPA_BIN default (compose + ENV)
    assert "COPY scripts/sherpa-diarization-wrapper.sh /opt/sherpa/sherpa-diarization-wrapper" in dockerfile
    assert "chmod +x /opt/sherpa/sherpa-diarization-wrapper" in dockerfile
    assert "SHERPA_BIN=/opt/sherpa/sherpa-diarization-wrapper" in dockerfile
    assert "SHERPA_CLUSTER_THRESHOLD=0.40" in dockerfile
    # AC5: the diarization layers run on CPU onnxruntime -- they must not
    # reference the CUDA engine assets, so the Vulkan build-arg variant
    # (t_f82ad22a) can share them unchanged.
    diar_block = dockerfile[dockerfile.index("--- diarization stack"):]
    assert "crispasr-linux" not in diar_block
    assert "cuda-cudart" not in diar_block
    assert "libcublas" not in diar_block


def test_compose_pins_gpu_backend_and_persists_crispasr_state() -> None:
    compose = read("docker-compose.yml")

    assert "driver: nvidia" in compose
    assert "count: all" in compose
    assert "CRISPASR_GPU_BACKEND: cuda" in compose
    assert "DIAR_METHOD: sherpa" in compose
    # t_5a287979: the sherpa method must ship its model wiring in compose,
    # same defaults on CUDA and Vulkan images (AC5).
    assert "SHERPA_SEGMENT_MODEL" in compose
    assert "SHERPA_EMBEDDING_MODEL" in compose
    assert "SHERPA_BIN" in compose
    assert "SHERPA_CLUSTER_THRESHOLD" in compose
    assert "VRAM_LIMIT_MIB: 3584" in compose
    assert "CRISPASR_TIMEOUT_SECONDS: 1800" in compose
    assert "crispasr-cache:/var/cache/crispasr" in compose
    assert "jobs-data:/data/jobs" in compose
    assert "MOCK_PIPELINE" in compose
    # torch-era knobs are retired with the NeMo image.
    assert "PYTORCH" not in compose
    assert "HF_HOME" not in compose


def test_env_example_exposes_documented_runtime_configuration() -> None:
    env_example = read(".env.example")

    for name in (
        "CRISPASR_BIN=",
        "GGUF_MODEL=",
        "DIAR_METHOD=",
        "CRISPASR_CACHE_DIR=",
        "CRISPASR_LIB_DIR=",
        "CRISPASR_GPU_BACKEND=",
        "VRAM_LIMIT_MIB=",
        "CRISPASR_TIMEOUT_SECONDS=",
        "CHUNK_SECONDS=",
        "OVERLAP_SECONDS=",
        "MAX_UPLOAD_BYTES=",
        "SPELL_CHECK=",
        "SELF_ID_PATTERNS=",
        "RETENTION_DAYS=",
    ):
        assert f"{name}" in env_example
    assert "DEVICE_DTYPE" not in env_example
    assert "PYTORCH" not in env_example


def test_dockerignore_keeps_model_weights_and_runtime_data_out_of_build_context() -> None:
    dockerignore = read(".dockerignore")

    assert ".env" in dockerignore
    assert "/models" in dockerignore
    assert "/data" in dockerignore


def test_retention_and_disk_env_defaults() -> None:
    """V2 retention knobs (card t_5e6200fc): RETENTION_DAYS is REAL now
    (30-day default), plus the disk-pressure pair. compose passes the same
    defaults through so a .env-less compose run behaves identically."""
    env_example = read(".env.example")
    assert "RETENTION_DAYS=30" in env_example
    assert "DISK_MIN_FREE_MIB=8192" in env_example
    assert "DISK_CHECK_INTERVAL_SECONDS=300" in env_example

    compose = read("docker-compose.yml")
    assert "RETENTION_DAYS: ${RETENTION_DAYS:-30}" in compose
    assert "DISK_MIN_FREE_MIB: ${DISK_MIN_FREE_MIB:-8192}" in compose
    assert "DISK_CHECK_INTERVAL_SECONDS: ${DISK_CHECK_INTERVAL_SECONDS:-300}" in compose
