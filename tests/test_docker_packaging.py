"""Docker packaging asserts the CrispASR single-Dockerfile / two-hardware
contract (cards t_423d1ee4 CUDA path, t_5a287979 sherpa diarization stack,
t_f82ad22a packaging: one Dockerfile with a GPU_BACKEND build ARG plus a
per-hardware compose file, zero-edit `docker compose up` on both).

Backend per spike t_7e17589e + adapter t_4d5f7d08: the crispasr prebuilt
(v0.8.43) pinned per target — cuda13 + apt CUDA-13 runtime libs on NVIDIA,
vulkan + libvulkan1/mesa-vulkan-drivers on AMD — the canonical cstr Q4_K
GGUF, and the env contract consumed by pipeline/transcribe.py. The
NeMo/torch image path is retired and must not reappear. The AMD side must
NEVER return to a generated Dockerfile.amd: the old operator script's
foxnose-pinned shape silently regressed t_5a287979.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]

# Spike-verified digests (t_7e17589e REPORT.md; re-checked against the GitHub
# release asset digest and the cstr HuggingFace LFS OID).
CRISPASR_CUDA13_SHA256 = (
    "fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54"
)
# Card t_f82ad22a AC2: independently sha-digested against the v0.8.43
# GitHub release asset for the Vulkan target.
CRISPASR_VULKAN_SHA256 = (
    "d5c676e023df6fac61971dcef06fb7cce22ff3de95367f8e01efb8743c11ca46"
)
GGUF_Q4K_SHA256 = (
    "22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd"
)
GGUF_NAME = "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"

CUDA_COMPOSE = "docker-compose.yml"
AMD_COMPOSE = "docker-compose.amd.yml"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def compose_doc(name: str) -> dict:
    return yaml.safe_load(read(name))


def compose_env_keys(name: str) -> set[str]:
    services = compose_doc(name)["services"]
    assert list(services) == ["autodiarist"], f"{name}: unexpected services"
    return set(services["autodiarist"].get("environment") or {})


def compose_code(name: str) -> str:
    """File content without comment noise — contract asserts hit code."""
    return "\n".join(
        line for line in read(name).splitlines() if not line.lstrip().startswith("#")
    )


# --- one Dockerfile, two hardware targets (AC2/AC4/AC5) ----------------------


def test_one_dockerfile_only_no_per_backend_copies() -> None:
    """AC4: the app layer is ONE source of truth. A forked
    Dockerfile.amd/.cuda is exactly how the operator script silently
    regressed the sherpa fix (foxnose defaults baked into a second
    generated Dockerfile) — such files must not exist in-repo."""
    assert (ROOT / "Dockerfile").is_file()
    for forbidden in ("Dockerfile.amd", "Dockerfile.cuda", "Dockerfile.vulkan"):
        assert not (ROOT / forbidden).exists(), forbidden


def test_dockerfile_fetches_pinned_crispasr_per_backend_tarballs() -> None:
    dockerfile = read("Dockerfile")

    assert (
        "github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/"
        "crispasr-linux-x86_64-cuda13.tar.gz" in dockerfile
    )
    assert (
        "github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/"
        "crispasr-linux-x86_64-vulkan.tar.gz" in dockerfile
    )
    assert CRISPASR_CUDA13_SHA256 in dockerfile
    assert CRISPASR_VULKAN_SHA256 in dockerfile
    assert "sha256sum -c" in dockerfile
    # ONE case on the build ARG resolves both digests — a single engine
    # layer, so the two targets cannot diverge into two downloads.
    assert 'case "$GPU_BACKEND"' in dockerfile


def test_dockerfile_gpu_backend_arg_selects_runtime_libs_both_targets() -> None:
    dockerfile = read("Dockerfile")

    # CUDA path unchanged in substance (T400/sm_75, CUDA-13 apt libs).
    # Tarball does NOT bundle libcudart/libcublas; without them crispasr
    # silently drops to CPU (~11x slower). Pinned via apt, dir exported.
    assert "ARG GPU_BACKEND=cuda" in dockerfile
    assert "cuda-cudart-13-0" in dockerfile
    assert "libcublas-13-0" in dockerfile
    assert "/usr/local/cuda-13.0/targets/x86_64-linux/lib" in dockerfile
    # AMD/Vulkan path: libvulkan1 loader + Mesa RADV — hardware Vulkan on
    # amdgpu, the pairing proven on 10.4.10.55 by amd-bringup v3.
    assert "libvulkan1" in dockerfile
    assert "mesa-vulkan-drivers" in dockerfile
    code = "\n".join(l for l in dockerfile.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "rocm" not in code.lower()
    # an unparseable backend fails the build loudly.
    assert "must be 'cuda' or 'vulkan'" in dockerfile


def test_dockerfile_backend_env_is_self_describing_from_the_build_arg() -> None:
    assert "CRISPASR_GPU_BACKEND=${GPU_BACKEND}" in read("Dockerfile")


def test_app_layer_is_identical_across_backends() -> None:
    """AC4/AC5: GPU_BACKEND may appear ONLY in comments, the ARG line, the
    ENV default expansion, and the two backend-specific RUN layers (apt
    runtime libs + engine tarball). From the GGUF/sherpa layers onward the
    Dockerfile is one shared, backend-independent app layer — the two
    build targets literally run the same instructions."""
    lines = read("Dockerfile").splitlines()

    from_layer = next(i for i, l in enumerate(lines)
                      if "cstr/nemotron-3.5-asr-streaming-0.6b-GGUF" in l)
    tail = [l for l in lines[from_layer:]
            if "GPU_BACKEND" in l and not l.lstrip().startswith("#")]
    assert tail == [], f"app layer must not branch on GPU_BACKEND: {tail}"

    allowed = re.compile(
        r"ARG GPU_BACKEND=cuda"
        r"|CRISPASR_GPU_BACKEND=\$\{GPU_BACKEND\}"
        r"|\bcase \"\$GPU_BACKEND\""
        r"|echo \"FATAL: GPU_BACKEND must be"  # loud-fail guard in same cases
    )
    for line in lines:
        if "GPU_BACKEND" in line and not line.lstrip().startswith("#"):
            assert allowed.search(line), f"unexpected GPU_BACKEND use: {line}"


def test_dockerfile_sherpa_stack_shared_once_both_compose_files_match() -> None:
    """The single Dockerfile ships the sherpa stack exactly once; BOTH
    compose files pin DIAR_METHOD=sherpa, so neither backend can silently
    regress to foxnose (the operator-script failure class, planner note)."""
    dockerfile = read("Dockerfile")
    assert dockerfile.count("COPY scripts/sherpa-diarization-wrapper.sh") == 1
    assert dockerfile.count("SHERPA_CLUSTER_THRESHOLD=0.40") == 1
    for name in (CUDA_COMPOSE, AMD_COMPOSE):
        assert "DIAR_METHOD: sherpa" in read(name)


# --- shared image contract (CUDA behavior preserved) -------------------------


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
    # uid 10001 `app` user on BOTH targets (planner AC3 correction).
    assert "USER app" in dockerfile
    assert "--uid 10001" in dockerfile
    # the uvicorn command moved verbatim into the baked entrypoint script
    # (test_dockerfile_wires_entrypoint_with_vulkan_icd_detection).
    # pipeline/transcribe.py env contract (adapter t_4d5f7d08). Backend
    # is NO LONGER a baked-in "cuda" literal: the image takes it from the
    # GPU_BACKEND ARG (default cuda) and each compose file pins the
    # literal for its hardware (see compose tests).
    for fragment in (
        "CRISPASR_BIN=/opt/crispasr/crispasr",
        f"GGUF_MODEL=/opt/models/{GGUF_NAME}",
        "CRISPASR_CACHE_DIR=/var/cache/crispasr",
        "CRISPASR_LIB_DIR=/usr/local/cuda-13.0/targets/x86_64-linux/lib",
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
    # build-time wrapper self-check (planner review 3): proves LD_LIBRARY_PATH
    # resolution and the injected flag against the vendored CLI at build time
    assert "/opt/sherpa/sherpa-diarization-wrapper --help" in dockerfile
    assert "clustering.cluster-threshold" in dockerfile
    # include/ cleanup targets the extraction dir, not /tmp (planner review 1)
    assert "rm -rf /opt/sherpa/include" in dockerfile
    assert "SHERPA_BIN=/opt/sherpa/sherpa-diarization-wrapper" in dockerfile
    assert "SHERPA_CLUSTER_THRESHOLD=0.40" in dockerfile
    # AC5: the diarization layers run on CPU onnxruntime -- they must not
    # reference the CUDA engine assets, so the Vulkan build-arg variant
    # (t_f82ad22a) shares them unchanged.
    diar_block = dockerfile[dockerfile.index("--- diarization stack"):]
    assert "crispasr-linux" not in diar_block
    assert "cuda-cudart" not in diar_block
    assert "libcublas" not in diar_block


def test_dockerfile_wires_entrypoint_with_vulkan_icd_detection() -> None:
    """AC3: the container entrypoint auto-selects the AMD Vulkan ICD.
    Selecting ALL icd.d jsons made the engine die at init with vulkan:11
    on the proven bring-up; the glob is restricted to *radeon*|*amdgpu*.
    The CUDA image has no /usr/share/vulkan/icd.d — the step no-ops."""
    dockerfile = read("Dockerfile")
    assert "COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]' in dockerfile
    assert "CMD [" not in dockerfile  # replaced; the inline CMD must not return

    entry = read("scripts/docker-entrypoint.sh")
    assert "VK_ICD_FILENAMES" in entry
    assert re.search(r"grep -Ei 'radeon\|amdgpu'", entry)
    assert "exec uvicorn api.main:app" in entry
    # never select all ICDs
    assert not re.search(r"icd\.d/\*\.?json?", entry)


# --- per-hardware compose files ----------------------------------------------


def test_compose_cuda_target_pins_backend_and_nvidia_reservation() -> None:
    compose = read(CUDA_COMPOSE)

    assert "GPU_BACKEND: cuda" in compose
    assert "driver: nvidia" in compose
    assert "count: all" in compose
    assert "CRISPASR_GPU_BACKEND: cuda" in compose
    # pinned literal, never ${VAR:-default} (silent CPU-fallback rule).
    assert "CRISPASR_GPU_BACKEND: ${" not in compose
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
    # the CUDA file keeps the nvidia reservation ONLY — raw device nodes
    # are the AMD block.
    assert "/dev/dri" not in compose


def test_compose_amd_target_vulkan_pinned_and_acl_device_shape() -> None:
    """AC3/AC7: static AMD file, pinned literals, proven device shape —
    mirrors amd-bringup v3 (the host's working ollama config)."""
    code = compose_code(AMD_COMPOSE)
    svc = compose_doc(AMD_COMPOSE)["services"]["autodiarist"]

    assert "GPU_BACKEND: vulkan" in code
    assert "CRISPASR_GPU_BACKEND: vulkan" in code
    assert "DIAR_METHOD: sherpa" in code
    assert "CRISPASR_GPU_BACKEND: ${" not in code
    assert 'CRISPASR_LIB_DIR: ""' in code
    # device access = direct nodes...
    assert svc["devices"] == ["/dev/dri", "/dev/kfd"]
    # ...via host-side permissions for uid 10001: NO group_add, NO
    # privileged, NO ROCm docker runtime, NO nvidia device_requests, NO
    # ENV{} escape hatch — none of these are .env-interpolable or
    # host-portable, and specific host gids NEVER enter a tracked file.
    for forbidden in ("group_add", "privileged", "rocm", "driver: nvidia", "ENV{"):
        assert forbidden not in code, forbidden
    assert svc["image"] == "autodiarist:local-amd"


def test_neither_compose_file_hard_requires_dotenv() -> None:
    """AC10: fresh clone + zero edits. env_file is gone from BOTH files
    (compose errors on a missing env_file target — that was the blocker),
    and no interpolation is bare: every ${VAR} carries an inline default,
    so `docker compose up` resolves with no .env present."""
    for name in (CUDA_COMPOSE, AMD_COMPOSE):
        raw = compose_code(name)  # comments may EXPLAIN the removal
        assert "env_file" not in raw, name
        assert "env_file" not in compose_doc(name)["services"]["autodiarist"], name
        bare = re.findall(r"\$\{(?![A-Za-z_][A-Za-z0-9_]*:-)[A-Za-z_][A-Za-z0-9_]*\}", raw)
        assert not bare, f"{name}: interpolation without default: {bare}"


def test_compose_env_keys_are_actually_read_by_the_service() -> None:
    """Planner review (t_f82ad22a): a compose env key the service never
    reads is a SILENT config regression — e.g. invented
    RETENTION_UNPROCESSED_DAYS/MIN_FREE_DISK_MB names vs the real
    RETENTION_DAYS/DISK_MIN_FREE_MIB api/main.py reads, or a compose
    SESSION_HEADER default overriding the API's. Cross-check BOTH
    environment blocks against names the code actually consumes: quoted
    literals in api//pipeline sources, plus shell ${NAME...} references
    in the scripts that are baked into the image."""
    py = "\n".join(
        p.read_text(encoding="utf-8")
        for p in list(ROOT.glob("api/*.py")) + list(ROOT.glob("pipeline/*.py"))
    )
    quoted = set(re.findall(r"[\"']([A-Z][A-Z0-9_]{2,})[\"']", py))
    scripts = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob("scripts/*.sh"))
    shell_refs = set(re.findall(r"\$\{([A-Z][A-Z0-9_]{2,})", scripts))
    consumed = quoted | shell_refs
    for name in (CUDA_COMPOSE, AMD_COMPOSE):
        unread = compose_env_keys(name) - consumed
        assert not unread, f"{name} sets env the service never reads: {sorted(unread)}"


def test_fresh_clone_yaml_shape() -> None:
    """Both files parse as v2 compose with exactly one service, the same
    build context, port shape and restart policy — the user only picks
    the file that matches their hardware."""
    for name in (CUDA_COMPOSE, AMD_COMPOSE):
        doc = compose_doc(name)
        assert set(doc) >= {"services", "volumes"}, name
        assert list(doc["services"]) == ["autodiarist"], name
        svc = doc["services"]["autodiarist"]
        assert svc["build"]["context"] == ".", name
        assert svc["ports"] == ["${HOST_PORT:-8000}:8000"], name
        assert svc["restart"] == "unless-stopped", name


# --- retired paths must stay retired -----------------------------------------


def test_hip_stays_unwired_and_vulkan_is_the_amd_target() -> None:
    """The -hip/ROCm asset has NO service-side backend name (the adapter
    allow-list is cpu/cuda/metal/vulkan) — it must not enter the pack.
    Vulkan is the AMD target (t_f82ad22a), and the CUDA asset pin is
    untouched."""
    compose_all = read(CUDA_COMPOSE) + read(AMD_COMPOSE)
    assert "crispasr-linux-x86_64-hip" not in compose_all
    assert "crispasr-linux-x86_64-hip" not in read("Dockerfile")
    # word-boundary: "ship"/"chip" must not trip this; the hip ASSET nor
    # the bare backend name may enter the pack.
    assert re.search(r"\bhip\b", compose_all + read("Dockerfile"), re.IGNORECASE) is None
    assert "crispasr-linux-x86_64-cuda13.tar.gz" in read("Dockerfile")


def test_pipeline_never_advertises_a_backend_its_image_cannot_run() -> None:
    import pipeline.transcribe as tr

    assert "vulkan" in tr.GPU_BACKENDS  # the adapter has always accepted it
    assert tr.BACKEND_DISPLAY_NAME == "CrispASR nemotron Q4_K"
    assert tr.GPU_BACKEND_DEFAULT == "cuda"
    # every backend the pack offers is buildable from THIS Dockerfile.
    dockerfile = read("Dockerfile")
    assert "cuda" in dockerfile and "vulkan" in dockerfile


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
    # t_f82ad22a: .env is OPTIONAL now — a fresh clone needs none.
    assert "needs NO .env" in env_example
    # host-specific gids never live in any tracked file (ACL-primary decision)
    assert "HOST_VIDEO_GID" not in env_example


def test_dockerignore_keeps_model_weights_and_runtime_data_out_of_build_context() -> None:
    dockerignore = read(".dockerignore")

    assert ".env" in dockerignore
    assert "/models" in dockerignore
    assert "/data" in dockerignore


def test_retention_and_disk_env_defaults() -> None:
    """V2 retention knobs (card t_5e6200fc): RETENTION_DAYS is REAL now
    (30-day default), plus the disk-pressure pair. compose passes the same
    defaults through so a .env-less compose run behaves identically. Both
    compose files — the AMD mirror must not silently drop the pair."""
    env_example = read(".env.example")
    assert "RETENTION_DAYS=30" in env_example
    assert "DISK_MIN_FREE_MIB=8192" in env_example
    assert "DISK_CHECK_INTERVAL_SECONDS=300" in env_example

    for name in (CUDA_COMPOSE, AMD_COMPOSE):
        compose = read(name)
        assert "RETENTION_DAYS: ${RETENTION_DAYS:-30}" in compose
        assert "DISK_MIN_FREE_MIB: ${DISK_MIN_FREE_MIB:-8192}" in compose
        assert "DISK_CHECK_INTERVAL_SECONDS: ${DISK_CHECK_INTERVAL_SECONDS:-300}" in compose
