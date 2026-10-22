#!/bin/sh
# Container entrypoint (packaging card t_f82ad22a). Identical for both
# GPU_BACKEND build targets; the only backend-aware step is the Vulkan ICD
# selection below, which no-ops on the CUDA image (no /usr/share/vulkan/icd.d).
#
# WHY ICD SELECTION EXISTS: the proven AMD bring-up on 10.4.10.55
# (~/CODE/HERMES-PATTERNS/amd-bringup.sh v3) found that letting the Vulkan
# loader enumerate every ICD json in /usr/share/vulkan/icd.d/ makes the
# engine die at init with vulkan:11 (VK_ERROR_INITIALIZATION_FAILED).
# Pinning VK_ICD_FILENAMES to the single RADV/amdgpu ICD is the fix. The
# glob accepts either name spelling so it can never silently miss the only
# GPU ICD (that miss is what "dies as vulkan:11" looks like).
set -eu

if [ -d /usr/share/vulkan/icd.d ] && [ -z "${VK_ICD_FILENAMES:-}" ]; then
    icd="$(ls /usr/share/vulkan/icd.d 2>/dev/null | grep -Ei 'radeon|amdgpu' | head -n 1 || true)"
    if [ -n "$icd" ]; then
        VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/$icd"
        export VK_ICD_FILENAMES
    fi
fi

cd /app
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
