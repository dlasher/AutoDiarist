#!/usr/bin/env bash
# =============================================================================
# AutoDiarist AMD/Vulkan host device-access helper (packaging card
# t_f82ad22a). Run on the AMD GPU host, once per boot unless --persist is
# used. SAFE TO RE-RUN. This grants /dev/dri + /dev/kfd access to the
# container's baked `app` user (uid 10001) — it changes no files inside the
# repo and requires NO editing of any tracked compose/Dockerfile.
#
# The device-access model mirrors the proven bring-up on the AMD host
# 10.4.10.55 (~/CODE/HERMES-PATTERNS/amd-bringup.sh v3): no ROCm docker
# runtime exists there, group_add is NOT interpolable from .env by compose
# (so per-host gids must never enter the compose file), and the container
# reaches the GPU as uid 10001 through host-side permissions instead.
#
#   PRIMARY : setfacl user-based ACL for uid 10001 on the live nodes.
#             Precise, survives nothing across udev recreations unless
#             --persist installs the matching udev rule.
#   FALLBACK: if `setfacl` is not available on the host, this script says
#             so explicitly and falls back to node mode 0666 (any-user rw)
#             on the same nodes. Broader than the ACL; chosen because the
#             zero-edit contract forbids putting host-specific gids into
#             the tracked compose file. For gid fine-control instead, run
#             docker with --group-add <gid> manually or in an untracked
#             compose override file.
#
# Gid facts are printed for diagnostics ONLY. Never hardcode them: the
# owning group of card0 vs renderD128 and the ids themselves differ per
# host and per driver stack (bring-up observed exactly this confusion —
# card-group membership, NOT video-group membership).
#
# Usage:
#   sudo bash scripts/vulkan-host-setup.sh [--persist]
#     --persist  also install /etc/udev/rules.d/99-autodiarist.rules so
#                the grant survives reboots and GPU re-plug.
# =============================================================================
set -u

CONTAINER_UID=10001          # `app` user baked into the AutoDiarist image
PERSIST="${1:-}"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run as root (sudo bash scripts/vulkan-host-setup.sh [--persist])" >&2
    exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

# --- collect the live device nodes ------------------------------------------
nodes=()
[ -e /dev/kfd ] && nodes+=(/dev/kfd)
for n in /dev/dri/renderD*; do [ -e "$n" ] && nodes+=("$n"); done
for n in /dev/dri/card*; do   [ -e "$n" ] && nodes+=("$n"); done
if [ "${#nodes[@]}" -eq 0 ]; then
    echo "ERROR: no /dev/kfd and no /dev/dri nodes found — is amdgpu loaded (modprobe amdgpu)?" >&2
    exit 1
fi

echo "== AutoDiarist AMD device access (container uid $CONTAINER_UID) =="
for n in "${nodes[@]}"; do
    gid=$(stat -c '%g' "$n"); group=$(stat -c '%G' "$n")
    echo "   $n  group=$group($gid)  mode=$(stat -c '%a' "$n")"
done

# --- PRIMARY path: per-user ACL ---------------------------------------------
if have setfacl; then
    for n in "${nodes[@]}"; do
        setfacl -m "u:$CONTAINER_UID:rw" "$n" \
            && echo "   setfacl u:$CONTAINER_UID:rw $n ... OK" \
            || { echo "   setfacl FAILED on $n"; exit 1; }
    done
    mode=acl
elif have chmod; then
    # --- FALLBACK: no setfacl on this host (said out loud, per contract) ---
    echo "NOTE: 'setfacl' is NOT available on this host." >&2
    echo "NOTE: falling back to node mode 0666 (broader than the ACL path)." >&2
    for n in "${nodes[@]}"; do
        chmod 666 "$n" && echo "   chmod 666 $n ... OK" || { echo "   chmod FAILED on $n"; exit 1; }
    done
    mode=0666
else
    echo "ERROR: neither setfacl nor chmod usable" >&2
    exit 1
fi

# --- --persist: udev rule so udev node recreation keeps the grant -----------
if [ "$PERSIST" = "--persist" ]; then
    rule=/etc/udev/rules.d/99-autodiarist.rules
    if [ "$mode" = acl ] && have setfacl; then
        sf=$(command -v setfacl)
        cat > "$rule" <<EOF
# AutoDiarist (card t_f82ad22a): keep uid $CONTAINER_UID rw on the Vulkan
# nodes after udev recreates them (boot / module reload / GPU re-plug).
KERNEL=="kfd",       SUBSYSTEM=="misc", RUN+="$sf -m u:$CONTAINER_UID:rw /dev/kfd"
KERNEL=="card*",     SUBSYSTEM=="drm",  RUN+="$sf -m u:$CONTAINER_UID:rw /dev/dri/%k"
KERNEL=="renderD*",  SUBSYSTEM=="drm",  RUN+="$sf -m u:$CONTAINER_UID:rw /dev/dri/%k"
EOF
    else
        cat > "$rule" <<'EOF'
# AutoDiarist (card t_f82ad22a): fallback persistence — node mode 0666
# survives udev recreations without setfacl.
KERNEL=="kfd",      SUBSYSTEM=="misc", MODE="0666"
KERNEL=="card*",    SUBSYSTEM=="drm",  MODE="0666"
KERNEL=="renderD*", SUBSYSTEM=="drm",  MODE="0666"
EOF
    fi
    udevadm control --reload-rules
    echo "   installed $rule (grant survives reboot / re-plug)"
fi

echo
echo "Next, from a fresh clone of the repo:"
echo "    docker compose -f docker-compose.amd.yml up --build"
echo "No file edits, no .env — the grant above is the only host-side step,"
echo "and with --persist it is a one-time step for the life of the host."
