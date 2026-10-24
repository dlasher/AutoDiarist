#!/bin/sh
# sherpa-onnx offline speaker-diarization wrapper (cards t_5a287979,
# t_38e52d89).
#
# The crispasr engine invokes the sherpa CLI with
# --clustering.num-clusters=<N> plus the two model paths, but has NO flag
# for --clustering.cluster-threshold, the knob that actually decides how
# many speakers FastClustering emits (measured on the NV host: threshold
# sweeps move the 249s panel file between 1 and 7 speakers).
#
# The engine accepts an explicit binary via --sherpa-bin (SHERPA_BIN here)
# and forwards all its own args unchanged, so the service injects the
# threshold by wrapping the real binary. Keep the flag name exactly in
# sync with the upstream CLI (k2-fsa/sherpa-onnx,
# csrc/sherpa-onnx-offline-speaker-diarization.cc).
#
# NV-host calibration (reverb-diarization-v1 segmentation + eres2net_base
# embedding, full crispasr engine contract, 2026-10-22):
#   0.35 -> panel 5 speakers (matches the reference count), 1-speaker
#           promo file 3 labels
#   0.40 -> panel 3, promo 2 (no worse than the retired foxnose default)
#   0.45 -> panel collapses to 1
# 0.40 ships as the default: it meets the card's >=3 bar while keeping
# single-speaker output closest to truth. SHERPA_CLUSTER_THRESHOLD is a
# runtime knob -- LOWER it (toward 0.35) to resolve more speakers on
# panels, RAISE it (toward 0.45) to merge aggressively on narrated
# uploads; no rebuild needed.
#
# SPEAKER-COUNT CEILING (card t_38e52d89). DIARIZATION_MAX_SPEAKERS
# (default 8; product owner: meetings with >10 participants are not
# expected) bounds the label count. The 118-min NASA SEWP Q&A file
# explodes to 100 distinct speakers through the auto-threshold path
# (reference: 6-7), while the 249s panel (3) and the 137s promo (1-2)
# are already compliant. Measured design constraints, from two sweeps:
#
#   * The CLI's own --help: "num-clusters ... If greater than 0, then
#     cluster threshold is IGNORED" -- and the engine ALWAYS passes its
#     own --clustering.num-clusters=0. APPENDING our flag is therefore a
#     SILENT NO-OP (sweep 1: byte-identical output for every N); the
#     engine's flag must be REMOVED from "$@" before injecting (AC7).
#   * A bare fixed num-clusters OVER-SPLITS cleanly-separated audio
#     (sweep 2: panel 2 -> 4 labels at NC=8; forcing k clusters splits
#     real speakers when the true count is lower). So the ceiling is
#     CONDITIONAL: pass 1 runs exactly the historical auto-threshold
#     behaviour with stdout captured; only when its distinct
#     speaker_XX label count EXCEEDS the ceiling does the wrapper
#     re-run with the stripped + injected value. Compliant output
#     (panel, promo, marklex) is replayed byte-identical; nothing
#     re-runs and the threshold default 0.40 is NOT retuned (out of
#     scope).
#   * DIARIZATION_MAX_SPEAKERS=0 disables the ceiling entirely: the
#     wrapper execs the pre-card command, unbuffered. Invalid/negative
#     values are treated as 0 (fail open to historical behaviour).
#
# Cost: only over-exploding inputs pay a second clustering pass
# (segmentation + embedding + clustering; no ASR re-decode -- the engine
# invokes this wrapper for its diarization timeline only). Ceiling-pass
# stderr is forwarded like pass 1's (CLI config/banner lines may appear
# twice on capped runs; transcripts are unaffected).
set -eu

SHERPA_REAL="${SHERPA_REAL:-/opt/sherpa/bin/sherpa-onnx-offline-speaker-diarization}"
SHERPA_CLUSTER_THRESHOLD="${SHERPA_CLUSTER_THRESHOLD:-0.40}"
SHERPA_LD_LIBRARY_PATH="${SHERPA_LD_LIBRARY_PATH:-/opt/sherpa/lib}"
DIARIZATION_MAX_SPEAKERS="${DIARIZATION_MAX_SPEAKERS:-8}"

LD_LIBRARY_PATH="${SHERPA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

# Non-negative integer or the ceiling is off (historical behaviour).
case "$DIARIZATION_MAX_SPEAKERS" in
    ''|*[!0-9]*) ceiling=0 ;;
    *) ceiling=$DIARIZATION_MAX_SPEAKERS ;;
esac

if [ "$ceiling" -le 0 ]; then
    exec "$SHERPA_REAL" \
        --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
        "$@"
fi

# pass 1: identical command to the pre-ceiling wrapper; stdout captured so
# the label count can be checked, stderr still forwarded live.
pass1=$(mktemp)
trap 'rm -f "$pass1"' EXIT HUP INT TERM
set +e
"$SHERPA_REAL" \
    --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
    "$@" >"$pass1"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
    cat "$pass1"
    exit "$rc"
fi

# Strict segment-line match ("<start> -- <end> speaker_XX"): the captured
# stdout also contains the CLI's config banner, which echoes the model
# paths — the real embedding model name (3dspeaker_...3dspeaker_16k.onnx)
# embeds "speaker_16" and would be counted as a phantom label by a loose
# substring grep (measured on the promo fixture, VM 2026-10-23).
count=$(awk '$2 == "--" && $NF ~ /^speaker_[0-9]+$/ { print $NF }' "$pass1" \
        | sort -u | wc -l)

if [ "$count" -le "$ceiling" ]; then
    cat "$pass1"
    exit 0
fi

# Ceiling pass. AC7: the engine always passes its own
# --clustering.num-clusters (measured: =0); leaving it in "$@" makes the
# injection a no-op, so REMOVE it (both --flag=v and --flag v shapes) by
# rebuilding "$@" in place: at every step the positional parameters are
# [unconsumed original args][kept args], one original arg is processed per
# iteration (n guards the boundary where kept args begin), and kept args
# rotate onto the tail.
n=$#
while [ "$n" -gt 0 ]; do
    a=$1
    shift
    n=$((n - 1))
    case $a in
        --clustering.num-clusters)
            # value form: consume the following arg as its value too
            if [ "$n" -gt 0 ]; then
                shift
                n=$((n - 1))
            fi
            ;;
        --clustering.num-clusters=*) : ;;
        *) set -- "$@" "$a" ;;
    esac
done

"$SHERPA_REAL" \
    --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
    --clustering.num-clusters="$ceiling" \
    "$@"
