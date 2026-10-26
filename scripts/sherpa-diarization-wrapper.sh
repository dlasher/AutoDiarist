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

# --- diarization thread count (card t_3cb8f97c) ---------------------------
# The sherpa CLI defaults ONE thread for BOTH the segmentation and the
# embedding network. Measured on the NVIDIA host (4 cores, 137.5 s clip):
# the diarization phase runs 58.3 s at 1 thread and 20.5 s at 4 (2.85x),
# speakers=4 at every count -- threading is a performance knob, NOT a
# quality knob. It REGRESSES above the core count (8 threads = 68 s, worse
# than 1), so no constant can be the default: a fixed 4 is right for this
# 4-core box but wrong for every other one, and a fixed 8 would make the
# NVIDIA host 2x SLOWER than today. The default is therefore computed,
# deterministically:
#
#   threads = min( max(1, floor(C / N)), CAP )
#     C    = container core count (nproc, else getconf _NPROCESSORS_ONLN).
#            Neither compose file sets CPU limits (deploy.resources is
#            GPU reservations only), so the container sees ALL host cores:
#            4 on the NVIDIA T400, 12 on the AMD box.
#     N    = MAX_PROCESSING_GLOBAL (concurrent queue workers, card
#            t_dbad2235). Each worker diarizes with its own T threads, so
#            total CPU demand is N x T; C / N is the safe per-job share.
#            Unset, non-integer or < 1 counts as 1 (same parse rule as
#            api/main.py: the queue must never freeze on a bad value).
#     CAP  = 8 by default (the card's sane range 1..8; and total demand
#            stays <= cores at N=1 too), RAISABLE at runtime with
#            SHERPA_MAX_THREADS (planner note 14:45, option (c): the AMD
#            8-vs-10-vs-12 sweep is pending, so a big box can lift the
#            auto ceiling to its own core count without a rebuild). The
#            cap is ALWAYS min(cap, cores): a fixed 8 on a 2-core host
#            must resolve to 2 — never oversubscribe by default.
#            Invalid SHERPA_MAX_THREADS values (0/negative/non-integer/
#            unset) fall back to 8, same parse rule as the other knobs.
#
# With the shipped default N=1 this yields T=min(C, 8): on this 4-core box
# that is the measured optimum, the full 2.85x. At N=3 on the 4-core box
# it yields 1 -- NEVER worse than the
# current single-thread behaviour (the regression floor is a hard product
# requirement; a default that speeds up N=1 but slows down N=3 must not
# ship). Cores unknown -> auto resolves to 1 (identical to today's default;
# NEVER 0 -- 0 threads would hang or fail the CLI).
#
# SHERPA_NUM_THREADS is the operator override: a valid positive integer is
# passed VERBATIM to both networks and takes precedence over the auto
# formula (MAX_PROCESSING_GLOBAL then only matters if the operator also
# raised it); the only clamp is down to the core count (explicit 9999 on 4
# cores -> 4; explicit 12 on 12 cores -> 12, an informed choice the card's
# measured curve already warned about). 0 / negative / non-integer / unset
# = auto. More than 9 digits = absurd -> auto (keeps shell arithmetic safe).
SHERPA_NUM_THREADS="${SHERPA_NUM_THREADS:-}"
SHERPA_MAX_THREADS="${SHERPA_MAX_THREADS:-}"
MAX_PROCESSING_GLOBAL="${MAX_PROCESSING_GLOBAL:-}"

cores=""
if command -v nproc >/dev/null 2>&1; then
    cores=$(nproc 2>/dev/null || true)
fi
case "$cores" in
    ''|*[!0-9]*) cores="" ;;
esac
if [ -z "$cores" ] && command -v getconf >/dev/null 2>&1; then
    cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
fi
case "$cores" in
    ''|*[!0-9]*) cores="" ;;
    *)
        if [ "$cores" -lt 1 ]; then cores=""; fi
        ;;
esac

workers=1
case "$MAX_PROCESSING_GLOBAL" in
    ''|*[!0-9]*) ;;
    *)
        if [ "$MAX_PROCESSING_GLOBAL" -ge 1 ]; then
            workers=$MAX_PROCESSING_GLOBAL
        fi
        ;;
esac

thread_cap=8
case "$SHERPA_MAX_THREADS" in
    ''|*[!0-9]*) ;;
    *)
        if [ "${#SHERPA_MAX_THREADS}" -le 9 ] \
           && [ "$SHERPA_MAX_THREADS" -ge 1 ]; then
            thread_cap=$SHERPA_MAX_THREADS
        fi
        ;;
esac
# The cap can never oversubscribe: min(cap, reported core count).
if [ -n "$cores" ] && [ "$thread_cap" -gt "$cores" ]; then
    thread_cap=$cores
fi
threads=""
requested=""
case "$SHERPA_NUM_THREADS" in
    ''|*[!0-9]*) ;;
    *)
        if [ "${#SHERPA_NUM_THREADS}" -le 9 ] \
           && [ "$SHERPA_NUM_THREADS" -ge 1 ]; then
            requested=$SHERPA_NUM_THREADS
        fi
        ;;
esac

if [ -n "$requested" ]; then
    mode=explicit
    threads=$requested
    if [ -n "$cores" ] && [ "$threads" -gt "$cores" ]; then
        echo "sherpa-wrapper: SHERPA_NUM_THREADS=$requested exceeds the $cores core(s) reported by the container; clamping to $cores" >&2
        threads=$cores
    fi
else
    mode=auto
    if [ -n "$cores" ]; then
        threads=$((cores / workers))
        if [ "$threads" -lt 1 ]; then threads=1; fi
        if [ "$threads" -gt "$thread_cap" ]; then threads=$thread_cap; fi
    else
        threads=1
    fi
fi

# Deterministic and logged, twice: (1) one line to stderr (the card's
# prescribed channel; lands in the engine's stderr file and in the job
# error tail when the run fails), and (2) an append to
# $JOBS_DIR/sherpa-threads.log, because the service DELETES the engine
# stderr temp file on success — without this second record ops could
# never see what the box resolved on a healthy run. Same volume that
# already holds the pre-existing _scheduler.json file. Skipped silently
# when JOBS_DIR is unset or not a writable directory (host-side probes
# and standalone runs must never fail over an ops log).
summary="threads=$threads ($mode: cores=${cores:-unknown} workers=$workers cap=$thread_cap requested=${SHERPA_NUM_THREADS:-unset} max=${SHERPA_MAX_THREADS:-unset})"
echo "sherpa-wrapper: diarization $summary" >&2
if [ -n "${JOBS_DIR:-}" ] && [ -d "${JOBS_DIR:-/nonexistent}" ] \
   && [ -w "${JOBS_DIR:-/nonexistent}" ]; then
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        "sherpa-wrapper: diarization $summary" \
        >> "$JOBS_DIR/sherpa-threads.log" 2>/dev/null || true
fi

if [ "$ceiling" -le 0 ]; then
    exec "$SHERPA_REAL" \
        --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
        --segmentation.num-threads="$threads" \
        --embedding.num-threads="$threads" \
        "$@"
fi

# pass 1: identical command to the pre-ceiling wrapper; stdout captured so
# the label count can be checked, stderr still forwarded live.
pass1=$(mktemp)
trap 'rm -f "$pass1"' EXIT HUP INT TERM
set +e
"$SHERPA_REAL" \
    --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
    --segmentation.num-threads="$threads" \
    --embedding.num-threads="$threads" \
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

# Ceiling re-run pass: same thread flags as pass 1 (the planner note on
# card t_3cb8f97c — the wrapper invokes sherpa TWICE on over-exploding
# inputs and the thread count applies to BOTH passes).
"$SHERPA_REAL" \
    --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
    --clustering.num-clusters="$ceiling" \
    --segmentation.num-threads="$threads" \
    --embedding.num-threads="$threads" \
    "$@"
