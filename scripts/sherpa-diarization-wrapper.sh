#!/bin/sh
# sherpa-onnx offline speaker-diarization wrapper (card t_5a287979).
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
set -eu

SHERPA_REAL="${SHERPA_REAL:-/opt/sherpa/bin/sherpa-onnx-offline-speaker-diarization}"
SHERPA_CLUSTER_THRESHOLD="${SHERPA_CLUSTER_THRESHOLD:-0.35}"
SHERPA_LD_LIBRARY_PATH="${SHERPA_LD_LIBRARY_PATH:-/opt/sherpa/lib}"

LD_LIBRARY_PATH="${SHERPA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

exec "$SHERPA_REAL" \
    --clustering.cluster-threshold="$SHERPA_CLUSTER_THRESHOLD" \
    "$@"
