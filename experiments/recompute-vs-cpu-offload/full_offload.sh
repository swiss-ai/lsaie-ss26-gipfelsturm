#!/bin/bash
set -euo pipefail

# Layer-level CPU activation offload baseline.
# Use MBS overrides to test how far full offload scales before OOM.

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-4}
MBS=${MBS:-1}

export PARTITION=${PARTITION:-debug}
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-mbs${MBS}-full-offload"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--cpu-offloading-num-layers 39"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP"
