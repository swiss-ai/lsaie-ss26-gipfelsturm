#!/bin/bash
set -euo pipefail

# Recompute only the dense MLP path: fc1 -> activation -> fc2.
# If this is close to full recompute, MLP dominates recompute overhead.

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-4}
MBS=${MBS:-1}

export PARTITION=${PARTITION:-debug}
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-mbs${MBS}-selective-mlp-recompute"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--recompute-granularity selective
    --recompute-modules mlp"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP"
