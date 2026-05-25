#!/bin/bash
set -euo pipefail

# Recompute only attention; keep the MLP path saved on GPU.
# If this fits and is faster than full recompute, MLP recompute is costly.

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-4}
MBS=${MBS:-1}

export PARTITION=${PARTITION:-debug}
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-mbs${MBS}-selective-core-attn-recompute"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--recompute-granularity selective
    --recompute-modules core_attn"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP"
