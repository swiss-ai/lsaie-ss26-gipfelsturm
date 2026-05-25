#!/bin/bash
set -euo pipefail

# Recompute attention plus layernorms, but still keep the MLP path saved.
# Tests whether extra non-MLP memory savings are enough to fit.

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-4}
MBS=${MBS:-1} 

export PARTITION=${PARTITION:-debug}
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-mbs${MBS}-selective-core-attn-layernorm-recompute"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--recompute-granularity selective
    --recompute-modules core_attn layernorm"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP"
