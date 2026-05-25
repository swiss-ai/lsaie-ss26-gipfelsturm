#!/bin/bash
set -euo pipefail

# PP=4 mixed offload + selective recompute (Experiment 3 reference recipe).
# Offload all attention/norm activations to CPU; selectively recompute mlp + layernorm.
# This is the best strategy from Experiment 3 (~43% MFU).

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-1}
PP=${PP:-4}
MBS=${MBS:-1}

export PARTITION=${PARTITION:-debug}
export NVTE_CPU_OFFLOAD_V1=1
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-mbs${MBS}-mixed-offload-recompute"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--fine-grained-activation-offloading
    --offload-modules core_attn qkv_linear attn_proj attn_norm mlp_norm
    --recompute-granularity selective
    --recompute-modules layernorm mlp"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP" "$PP"
