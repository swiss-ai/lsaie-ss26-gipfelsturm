#!/bin/bash
set -euo pipefail

# Offload selected attention activations to CPU without recompute.
# MLP activations should stay on GPU if this configuration fits.

MODEL=${MODEL:-22b}
NODES=${NODES:-1}
STEPS=${STEPS:-15}
TP=${TP:-4}
MBS=${MBS:-1}

export PARTITION=${PARTITION:-debug}
export NVTE_CPU_OFFLOAD_V1=1
export MBS
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-mbs${MBS}-fine-offload-core-qkv-no-recompute"
export PROJECT_NAME=${PROJECT_NAME:-"lsaie-cpu-activation-offloading-understanding"}
export EXTRA_ARGS="--fine-grained-activation-offloading
    --offload-modules core_attn qkv_linear"

$(dirname "$0")/launch.sh throughput "$MODEL" "$STEPS" "$NODES" "$TP"
