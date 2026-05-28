#!/bin/bash
set -e

scripts=(
    configs/flash_attention_launch.sh
    configs/fused_attention_launch.sh
    configs/unfused_attention_launch.sh
)

for i in "${!scripts[@]}"; do
  s="${scripts[$i]}"

  echo "Starting $s at $(date)"
  bash "$s" "$@"
  echo "Finished $s at $(date)"

  if (( i < ${#scripts[@]} - 1 )); then
    echo "Sleeping 30 minutes..."
    sleep 30m
  fi
done

echo "All done at $(date)"
EOF
