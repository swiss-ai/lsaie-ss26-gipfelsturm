#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOGS = [
    "logs/gipfel-throughput-350m-50s-1n-baseline-2413480.log",
    "logs/gipfel-throughput-350m-50s-1n-flash_attention-2413482.log",
    "logs/gipfel-throughput-350m-50s-1n-unfused_attention-2413484.log",
    "logs/gipfel-throughput-350m-50s-1n-fused_attention-2413483.log",
    "logs/gipfel-throughput-350m-50s-1n-dsa-2413521-unfused.log",
    "logs/gipfel-throughput-350m-50s-1n-dsa-2413510-sparse.log",
    "logs/gipfel-throughput-350m-50s-1n-dsa-2413481-fused.log",
    "logs/gipfel-throughput-350m-50s-1n-gated_delta_net-2418416.log"
]

NAME_MAP = {
    "gipfel-throughput-350m-50s-1n-baseline-2413480.log": "Auto",
    "gipfel-throughput-350m-50s-1n-flash_attention-2413482.log": "Flash",
    "gipfel-throughput-350m-50s-1n-unfused_attention-2413484.log" : "Unfused",
    "gipfel-throughput-350m-50s-1n-fused_attention-2413483.log" : "Fused",
    "gipfel-throughput-350m-50s-1n-dsa-2413521-unfused.log" : "DSA - Unfused",
    "gipfel-throughput-350m-50s-1n-dsa-2413510-sparse.log" : "DSA - Sparse",
    "gipfel-throughput-350m-50s-1n-dsa-2413481-fused.log" : "DSA - Fused",
    "gipfel-throughput-350m-50s-1n-gated_delta_net-2418416.log" : "Gated DeltaNet"
}

OUT = "throughput_per_gpu.png"


LINE_RE = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"
)


def extract_throughput(log_path: str, max_step: int = 50):
    steps = []
    tflops = []

    text = Path(log_path).read_text(errors="ignore")

    for match in LINE_RE.finditer(text):
        step = int(match.group(1))
        throughput = float(match.group(3))

        if 0 <= step <= max_step:
            steps.append(step)
            tflops.append(throughput)

    return steps, tflops


def label_for(log_path: str):
    name = Path(log_path).name
    return NAME_MAP.get(name, name)

def main():
    plt.figure(figsize=(10, 6))

    for log in LOGS:
        steps, tflops = extract_throughput(log)

        if not steps:
            print(f"Warning: no throughput data found in {log}")
            continue

        plt.plot(
            steps,
            tflops,
            marker="o",
            linewidth=1.5,
            label=label_for(log),
        )

    plt.xlabel("Step")
    plt.ylabel("Throughput per GPU (TFLOP/s/GPU)")
    plt.title("Throughput per GPU over Training Steps")

    plt.xlim(0, 52)
    plt.xticks(range(0, 51, 5))

    plt.grid(True, alpha=0.3)

    # Legend on top
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()

    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()