# Gipfelsturm

*Gipfelsturm* (German: "summit attempt"), a race to *peak performance*. Inspired by nanoGPT/nanochat which are educational single-node setups, Gipfelsturm focuses on distributed LLM training on production-grade infrastructure. We use [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), the de facto industry standard for distributed LLM training, running on the [CSCS Alps supercomputer](https://arxiv.org/abs/2507.02404) with [GH200 compute nodes connected via Slingshot-11](https://arxiv.org/abs/2408.11556).

There are two dimensions on which you can improve. Your team should focus on at least one.

### Challenge 1: improve loss or compare with alternative method given fixed time

Compute the empirical throughput of your model and batch size. Then compute the number of tokens you can process in theory in a 30 min, 1 hour, and 2 hour time window. This defines the number of steps your model/batch size combination should be trained. Model size, learning rate, schedule, batch size, and training recipe are all your choice. The only constraint is the clock on the wall. Aim for lowest eval loss.

| Theoretical wall-clock time | GPU-hours (at 32 GPUs) | Natural Model Scale |
|-----------------------------|------------------------|---------------------|
| 30 min | 16 | 125m – 3b |
| 1 hour | 32 | 3b – 8b |
| 2 hours | 64 | 8b+ |

### Challenge 2: maximum throughput

Achieve the highest tokens/sec/GPU for a given model size on up to 8 nodes (32 GPUs).

| Model-Parallelism | Model Size | Systems Challenge |
|--------------------|------------|-------------------|
| Single-GPU | ~8B | No model parallelism (TP=1, PP=1) |
| Single-Node | ~32B | Intra-node tensor parallelism (TP=4) |
| Multi-Node | ~140B | Pipeline parallelism across nodes (TP=4, PP=4) |

## Setup

**1. Configure your paths:**

```bash
cp config.sh.example config.sh
# Edit config.sh: set WORKDIR (your scratch directory), SBATCH_ACCOUNT (your SLURM account), and WANDB_API_KEY (optional)
source config.sh
```

`config.sh` is git-ignored; the example file is committed as a template.

**2. Initialize the Megatron-LM submodule:**

The `Megatron-LM/` directory is empty after cloning. Remove any leftover files and initialize:

```bash
rm -rf Megatron-LM
git submodule update --init
```

**3. Set up the EDF container environment:**

Copy [`alps3.toml`](alps3.toml) to `~/.edf/` and update `workdir` to your home directory:

```bash
mkdir -p ~/.edf
sed "s|workdir = .*|workdir = \"$HOME\"|" alps3.toml > ~/.edf/alps3.toml
```

**4. Verify the infrastructure:**

```bash
sbatch test-infra.sbatch
```

**5. Benchmark Megatron-LM training throughput:**

Run a `125m` model benchmark with 50 steps on a single node:

```bash
./launch.sh throughput 125m 50 1
```

## Container Image

We use the **alps3** extended image based on NGC PyTorch 26.01-py3:

```
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3
```

Includes: NCCL 2.29.3-1 (patched), libfabric 2.5.0a1, OpenMPI 5.0.9, nvshmem 3.4.5-0.

See [Alps Extended Images](https://docs.cscs.ch/software/alps-extended-images/) for details. A working EDF environment is provided in [`alps3.toml`](alps3.toml) (copy to `~/.edf/` on Clariden).

To verify the setup, run the infrastructure test (`test-infra.sbatch`) which benchmarks NCCL all-reduce throughput across message sizes from 128 MB to 16 GB. Expected results on 4x GH200 nodes: ~340 GB/s bus bandwidth intra-node (NVLink) and ~93 GB/s inter-node (Slingshot-11), both well within the theoretical hardware ceiling (450 GB/s NVLink-C2C per direction, 100 GB/s Slingshot per node).

## Dataset

[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), 400B tokens, pre-tokenized with GPT-2 tokenizer. Using GPT-2 vocabulary allows direct comparison with nanoGPT baselines.

We use the `climbmix_small` subset (~49 GB, ~6-12B tokens, 100 Parquet shards). The data is already converted to Megatron's binary format (`.bin`/`.idx`) and stored on capstor:

```
/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small.{bin,idx}
```

The raw Parquet shards are at:

```
/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small
```

To re-download or re-convert, see `data/download_climbmix.sh` and `data/convert_data.sbatch`.

## Training

We use the GPT-2 BPE tokenizer (`data/gpt2-vocab.json`, `data/gpt2-merges.txt`) for direct comparability with nanoGPT/nanochat baselines. All baselines are transformer-style models (RoPE, GQA, SwiGLU, RMSNorm).

Training runs on the [Swiss AI Initiative's](https://swiss-ai.org) partition on Alps called Clariden, which uses SLURM.

All training is launched via `launch.sh <mode> <model_size> [steps] [nodes]`. The launcher generates a self-contained SLURM script in `logs/` for reproducibility and submits it. Model sizes: 125m, 350m, 760m, 1.5b, 3b, 8b. Nodes default to 4 (max 8).

Launch scripts can be regenerated from `launch.template.sh` with `launch-config.toml`:

```bash
./generate_launch.py launch-config.toml
```

By default this writes `launch-config_launch.sh`. Use `--output launch.sh` if you want to overwrite the canonical launcher name. The TOML sections map directly to launcher argument groups such as `TRANSFORMER_ENGINE_ARGS` or `TRAINING_ARGS`. Set a flag to `true` for a boolean switch, to a string/number for `--flag value`, to an array for multi-value flags, or to `false` to remove a flag that already exists in the template.

**Throughput** mode runs 50 steps (by default) to measure tokens/sec/GPU:

```bash
./launch.sh throughput 760m           # 50 steps, 4 nodes
./launch.sh throughput 8b 50 1        # 50 steps, 1 node
```

**Train** mode runs a specified number of steps with W&B and Tensorboard logging:

```bash
./launch.sh train 760m 5000           # 5000 steps, 4 nodes
./launch.sh train 1.5b 3000 8         # 3000 steps, 8 nodes
```

**Single-GPU throughput baselines** (50 steps, SEQ_LEN=4096, TP=1, PP=1):

| Model | MBS | tok/s/GPU |
|-------|-----|-----------|
| 125m | 16 | 54,671 |
| 350m | 8 | 62,711 |
| 760m | 6 | 74,994 |
| 1.5b | 4 | 34,054 |
| 3b | 4 | 19,842 |
| 8b | 2 | 10,882 |


### Logging

- **W&B**: Enabled automatically if `WANDB_API_KEY` is set in `config.sh`, or in your shell environment on Clariden. Add `export WANDB_API_KEY=<your-key>` to your `~/.bashrc`.
- **Tensorboard**: Written to `/iopsstor/scratch/cscs/$USER/gipfelsturm/<project>/<exp>/tensorboard/`
- **Checkpoints**: Saved to `/iopsstor/scratch/cscs/$USER/gipfelsturm/<project>/<exp>/checkpoints/` (`torch` format). Currently disabled due to a [known SIGSEGV bug](https://github.com/NVIDIA/Megatron-LM/issues/1861) in checkpoint saving on GH200/ARM64. Note: iopsstor scratch has a three-week deletion policy.

## Megatron-LM Patches

Megatron-LM is included as a git submodule pinned to a specific release. To keep the submodule clean and upgradeable, local modifications are managed as patch files in the `patches/` directory. The submit scripts apply all patches automatically before training via `git apply`. Each patch should be as isolated as possible, targeting a single concern so it can be reviewed, applied, or dropped independently.

### How patches work

A patch is a text file containing a diff with a comment header. The header documents what the patch does, why, where in the code, and how to locate the relevant code if line numbers shift in a future Megatron version. This makes patches self-contained and maintainable.

**Applying patches** (done automatically by `launch.sh`):

```bash
cd Megatron-LM
git apply ../patches/*.patch
```

**Verifying a patch applies cleanly:**

```bash
cd Megatron-LM
git apply --check ../patches/0001-log-tokens-per-sec-to-wandb.patch
```

### Creating a new patch (walkthrough)

As an example, here is how the `0001-log-tokens-per-sec-to-wandb.patch` was created.

**1. Make your changes inside the submodule:**

```bash
cd Megatron-LM
# edit megatron/training/training.py to add tokens/sec/GPU logging
```

**2. Generate the patch:**

```bash
git diff > ../patches/0001-log-tokens-per-sec-to-wandb.patch
```

**3. Revert the submodule** so it stays clean (this only affects `Megatron-LM/`, not the parent repo):

```bash
git checkout -- .
cd ..
```

**4. Add a comment header** to the top of the patch file (before the `diff --git` line). The header uses `#` comment lines which `git apply` ignores. It should explain:

- What problem the patch solves
- What the patch does (which file, which function)
- How to find the right location if line numbers change in a future version

For example, the tokens/sec patch header looks like:

```
# Patch: Log tokens/sec/GPU to stdout, TensorBoard, and W&B
#
# Problem:
#   Megatron-LM logs throughput as TFLOP/s/GPU but does not log tokens/sec/GPU.
#   For an efficiency challenge where wall-clock token throughput matters,
#   tokens/sec/GPU is the more actionable metric.
#
# What this patch does:
#   Adds a `tokens_per_sec_per_gpu` metric in megatron/training/training.py,
#   function training_log(), inside the `if args.log_throughput:` block.
#
# How to find the right location in a future version:
#   Search for `log_throughput` in training.py. Find the block that computes
#   `throughput = num_floating_point_operations(...)` and logs it to
#   writer/wandb_writer. Add the tokens_per_sec_per_gpu computation and
#   logging calls in the same block.
#
# Applied to: Megatron-LM core_v0.16.1
```

**5. Verify and commit** the patch file (not the submodule changes):

```bash
cd Megatron-LM
git apply --check ../patches/0001-log-tokens-per-sec-to-wandb.patch
cd ..
git add patches/0001-log-tokens-per-sec-to-wandb.patch
git commit -m "Add tokens/sec/GPU logging patch"
```

### Upgrading Megatron-LM

As we upgrade Megatron-LM to a new release version, we will attempt to apply all existing patches. If a patch fails, read its comment header to understand the intent, find the equivalent location in the new code, and update the patch accordingly.

### Current patches

| Patch | Description |
|-------|-------------|
| `0001-log-tokens-per-sec-to-wandb.patch` | Logs tokens/sec/GPU to stdout, TensorBoard, and W&B |

## References

- [Alps Supercomputer](https://arxiv.org/abs/2507.02404): system overview
- [GH200 Compute Nodes and Slingshot-11](https://arxiv.org/abs/2408.11556): node architecture and network details
- [CSCS Documentation](https://docs.cscs.ch/): Alps/Clariden cluster docs (SLURM, containers, networking, storage)

## Dependencies

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) `core_v0.16.1` (included as git submodule)
