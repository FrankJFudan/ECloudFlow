# Distributed execution

Production training targets four NVIDIA H100 80 GB GPUs with Lightning DDP,
BF16 mixed precision, fixed global work, and rank-zero artifact publication.
Local CPU/GPU smoke tests remain useful for wiring but are not throughput
measurements.

Run `ecloudflow doctor --server` before allocation-heavy work. Server mode
requires four visible CUDA devices, BF16-capable compute capabilities, and an
NCCL-enabled PyTorch build. It reports device names, capabilities, PyTorch, and
CUDA runtime versions without initializing a process group.

```bash
ecloudflow doctor --require-gpu --json
export ECLOUDFLOW_DATA_MANIFEST=data/processed/pdbbind/manifest.json
ECLOUDFLOW_RUN_NCCL=1 bash scripts/run_h100_smoke.sh \
  --config experiment=h100_smoke \
  --output runs/h100-smoke
ECLOUDFLOW_RUN_NCCL=1 bash scripts/benchmark_scaling.sh \
  --config experiment=h100_large \
  --output runs/scaling

# Local artifact-schema check only.
ecloudflow benchmark --devices 1 --devices 2 --devices 4 --steps 1 \
  --config experiment=h100_large --output-dir runs/scaling-dry --dry-run
```

The benchmark commands above measure scaling with model-shaped synthetic work.
The separate acceptance command below executes the real sharded DataModule,
Lightning fit, checkpoint publication, and strict resume path:

```bash
bash scripts/acceptance_h100.sh \
  data/processed/pdbbind/manifest.json runs/h100-acceptance
```

The smoke script records Git, environment, configuration, and dataset hashes.
Keep `ECLOUDFLOW_RUN_NCCL=1` set on the server to invoke the four-rank torchrun
path; without it, the scripts deliberately emit simulated dry-run rows. The
benchmark keeps the resolved ECloudFlow architecture and global batch fixed
while varying device count. Non-dry runs execute real joint-model forward,
backward, optimizer, and DDP operations on deterministic model-shaped complex
tensors, then measure examples/second, optimizer steps/second, peak allocated
and reserved memory, speedup, efficiency, model forward calls, and GPU-hours.
These synthetic inputs do not measure chemical validity, sampling NFE, binding,
or docking; those fields remain `null` until a separate generation benchmark
observes them. Dry runs are analytical artifact-schema estimates and cannot be
merged with measured GPU reports.
NCCL scaling keeps the raw one-, two-, and four-process reports in separate
`dev1/`, `dev2/`, and `dev4/` directories and writes a combined root report
after all three jobs succeed. If no manifest is available, the report labels
its data fingerprint as a configuration fallback rather than a dataset hash.

## Correctness rules

Distributed loaders partition sample IDs without overlap and preserve manifest
ordering for deterministic seeds. Rank zero alone writes shared artifacts;
other ranks send summaries through the process group. Checkpoint resume includes
optimizer, EMA, scheduler, RNG, and rank-local loader state. A changed world
size or semantic precision is rejected unless the experiment explicitly starts
fresh.

## Troubleshooting

Verify identical environments and visible devices on every rank. Use Gloo CPU
tests when NCCL is unavailable, but label them as functional checks. A failed
rank must stop the job rather than leaving a partial checkpoint that looks
complete. Compare scaling only after warm-up and with the same data manifest.
