# Distributed execution

Production training targets four NVIDIA H100 80 GB GPUs with Lightning DDP,
BF16 mixed precision, fixed global work, and rank-zero artifact publication.
Local CPU/GPU smoke tests remain useful for wiring but are not throughput
measurements.

```bash
ecloudflow doctor --require-gpu --json
bash scripts/run_h100_smoke.sh --config experiment=h100_smoke \
  --output runs/h100-smoke
bash scripts/benchmark_scaling.sh --config experiment=h100_large \
  --output runs/scaling
ecloudflow benchmark --devices 1 --devices 2 --devices 4 --steps 100 \
  --config experiment=h100_large --output-dir runs/scaling
```

The smoke script records Git, environment, configuration, and dataset hashes.
Set `ECLOUDFLOW_RUN_NCCL=1` on the server to invoke the four-rank torchrun
path. The benchmark keeps model and global batch fixed while varying device
count, measuring examples/second, optimizer steps/second, peak allocated and
reserved memory, speedup, efficiency, NFE, valid yield, and GPU-hours.
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
