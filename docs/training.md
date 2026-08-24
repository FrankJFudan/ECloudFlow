# Training workflow

Training is a four-stage curriculum. Stage 1 (`electron_tokenizer`) learns
radial-spherical reconstruction and latent irreps from genuine QM fields.
Stage 2 (`ligand_pretrain`) learns graph and pose paths without pocket affinity
supervision. Stage 3 (`pocket_multitask`) adds pocket geometry, contacts, field
conditioning, and affinity auxiliaries. Stage 4 (`high_quality_finetune`)
raises field and chemistry weights and validates with the quality sampler.

## Preflight and local smoke

```bash
ecloudflow doctor --json
ecloudflow config show +experiment=smoke
ecloudflow train --dry-run --output-dir runs/train-smoke \
  +experiment=smoke trainer.max_steps=2
```

The preflight resolves strict Pydantic configuration and records it. A real
training run requires prepared shards and a suitable accelerator. Lightning
owns mixed precision, optimizer state, EMA, checkpointing, and DDP; data
workers own deterministic rank/worker sharding.

## Objective and checkpointing

The objective combines flow velocity, terminal score, simplex cross-entropy,
field reconstruction, chemistry penalties, and interaction supervision. Loss
weights have explicit warm-up intervals and distributed running-RMS
normalization. BF16 is the production default on H100; CPU smoke uses 32-bit.
Checkpoints include model, optimizer, scheduler, EMA, resolved config, Git
commit, dataset hash, seed, and rank-local loader state. Resume only when the
semantic configuration is compatible; incompatible precision or topology must
fail before loading weights.

## Monitoring and limits

Watch non-finite batches, gradient norms, valid-yield previews, and validation
metrics. A bounded diagnostic artifact is emitted before stopping on repeated
NaN/Inf. Training loss is not a binding-quality result. Publish held-out
per-pocket metrics, confidence intervals, and ablations rather than selecting
the checkpoint from a single docking score.
