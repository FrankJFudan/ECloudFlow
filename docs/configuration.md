# Configuration

Configuration is composed from strict Pydantic models. Unknown fields and
out-of-range values fail before work starts. Inspect the resolved object with:

```bash
ecloudflow config show
ecloudflow config show model=large sample.profile=quality
ecloudflow config show --override sample.num_molecules=256
ecloudflow config explain trainer.devices
```

Commands accept repeatable `--override/-O` options and trailing `key=value`
arguments. The latter are useful for scripts, for example:

```bash
ecloudflow sample pocket.pdb -n 500 sample.corrector_steps=4
ecloudflow train --output-dir runs/stage3 stage=pocket_multitask \
  trainer.precision=bf16-mixed trainer.devices=1
```

## Important groups

`model` selects tiny/base/large width and equivariant `lmax`. `data` controls
dataset, shard directory, partition, batch size, workers, bucketing, and hash
verification. `sample` controls profile, target count, solver, steps, corrector,
and bounded attempts. `trainer` controls accelerator, precision, DDP, gradient
accumulation, checkpoints, deterministic algorithms, and NaN limits.

`loss` contains flow, score, discrete, field, chemistry, interaction, and
normalization weights. `evaluation` selects metric groups, bootstrap policy,
raw/relaxed source, and optional backends. `visualization` fixes theme,
isovalue, dimensions, formats, DPI, and plotting seed. `benchmark` fixes global
work, device counts, warm-up, measurement steps, and tolerances.

Resolved configuration is serialized into run manifests and checkpoints.
Machine paths, credentials, and local overrides belong outside version control.

Editable installs read the repository-root `configs/` tree. Regular wheel
installs read an identical packaged copy. Set `ECLOUDFLOW_CONFIG_DIR` to a
complete external tree containing `config.yaml` when server-specific presets
must remain outside the checkout. A unit test requires the editable and
packaged defaults to remain byte-identical.
