# ECloudFlow

ECloudFlow is a research implementation of pocket-conditioned 3D ligand
generation.  It combines an SE(3)-equivariant graph model, atom-centered
electron-cloud latents, flow matching, diffusion score correction, and exact
chemical graph decoding.  The public pipeline generates coordinates in the
pocket binding frame, supports de novo and fragment-conditioned objectives,
and records every accepted, rejected, and failed attempt.

This repository is an engineering and reproducibility scaffold.  It does not
ship a trained checkpoint and it does not claim a state-of-the-art binding
benchmark.  Binding quality must be measured with a declared docking protocol,
matched data split, and independent validation.

## Highlights

- Pocket and ligand electron fields are represented as radial spherical
  coefficients and packed equivariant latent tokens.
- Continuous positions and electron latents use a hybrid flow/score objective;
  categorical atom, charge, bond, and count variables use an interpolating
  path.
- A Heun flow solver, Euler fast profile, and terminal Langevin corrector share
  one bounded sampler.  Fast, balanced, and quality profiles expose explicit
  NFE contracts.
- Valence projection, fixed-fragment masks, connectivity checks, RDKit
  sanitization, and a CP-SAT exact decoder reject invalid graphs.
- Raw binding-pose SDF and optional relaxed SDF are always separate artifacts.
- Docking-aware deterministic ranking uses Vina score ascending, QED descending,
  SA ascending, and canonical isomeric SMILES as a final tie break.
- Evaluation is organized into chemistry, distribution, geometry, binding,
  electron-cloud, conditional, and efficiency domains.  Missing optional tools
  produce explicit `unavailable` statuses rather than fabricated numbers.
- CSV, Parquet, Excel, JSON, SDF, self-contained HTML, SVG, PDF, and PNG
  artifacts are produced with stable provenance and hashes.

## Installation

Use the supplied Python environment or create an isolated environment with
Python 3.10--3.12.  Install the package in editable mode:

```bash
conda activate 3dmolecule
python -m pip install -e .
python -m pip install -e '.[dev,eval,viz,train]'
```

The core package does not require Vina, xTB, PoseBusters, or Py3Dmol.  Install
those tools separately when the corresponding metric or viewer is enabled.

## Doctor

Run diagnostics before an expensive preprocessing or training job:

```bash
ecloudflow doctor
ecloudflow doctor --json
ecloudflow doctor --dataset data/processed/pdbbind --output-dir runs/doctor
ecloudflow doctor --checkpoint checkpoints/ecloudflow-large.ckpt --require-gpu
```

Required Python packages are checked separately from optional executables.  A
missing Vina or xTB binary is reported and disables only the dependent metric;
`--require-gpu` makes CUDA visibility a hard preflight requirement.

## Tiny Quick Start

The smoke mode is explicit and deterministic.  It validates the complete
generation, ranking, evaluation, report, and viewer wiring without pretending
that a toy score is a physical docking result:

```bash
ecloudflow sample tests/fixtures/complex/toy_pocket.pdb \
  --num-molecules 8 --profile fast --smoke --docking deterministic \
  --output-dir runs/smoke
ecloudflow evaluate runs/smoke --profile smoke
ecloudflow report runs/smoke --format paper
ecloudflow visualize molecule runs/smoke --id toy_pocket-000001
ecloudflow visualize ecloud runs/smoke
```

The run contains `generation.json`, `samples.csv`, `samples.parquet`,
`summary.xlsx`, `summary.json`, `ranked.sdf`, `failed.csv`, raw/relaxed SDF
poses, `evaluation.json`, `report.html`, and publication figures.

## Data Preparation

PDBBind, CrossDocked, and ligand-pretraining data are represented by immutable
content-addressed WebDataset shards.  The importer preserves the source frame,
source hashes, split assignments, and optional QM provenance.  A preparation
preflight writes a resolved manifest without inventing records:

```bash
ecloudflow data prepare --dataset pdbbind --output-dir data/prepared
```

For one explicit pocket/ligand pair, build a canonical sample and manifest:

```bash
ecloudflow data prepare --dataset pdbbind \
  --pocket examples/3ztx_pocket.pdb --ligand examples/3ztx_ligand.sdf \
  --sample-id 3ZTX --output-dir data/processed/pdbbind --no-fields
```

`--no-fields` is useful for graph-only debugging.  Physical fields should be
constructed with the configured pocket builder and xTB runner for scientific
training.  A failed xTB calculation is stored as typed unavailable QM
provenance and never replaced by an approximate label.

## Four Training Stages

The curriculum is explicit and resumable:

1. `electron_tokenizer` learns field coefficients, latent irreps, and decoder
   reconstruction using genuine QM masks.
2. `ligand_pretrain` learns ligand geometry and categorical graph paths without
   claiming pocket interaction supervision.
3. `pocket_multitask` jointly conditions on pocket geometry, fields, contacts,
   and affinity auxiliaries.
4. `high_quality_finetune` increases electron and chemical-constraint weights
   and uses the quality sampling protocol for validation.

Inspect a resolved configuration or write a training preflight:

```bash
ecloudflow config show +experiment=smoke
ecloudflow train --dry-run --output-dir runs/train-smoke \
  +experiment=smoke trainer.max_steps=2
```

The preflight records the exact Pydantic configuration.  The Lightning module
owns optimizer, EMA, checkpoint, mixed precision, and DDP state; data loading
owns rank/worker sharding and resume positions.  There is no manual CUDA move
in model or loss code.

## Four H100 GPUs

The production preset targets four 80 GB H100 devices with BF16 mixed
precision, DDP, gradient accumulation, deterministic checkpoint metadata, and
large worker/prefetch settings:

```bash
ecloudflow config show +experiment=pdbbind_large
ecloudflow train +experiment=pdbbind_large --output-dir runs/pdbbind-large
```

Run the benchmark harness after a dataset and checkpoint are available:

```bash
bash scripts/run_h100_smoke.sh --config experiment=h100_smoke --output runs/h100-smoke
bash scripts/benchmark_scaling.sh --config experiment=h100_large --output runs/scaling
ecloudflow benchmark --devices 1 --devices 2 --devices 4 \
  --steps 100 --config experiment=h100_large --output-dir runs/scaling
```

The scripts print Git and dataset-manifest hashes.  `run_h100_smoke.sh` uses a
local benchmark by default; set `ECLOUDFLOW_RUN_NCCL=1` on the server to invoke
`torchrun --nproc_per_node=4`.  Local CI uses a clearly labeled CPU simulation
when requested device counts are not visible.

## Sampling

The normal command requires a trained checkpoint and a pocket PDB in the
desired output coordinate frame:

```bash
ecloudflow sample 3ztx_pocket.pdb --checkpoint checkpoints/ecloudflow-large.ckpt \
  --num-molecules 100 --profile balanced --output-dir runs/3ztx
```

`--num-molecules` is the target number of valid unique molecules, not the
number of raw attempts.  The default attempt bound is five times the target;
`--max-attempts` and `--strict-count` expose bounded-shortfall behavior.

Fragment-conditioned optimization uses one shared pipeline:

```bash
ecloudflow sample 3ztx_pocket.pdb --checkpoint checkpoints/ecloudflow-large.ckpt \
  --fragment hit_fragment.sdf --mode grow -n 100 --profile balanced
ecloudflow sample 3ztx_pocket.pdb --fragment hit_fragment.sdf --mode link -n 100
ecloudflow sample 3ztx_pocket.pdb --fragment hit_fragment.sdf --mode replace -n 100
ecloudflow sample 3ztx_pocket.pdb --fragment hit_fragment.sdf --mode merge -n 100
```

Fixed atom identity, charge, internal bonds, and coordinates are clamped after
every solver/corrector operation.  The final graph decoder unions fixed
components before adding attachment edges and rejects valence, connectivity,
sanitization, and conformer failures.

### Profiles and NFE

| Profile | Integrator | Nominal NFE | Intended use |
| --- | --- | ---: | --- |
| `fast` | Euler, 20 steps | 20 | screening |
| `balanced` | Heun, 40 steps, terminal corrector | 82 | default |
| `quality` | Heun, 100 steps, stronger corrector | 208 | final candidates |

The NFE values are reporting contracts; actual model calls and optional
guidance/decoding work are recorded in efficiency diagnostics.

## Docking, Ranking, and Outputs

Docking is an evaluation operation and never overwrites the raw binding pose.
With Vina installed, select `--docking vina` or use `--docking auto`.  Missing
scores remain in `failed.csv` with an explicit status.  Smoke mode can use
`--docking deterministic`, which is only a wiring fixture and must not be used
as a scientific affinity claim.

Successfully scored rows are sorted by:

1. docking score ascending (more negative is better);
2. QED descending;
3. conventional SA ascending (lower is easier);
4. canonical isomeric SMILES lexicographically.

IDs are exactly `<POCKET_ID>-<RANK:06d>`, for example `3ZTX-000001`; no `ECLF`
component is inserted.  Every ranked row includes SMILES, SA, QED, docking
score, rank, pocket ID, raw/relaxed paths, seed, and checkpoint hash.

## Evaluation and Reporting

```bash
ecloudflow evaluate runs/3ztx
ecloudflow evaluate runs/3ztx --profile smoke
ecloudflow report runs/3ztx --format paper --top-n 20
```

The registry reports seven domains: chemistry, distribution, geometry,
binding, electron-cloud, conditional/fragment, and efficiency.  Per-pocket
macro averages give every pocket equal weight; seeded bootstrap intervals are
included where observations exist.  Optional PoseBusters, FCD, PLIF, Vina,
QM-density, and memory metrics return `unavailable` when their backend is not
configured.

## Configuration

All settings are strict Pydantic models composed by Hydra.  Unknown keys fail
before work starts.  Use either repeatable `--override` flags or trailing
`key=value` arguments:

```bash
ecloudflow config show model=large sample.profile=quality
ecloudflow config show --override sample.num_molecules=256
ecloudflow config explain trainer.devices
ecloudflow sample pocket.pdb -n 500 sample.corrector_steps=4
```

The resolved configuration is serialized into training, benchmark, and
evaluation artifacts.  Keep machine-specific paths in ignored local override
files rather than committing them.

## Python API

```python
from ecloudflow import ECloudFlowPipeline

pipeline = ECloudFlowPipeline.from_pretrained(
    "checkpoints/ecloudflow-large.ckpt",
    map_location="cuda",
)
result = pipeline.generate(
    pocket="3ztx_pocket.pdb",
    fragment="hit_fragment.sdf",
    mode="grow",
    num_molecules=100,
    profile="balanced",
    output_dir="runs/3ztx",
)
docking = pipeline.dock_and_rank(result, "3ZTX", pocket="3ztx_pocket.pdb")
result.to_excel("runs/3ztx/generation.xlsx")
```

The typed result objects expose `valid`, `attempt_records`, `shortfall`,
`model_checkpoint_hash`, ranked rows, and output manifests.  Python and CLI
workflows use the same serialization and ranking implementations.

## Troubleshooting

- `checkpoint does not exist`: pass a real checkpoint or use explicit
  `--smoke` only for pipeline wiring tests.
- `dataset manifest does not exist`: run the importer/preparation step and
  verify `data.shard_dir` and `data.manifest` in the resolved config.
- `unavailable` docking: install Vina and prepare receptor/ligand PDBQT inputs;
  do not interpret an unavailable value as zero.
- xTB failure: inspect `QMProvenance` and rerun with the recorded executable,
  charge, multiplicity, and command.
- fragment shortfall: inspect rejection reasons, attachment masks, and the
  bounded `max_attempts`; fixed coordinates are intentionally never relaxed.
- NCCL launch failure: run `ecloudflow doctor --require-gpu`, verify four
  visible devices, and launch from the same environment on every rank.

## Repository Map

```text
src/ecloudflow/core          tensor, frame, and mask contracts
src/ecloudflow/ecloud        electron fields, basis, tokenizer, QM provenance
src/ecloudflow/models        equivariant pocket/ligand joint model
src/ecloudflow/process       flow, score, and categorical paths
src/ecloudflow/chemistry     valence projection, decoder, relaxation
src/ecloudflow/data          parsers, split audits, shards, DataModule
src/ecloudflow/sampling      priors, solvers, corrector, generation records
src/ecloudflow/docking       typed Vina adapter
src/ecloudflow/evaluation    ranking, metrics, aggregation, output writers
src/ecloudflow/visualization viewers, plots, and HTML reports
src/ecloudflow/training      Lightning module, checkpoints, benchmark harness
src/ecloudflow/cli           Typer commands and environment doctor
configs                      tiny/base/large and experiment presets
docs                          theory and operational references
```

## Limitations and Responsible Use

The repository contains architecture and validation code, not a validated
clinical or production drug-discovery system.  Generated molecules require
human medicinal-chemistry review, independent structure validation, docking
protocol checks, and experimental confirmation.  A docking score is not a
binding free energy.  Electron-cloud latents improve the conditioning signal
only when field labels and coordinate conventions are correct.  Benchmark
comparisons must report seeds, split policy, postprocessing, NFE, and tool
versions.

## References and Attribution

The design is informed by DiffGui, ECloudGen, PropMolFlow, JODO, and CoCoGraph
ideas.  File-level adaptation notes and license responsibilities are listed in
`THIRD_PARTY_NOTICES.md`.  See `docs/theory.md` for the mathematical model and
`docs/reproducibility.md` for run manifests, hashes, and reporting policy.
