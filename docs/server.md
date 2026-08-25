# Server installation and execution

This runbook targets one Linux node with four NVIDIA H100 80 GB GPUs. The
repository uses Lightning DDP, BF16 mixed precision, immutable sharded data,
and rank-zero checkpoint publication. Multi-node training is not declared by
the supplied presets.

## Upload or clone

Push the ordinary Git repository to a private or public remote, then clone it
on the server. PDBBind data, CrossDocked data, checkpoints, run directories,
and credentials are ignored and must remain outside Git.

```bash
git clone <YOUR_REPOSITORY_URL> ECloudFlow
cd ECloudFlow
git rev-parse HEAD
git status --short
```

For an offline server, transfer a Git bundle or source archive together with
the separately acquired datasets. Do not place licensed PDBBind files in a
public repository.

## Environment

CUDA 12.8 PyTorch requires a compatible NVIDIA driver on the host. The setup
script creates an isolated environment, installs the verified direct versions,
installs ECloudFlow in editable mode without re-resolving dependencies, runs
`pip check`, and executes the four-GPU preflight.

```bash
bash scripts/setup_h100.sh ecloudflow-h100
conda activate ecloudflow-h100
nvidia-smi
ecloudflow doctor --server --json
```

Install optional scientific executables through the site package manager or a
separate Conda transaction, then rerun `ecloudflow doctor`. Record exact xTB,
Vina, and Open Babel versions in every published experiment.

## Data placement

Raw archives and processed shards should use fast storage outside the checkout:

```text
/data/ecloudflow/raw/pdbbind/
/data/ecloudflow/raw/crossdocked/
/data/ecloudflow/processed/pdbbind/manifest.json
/data/ecloudflow/processed/pdbbind/generations/<generation>/shard-*.tar
```

The `manifest.json` path is the only dataset argument needed by the server
launch scripts. The manifest refers to immutable generation-relative shards;
moving one complete processed dataset root is supported, while moving only the
manifest is not.

## Four-GPU acceptance

Run this before allocating a long queue job:

```bash
bash scripts/acceptance_h100.sh \
  /data/ecloudflow/processed/pdbbind/manifest.json \
  /runs/ecloudflow/acceptance
```

Success means the current environment passed the four-device, BF16, and NCCL
checks and completed a real two-step DDP fit, checkpoint publication, and
strict resume to step three. It does not establish model quality or throughput.

## Training

Launch one stage directly:

```bash
bash scripts/train_4xh100.sh \
  --manifest /data/ecloudflow/processed/pdbbind/manifest.json \
  --output /runs/ecloudflow/stage3 \
  --stage 3
```

Resume an interrupted stage with `--resume-from`. Start a new curriculum stage
with `--init-from`; the latter transfers only model groups and intentionally
does not transfer optimizer or stream position.

```bash
bash scripts/train_4xh100.sh \
  --manifest /data/ecloudflow/processed/pdbbind/manifest.json \
  --output /runs/ecloudflow/stage3-resumed \
  --stage 3 \
  --resume-from /runs/ecloudflow/stage3/checkpoints/last.ckpt
```

The complete default curriculum is:

```bash
bash scripts/train_curriculum_4xh100.sh \
  /data/ecloudflow/processed/pdbbind/manifest.json \
  /runs/ecloudflow/curriculum
```

Override stage lengths through `ECLOUDFLOW_STAGE1_STEPS` through
`ECLOUDFLOW_STAGE4_STEPS`. Keep `CUDA_VISIBLE_DEVICES=0,1,2,3` unless the
scheduler supplies a different four-device allocation.

## Sampling and evaluation

Use the trained stage-4 checkpoint with a pocket PDB in the desired global
binding-pose frame:

```bash
ecloudflow sample pocket.pdb \
  --checkpoint /runs/ecloudflow/curriculum/stage4/checkpoints/last.ckpt \
  --num-molecules 100 --profile balanced \
  --output-dir /runs/ecloudflow/sample-pocket
ecloudflow evaluate /runs/ecloudflow/sample-pocket --profile full
ecloudflow report /runs/ecloudflow/sample-pocket --format paper --top-n 20
```

Retain the Git revision, resolved config, dependency list, dataset manifest
hash, checkpoint hash, generated SDF files, ranked tables, failed attempts,
docking logs, evaluation JSON, and report figures together.

## Scheduler boundary

The supplied scripts are scheduler-neutral and run on a single allocated node.
In Slurm, request one node and four GPUs, activate the environment inside the
job, and invoke the script once. Do not run four copies of the script and do
not wrap `ecloudflow train` in `torchrun`; Lightning owns the four child
processes for this configuration.
