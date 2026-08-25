# Data preparation

ECloudFlow does not redistribute PDBBind, CrossDocked, or derived checkpoints.
Acquire each dataset from its publisher, keep the raw files outside the Git
checkout, and use `ecloudflow data import-local` to create immutable,
content-addressed WebDataset shards. The importer never downloads data or
accepts a license on the user's behalf.

## Dataset sources

### PDBBind

Download an authorized release from the official
[PDBBind download page](https://www.pdbbind-plus.org.cn/download). Registration
or a license agreement may be required. Keep the downloaded archive and its
license under access controls appropriate for that release.

The importer accepts the usual extracted PDBBind hierarchy. The
`--source-root` may contain refined and general-set directories at any depth,
provided the complex files and the general protein-ligand index are below it:

```text
/data/ecloudflow/raw/pdbbind/
|-- index/
|   `-- INDEX_general_PL_data.2020
|-- refined-set/
|   `-- 1abc/
|       |-- 1abc_protein.pdb
|       |-- 1abc_ligand.sdf
|       `-- 1abc_pocket.pdb
`-- v2020-other-PL/
    `-- 2xyz/
        |-- 2xyz_protein.pdb
        `-- 2xyz_ligand.sdf
```

An existing `*_pocket*.pdb` is used when present. Otherwise, the importer
extracts all protein residues within `--pocket-radius` angstroms of the bound
ligand. The general INDEX contributes the published pK value, assay family
(`Kd`, `Ki`, `IC50`, or `Ka`), censoring relation, original expression, unit,
resolution, and release year. These labels are preserved separately so unlike
assays are not silently merged.

Run a small graph-only validation before a full field-enabled import:

```bash
ecloudflow data import-local \
  --dataset pdbbind \
  --source-root /data/ecloudflow/raw/pdbbind \
  --output-dir /data/ecloudflow/processed/pdbbind-smoke \
  --limit 32 --workers 4 --no-fields
```

Then publish the production shards. Supply `--index` only when the index is
not under `--source-root` or when several releases are present:

```bash
ecloudflow data import-local \
  --dataset pdbbind \
  --source-root /data/ecloudflow/raw/pdbbind \
  --index /data/ecloudflow/raw/pdbbind/index/INDEX_general_PL_data.2020 \
  --output-dir /data/ecloudflow/processed/pdbbind \
  --workers 16 --strict-sources
```

### CrossDocked2020

Download CrossDocked2020 and the matching split/type files from the official
[CrossDocked file server](https://bits.csb.pitt.edu/files/crossdock2020/).
Extract them without flattening paths. `--source-root` must be the common
parent against which receptor and ligand paths in the selected `.types` file
resolve:

```text
/data/ecloudflow/raw/crossdocked/
|-- types/
|   `-- it2_tt_v1.1_completeset_train0.types
`-- crossdocked_pocket10/
    `-- <target-directory>/
        |-- <receptor>.pdb
        `-- <ligand>.sdf.gz
```

Some distributions place the target directories directly below the extracted
root. That is also supported when the `.types` paths match that layout. Direct
one-record ligand SDF files and the official numbered virtual records backed
by multi-record `.sdf.gz` archives are both accepted.

The default import keeps poses with RMSD at most 1.0 angstrom. Use the exact
`.types` file shipped with the downloaded release:

```bash
ecloudflow data import-local \
  --dataset crossdocked \
  --source-root /data/ecloudflow/raw/crossdocked \
  --index /data/ecloudflow/raw/crossdocked/types/it2_tt_v1.1_completeset_train0.types \
  --rmsd-threshold 1.0 \
  --output-dir /data/ecloudflow/processed/crossdocked \
  --workers 16 --strict-sources
```

If the archive has an additional top-level directory, either pass that
directory as `--source-root` or retain the common parent and pass the matching
index explicitly. Do not rewrite `.types` paths unless the rewrite is recorded
as part of dataset provenance.

## Electron fields

Imports build pocket and ligand electron-field inputs by default. Ligand QM
fields require the configured xTB executable and can be expensive. Confirm the
scientific toolchain before a full import:

```bash
ecloudflow doctor
```

Use `--no-fields` only for graph-only diagnostics, importer smoke tests, or an
explicit ablation. Missing or failed QM calculations are represented by typed
availability masks and provenance; the importer never invents a QM target.

## Leakage-controlled splits

Before writing shards, the importer groups complexes by protein identity and
ligand Bemis-Murcko scaffold. The default connected-component split targets
80% train, 10% validation, and 10% test. For more than 5,000 raw protein
sequences, provide a scalable two-column cluster file instead of invoking the
quadratic sequence fallback:

```text
1abc cluster_0001
2xyz cluster_0002
```

```bash
ecloudflow data import-local \
  --dataset pdbbind \
  --source-root /data/ecloudflow/raw/pdbbind \
  --output-dir /data/ecloudflow/processed/pdbbind \
  --protein-clusters /data/ecloudflow/metadata/pdbbind_clusters.tsv \
  --train-fraction 0.8 --val-fraction 0.1 --split-seed 2026 \
  --workers 16
```

Cluster keys may be PDB IDs, receptor stems, or dataset-relative receptor
paths. Every source must have a cluster assignment when the file is supplied.

## Processed format

One successful import produces a movable dataset root:

```text
/data/ecloudflow/processed/pdbbind/
|-- manifest.json
|-- import-summary.json
`-- generations/
    `-- <content-id>/
        |-- shard-000000.tar
        |-- shard-000001.tar
        `-- publication.json
```

Each tar member contains canonical graph tensors, binding-frame coordinates,
optional electron-field coefficients and masks, properties, source hashes, and
preprocessing provenance. `manifest.json` records shard hashes, sample IDs,
train/validation/test assignments, leakage audit data, skips, and the active
immutable generation. `import-summary.json` reports discovered, filtered,
accepted, serialized, and rejected record counts.

Shards target 1 GiB by default and may be configured between 0.5 and 2 GiB:

```bash
ecloudflow data import-local \
  --dataset pdbbind \
  --source-root /data/ecloudflow/raw/pdbbind \
  --output-dir /data/ecloudflow/processed/pdbbind \
  --override data.target_shard_size_gb=2.0
```

Move or back up the complete processed root, not an isolated manifest. Verify
hashes during scientific training and retain the raw release, INDEX/types file,
import summary, resolved configuration, and manifest hash for reproducibility.

## Training configuration

The training loader needs only the published manifest. Override it on the
command line:

```bash
ecloudflow train +experiment=pdbbind_large \
  data.manifest=/data/ecloudflow/processed/pdbbind/manifest.json \
  --output-dir /runs/ecloudflow/stage3
```

or edit a copied YAML preset:

```yaml
defaults:
  - override /data: pdbbind

data:
  manifest: /data/ecloudflow/processed/pdbbind/manifest.json
  batch_size: 8
  num_workers: 16
  prefetch_factor: 4
  verify_shard_hashes: true
```

`data.partition` defaults to `train`; validation and test readers use the
assignments frozen in the same manifest. Server launch scripts accept the
manifest path directly, so raw data paths never enter training commands.

## Single-complex diagnostics

`data prepare` remains available for a single explicit pair and for
configuration preflight. It is not the production batch importer:

```bash
ecloudflow data prepare --dataset pdbbind \
  --pocket examples/toy_pocket.pdb --ligand examples/toy_ligand.sdf \
  --sample-id TOY --output-dir data/processed/toy --no-fields
```

No command silently substitutes a missing molecule, affinity value, or field.
Use `--strict-sources` for release builds; use non-strict mode when auditing a
dataset, then inspect every entry in `import-summary.json` before training.
