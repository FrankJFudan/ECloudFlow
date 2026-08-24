# Data preparation

The importer converts PDBBind, CrossDocked, or a ligand-pretraining source
into immutable complex records and content-addressed WebDataset shards. Each
record keeps a source identifier, pocket and ligand paths, coordinate-frame
metadata, canonical SMILES, split assignment, and hashes. Shards are written
atomically with a manifest so interrupted preparation cannot look complete.

## Quick commands

```bash
ecloudflow data prepare --dataset pdbbind --output-dir data/prepared
ecloudflow data prepare --dataset pdbbind --pocket examples/3ztx_pocket.pdb \
  --ligand examples/3ztx_ligand.sdf --sample-id 3ZTX \
  --output-dir data/processed/pdbbind --no-fields
```

The first form writes a preflight manifest. The second parses one explicit
pair. `--no-fields` is graph-only diagnostics; production imports should run a
configured QM backend and record charge, multiplicity, executable, and field
grid settings. A failed calculation is typed as unavailable and remains
visible in the manifest.

## Normalization and leakage control

Pockets are standardized without changing the requested output frame. Ligands
are sanitized, assigned formal charges, and canonicalized with isomeric SMILES.
Duplicate complexes and near-identical proteins are grouped before splitting.
The grouped split prevents a ligand scaffold or pocket family from appearing
in both training and evaluation. Split manifests are immutable inputs to every
run; changing one requires a new manifest hash.

## Shards and validation

Samples use flattened PyG-style tensors: `N` nodes, `E` unordered halfedges,
positions `[N,3]`, node features `[N,C]`, bond features `[E,Cb]`, and batch
indices for complexes. Validate hashes and schema before training, inspect
rejection counts, and retain source paths for forensic review. Keep credentials
and machine paths in ignored local overrides.

## Provenance

Record dataset release, importer version, source checksums, standardization
options, split seed, field backend/version, and shard manifest hash. The output
run stores the resolved configuration and these provenance fields. No importer
silently substitutes a missing molecule or invents a QM value.
