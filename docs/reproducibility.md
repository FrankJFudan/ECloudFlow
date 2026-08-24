# Reproducibility checklist

Every published run should include the Git commit, Python and package versions,
CUDA/driver details, checkpoint SHA-256, dataset and shard-manifest hashes,
resolved configuration, seed, split policy, and optional-tool versions.

Start with a diagnostic record:

```bash
ecloudflow doctor --json --output-dir runs/doctor
ecloudflow config show +experiment=smoke > runs/smoke/resolved-config.json
```

For generation, retain `resolved-config.json`, `generation.json`,
`samples.csv`, `samples.parquet`, `summary.xlsx`, `summary.json`, `ranked.sdf`,
raw and relaxed poses, and `failed.csv`.  The resolved configuration includes
the effective profile, attempt bound, seed, checkpoint, and docking request.
For evaluation, retain `evaluation.json`, report HTML, figures, and the exact
command line. Temporary files are atomically renamed so a published artifact
is complete.

Use fixed seeds for preprocessing, model initialization, sampling, bootstrap,
and plotting. Record NFE and attempt bounds, not just wall time. If CUDA,
parallel kernels, or external docking introduce nondeterminism, state the
known tolerance and provide repeated-seed summaries. Never commit credentials,
absolute machine paths, or private data.

## Publication policy

Report per-pocket macro means and seeded 95% intervals, raw and relaxed metrics
separately, and unavailable metrics explicitly. State whether a checkpoint was
trained to convergence and identify all ablations. A smoke deterministic score
is a wiring test, not evidence of binding quality. Independent chemistry and
experimental review remain necessary for any generated molecule.
