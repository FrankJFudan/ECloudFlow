# Evaluation and ranking

Run evaluation only after generation and docking artifacts are complete:

```bash
ecloudflow evaluate runs/pocket
ecloudflow evaluate runs/smoke --profile smoke
ecloudflow report runs/pocket --format paper --top-n 20
```

The registry separates seven domains: chemistry, distribution, geometry,
binding, electron-cloud, conditional/fragment, and efficiency. The smoke
profile selects inexpensive chemistry, conditional, and efficiency checks.
Missing Vina, PoseBusters, FCD, PLIF, or QM backends produce `unavailable` with
provenance rather than zero or an imputed score.

## Metrics and aggregation

Chemistry covers sanitization, valence, connectivity, stability, and filters.
Distribution covers uniqueness, novelty, scaffold diversity, descriptor
distances, and optional FCD. Geometry covers bond/angle/dihedral distributions,
clashes, strain, raw-to-relaxed RMSD, and PoseBusters. Binding covers Vina,
contacts, occupancy, clashes, and field complementarity. Conditional metrics
measure requested-property error and exact fragment preservation. Efficiency
records NFE, wall time, memory, and valid yield.

Per-pocket macro means give every pocket equal weight. Seeded bootstrap resamples
produce 95% intervals where observations exist; paired pocket tests and multiple
seeds are preferred for comparisons. Raw and relaxed poses are reported in
separate columns. The deterministic ranking order is Vina ascending, QED
descending, SA ascending, then canonical isomeric SMILES.

## Interpretation

Docking is a ranking heuristic, not a binding free energy. A report is valid
only with its split, postprocessing, docking box, software versions, seed, and
checkpoint hash. Compare baselines under identical requested counts and tool
availability; do not turn unavailable metrics into favorable averages.
