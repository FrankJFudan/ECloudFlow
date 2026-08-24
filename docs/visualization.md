# Visualization and reports

Generate an interactive viewer or paper bundle from a completed run:

```bash
ecloudflow visualize molecule runs/pocket --id 3ZTX-000001
ecloudflow visualize ecloud runs/pocket --id 3ZTX-000001
ecloudflow report runs/pocket --format html
ecloudflow report runs/pocket --format paper --top-n 20
```

The molecule view can show pocket context, sticks, contacts, clashes, fixed
fragments, and raw versus relaxed poses. The electron view renders density
isosurfaces and differences when field artifacts exist. Reports include ranked
tables, downloadable metric data, confidence intervals, trajectory frames, and
static SVG/PDF/300-DPI PNG figures.

Plots use a deterministic seed, fixed dimensions, explicit units, and a
colorblind-safe paper theme. Standard panels are ECDF/violin intervals,
paired-pocket comparisons, geometry heatmaps, property distributions,
top-molecule grids, and quality-speed Pareto plots. Reports never recompute or
silently alter evaluation values; they consume the serialized result.

When Py3Dmol, Plotly, or Matplotlib is absent, the command reports the missing
optional dependency clearly. Preserve HTML, source JSON, and hashes together so
an image can be regenerated from the same metrics.

Viewer inputs are read-only: rendering does not relax, reorder, or mutate
molecules. Use the displayed molecule ID to trace a panel back to its ranked
SDF record and then to the generation attempt. Large reports can be limited
with `--top-n` and regenerated with the same seed from `visualization.seed`.
