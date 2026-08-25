"""Deterministic, colorblind-safe publication plotting helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442")


def plot_quality_speed_pareto(data: Any, path: str | Path, *, seed: int = 2026) -> Path:
    """Plot quality against measured per-sample generation duration.

    :param data: Rows containing a quality value and optionally a timing value.
    :param path: SVG/PDF/PNG destination selected from its suffix.
    :param seed: Stable SVG identifier salt.
    :return: Written publication figure path.
    :rtype: pathlib.Path

    The function prefers ``elapsed_seconds`` because it is the per-attempt
    measurement emitted by the generation manifest. Legacy ``wall_time`` and
    ``time`` fields remain supported. When no valid timing values exist, the
    plot explicitly falls back to sample order and changes its title so a
    profile plot is never misrepresented as a speed comparison.
    """
    rows = _rows(data)
    time_metric = _first_numeric_metric(rows, ("elapsed_seconds", "wall_time", "time"))
    quality_metric = _first_numeric_metric(rows, ("quality", "qed"))
    if time_metric is None:
        x = list(range(1, len(rows) + 1))
        y = [
            _number(row.get(quality_metric), 0.0) if quality_metric else 0.0
            for row in rows
        ]
        xlabel = "Sample order"
        title = "Quality profile (timing unavailable)"
    else:
        paired = [
            (_number(row.get(time_metric)), _number(row.get(quality_metric)))
            for row in rows
            if quality_metric is not None
        ]
        x = [value[0] for value in paired if value[0] is not None and value[1] is not None]
        y = [value[1] for value in paired if value[0] is not None and value[1] is not None]
        xlabel = _axis_label(time_metric)
        title = "Quality-speed Pareto"
    return _scatter(
        x,
        y,
        path,
        xlabel,
        _axis_label(quality_metric or "quality"),
        title,
        seed=seed,
    )


def plot_metric_distribution(
    data: Any, path: str | Path, *, metric: str | None = None
) -> Path:
    """Plot a compact distribution for an explicit or detected metric column.

    :param data: Mapping, row sequence, or dataframe-like measurement source.
    :param path: SVG/PDF/PNG destination selected from its suffix.
    :param metric: Optional requested metric column. When omitted, binding and
        medicinal-chemistry metrics are selected before generic numeric fields.
    :return: Written publication figure path.
    :rtype: pathlib.Path

    Generation reports normally contain ``docking_score``, ``qed``, and ``sa``
    rather than a synthetic column named ``value``. Selecting from actual
    numeric columns keeps the default report informative while preserving exact
    caller control through the explicit ``metric`` argument.
    """
    rows = _rows(data)
    selected_metric = metric or _first_numeric_metric(
        rows,
        ("docking_score", "vina_score", "qed", "sa", "sa_score", "quality", "value"),
    )
    if selected_metric is None:
        selected_metric = "value"
    values = [
        _number(row.get(selected_metric))
        for row in rows
        if _number(row.get(selected_metric)) is not None
    ]
    return _simple_plot(
        values,
        path,
        selected_metric,
        f"{_axis_label(selected_metric)} distribution",
    )


def plot_ecdf(data: Any, path: str | Path, *, metric: str = "value") -> Path:
    """Export an empirical cumulative distribution plot."""
    values = sorted(
        _number(row.get(metric))
        for row in _rows(data)
        if _number(row.get(metric)) is not None
    )
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    if values:
        axis.step(
            values,
            [(index + 1) / len(values) for index in range(len(values))],
            color=_COLORS[0],
        )
    axis.set_xlabel(metric)
    axis.set_ylabel("ECDF")
    axis.grid(alpha=0.25)
    return _save(fig, path)


def plot_violin(data: Any, path: str | Path, *, metric: str = "value") -> Path:
    """Export a compact violin plot for one metric column."""
    values = [
        _number(row.get(metric))
        for row in _rows(data)
        if _number(row.get(metric)) is not None
    ]
    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    if values:
        axis.violinplot([values], showmeans=True)
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_geometry_heatmap(data: Any, path: str | Path) -> Path:
    """Export a geometry quality heatmap from rectangular numeric rows."""
    rows = _rows(data)
    keys = [
        key
        for key, value in (rows[0].items() if rows else [])
        if isinstance(value, (int, float))
    ]
    matrix = [[_number(row.get(key, 0.0), 0.0) for key in keys] for row in rows]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    if matrix:
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=axis)
    axis.set_xlabel("geometry metric")
    axis.set_ylabel("sample")
    return _save(fig, path)


def plot_property_distribution(
    data: Any, path: str | Path, *, metric: str = "qed"
) -> Path:
    """Export a property histogram."""
    values = [
        _number(row.get(metric))
        for row in _rows(data)
        if _number(row.get(metric)) is not None
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    if values:
        axis.hist(
            values, bins=min(20, max(5, len(values))), color=_COLORS[2], alpha=0.85
        )
    axis.set_xlabel(metric)
    axis.set_ylabel("count")
    return _save(fig, path)


def _scatter(
    x: list[float],
    y: list[float],
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    *,
    seed: int = 2026,
) -> Path:
    """Create and save one deterministic scatter figure."""
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.scatter(x, y, color=_COLORS[0], edgecolors="white", linewidths=0.5)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    return _save(fig, path, seed=seed)


def _simple_plot(
    values: list[float], path: str | Path, xlabel: str, title: str
) -> Path:
    """Plot a deterministic sorted line distribution."""
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(range(len(values)), sorted(values), color=_COLORS[0])
    axis.set_xlabel("sample")
    axis.set_ylabel(xlabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    return _save(fig, path)


def _save(fig: Any, path: str | Path, *, seed: int = 2026) -> Path:
    """Save SVG/PDF/PNG with stable metadata and close the figure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Creator": "ECloudFlow"}
    # Matplotlib hashes marker and clip-path IDs unless an explicit salt is set.
    # A seed-derived salt keeps independently rendered figures byte-identical.
    with matplotlib.rc_context({"svg.hashsalt": f"ECloudFlow:{seed}"}):
        fig.savefig(destination, dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    return destination


def _rows(data: Any) -> list[dict[str, Any]]:
    """Normalize mappings, sequences, and pandas-like records."""
    if data is None:
        return []
    if isinstance(data, Mapping):
        return [dict(data)]
    if hasattr(data, "to_dict"):
        converted = data.to_dict("records")
        return [dict(row) for row in converted]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return [
            dict(row) if isinstance(row, Mapping) else {"value": row} for row in data
        ]
    return [{"value": data}]


def _first_numeric_metric(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> str | None:
    """Return the first candidate column containing at least one finite value."""
    for candidate in candidates:
        if any(_number(row.get(candidate)) is not None for row in rows):
            return candidate
    return None


def _axis_label(metric: str) -> str:
    """Return a compact publication label for common ECloudFlow measurements."""
    labels = {
        "docking_score": "Docking score (kcal/mol)",
        "vina_score": "Vina score (kcal/mol)",
        "elapsed_seconds": "Elapsed time (s)",
        "wall_time": "Sampling time (s)",
        "time": "Sampling time (s)",
        "qed": "QED",
        "sa": "SA score",
        "sa_score": "SA score",
    }
    return labels.get(metric, metric.replace("_", " "))


def _number(value: Any, default: float | None = None) -> float | None:
    """Convert a scalar to finite float, returning ``default`` otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


__all__ = [
    "plot_ecdf",
    "plot_geometry_heatmap",
    "plot_metric_distribution",
    "plot_property_distribution",
    "plot_quality_speed_pareto",
    "plot_violin",
]
