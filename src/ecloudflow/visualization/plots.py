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
    """Plot quality versus sampling speed and export a deterministic vector."""
    rows = _rows(data)
    x = [_number(row.get("wall_time", row.get("time", 0.0)), 0.0) for row in rows]
    y = [_number(row.get("quality", row.get("qed", 0.0)), 0.0) for row in rows]
    return _scatter(
        x,
        y,
        path,
        "Sampling time (s)",
        "Quality",
        "Quality-speed Pareto",
        seed=seed,
    )


def plot_metric_distribution(
    data: Any, path: str | Path, *, metric: str = "value"
) -> Path:
    """Plot a compact distribution for one metric column."""
    values = [
        _number(row.get(metric))
        for row in _rows(data)
        if _number(row.get(metric)) is not None
    ]
    return _simple_plot(values, path, metric, "Metric distribution")


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
