"""Seeded pocket-level macro aggregation and confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BootstrapSummary:
    """Summarize an equally weighted pocket macro-average."""

    mean: float | None
    ci_low: float | None
    ci_high: float | None
    std: float | None = None
    pockets: int = 0
    resamples: int = 0
    value: str = "value"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary mapping."""
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std": self.std,
            "pockets": self.pockets,
            "resamples": self.resamples,
            "value": self.value,
        }


def bootstrap_macro_summary(
    rows: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]],
    *,
    value: str = "value",
    seed: int = 2026,
    resamples: int = 1000,
) -> BootstrapSummary:
    """Compute a seeded 95% CI after equal weighting of pocket means.

    :param rows: Mapping from pocket ID to observations, or row mappings with
        ``pocket_id`` and ``value`` keys.
    :param value: Value key when row mappings are supplied.
    :param seed: Local random seed; global random state is never modified.
    :param resamples: Positive bootstrap replicate count.
    :return: Macro mean and percentile confidence interval.
    :rtype: BootstrapSummary
    :raises ValueError: If no finite observations or invalid controls exist.

    A pocket contributes one mean regardless of how many molecules it contains;
    this prevents a large pocket from dominating a small-pocket benchmark.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    pocket_values = _group_values(rows, value)
    means = [sum(values) / len(values) for values in pocket_values.values() if values]
    if not means:
        return BootstrapSummary(None, None, None, None, 0, resamples, value)
    mean = sum(means) / len(means)
    if len(means) == 1:
        return BootstrapSummary(mean, mean, mean, 0.0, 1, resamples, value)
    rng = random.Random(seed)
    replicates = []
    for _ in range(resamples):
        sample = [means[rng.randrange(len(means))] for _ in means]
        replicates.append(sum(sample) / len(sample))
    replicates.sort()
    std = math.sqrt(sum((item - mean) ** 2 for item in means) / (len(means) - 1))
    return BootstrapSummary(
        mean,
        _quantile(replicates, 0.025),
        _quantile(replicates, 0.975),
        std,
        len(means),
        resamples,
        value,
    )


def _group_values(
    rows: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]], value: str
) -> dict[str, list[float]]:
    """Normalize supported row representations and validate finite values."""
    if isinstance(rows, Mapping):
        grouped = {
            str(key): [float(item) for item in values] for key, values in rows.items()
        }
    else:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            pocket = row.get("pocket_id", row.get("pocket", "default"))
            item = row.get(value)
            if item is None:
                continue
            grouped.setdefault(str(pocket), []).append(float(item))
    for pocket, values in grouped.items():
        if any(not math.isfinite(item) for item in values):
            raise ValueError(f"non-finite metric value in pocket {pocket!r}")
    return grouped


def _quantile(values: Sequence[float], probability: float) -> float:
    """Compute a deterministic linear-interpolated percentile."""
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


__all__ = ["BootstrapSummary", "bootstrap_macro_summary"]
