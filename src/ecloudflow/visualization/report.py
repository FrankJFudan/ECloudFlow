"""Self-contained HTML report assembly and artifact hashing."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecloudflow.evaluation.types import EvaluationResult
from ecloudflow.visualization.plots import (
    plot_metric_distribution,
    plot_quality_speed_pareto,
)


@dataclass(frozen=True)
class ReportBundle:
    """Describe report HTML, figures, and content hashes."""

    paths: tuple[Path, ...]
    hashes: dict[str, str]

    @property
    def html_path(self) -> Path:
        """Return the primary report HTML path."""
        for path in self.paths:
            if path.name == "report.html":
                return path
        raise ValueError("report bundle has no report.html")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report manifest."""
        return {
            "paths": [str(path) for path in self.paths],
            "hashes": dict(self.hashes),
        }


def build_report(
    evaluation: EvaluationResult | Any,
    output_dir: str | Path,
    top_n: int = 20,
) -> ReportBundle:
    """Build a self-contained scientific HTML and publication figure bundle.

    :param evaluation: Per-molecule, per-pocket, aggregate, and provenance
        results.  Plain mappings are accepted for lightweight smoke reports.
    :param output_dir: Destination for HTML, SVG, PDF, PNG, and data tables.
    :param top_n: Number of top rows summarized in the HTML.
    :return: Paths and SHA-256 hashes for generated artifacts.
    :rtype: ReportBundle
    :raises ValueError: If ``top_n`` is not positive.

    The report only renders already-computed metric values.  It never reruns a
    docking tool or changes generated molecules, and all writes use sibling
    temporary files followed by replacement.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = (
        evaluation.as_dict()
        if isinstance(evaluation, EvaluationResult)
        else _safe(evaluation)
    )
    rows = (
        _context_molecule_rows(evaluation.context)
        if isinstance(evaluation, EvaluationResult)
        else _extract_rows(payload)
    )
    if not rows:
        rows = _extract_rows(payload)
    figure_paths = [
        plot_metric_distribution(rows, destination / "metric_distribution.svg"),
        plot_quality_speed_pareto(rows, destination / "quality_speed_pareto.svg"),
    ]
    # PDF and PNG are separate publication artifacts with the same deterministic
    # data; a compact metric figure keeps the report useful for small fixtures.
    figure_paths.extend(
        [
            plot_metric_distribution(rows, destination / "metric_distribution.pdf"),
            plot_metric_distribution(rows, destination / "metric_distribution.png"),
        ]
    )
    report_path = destination / "report.html"
    summary_path = destination / "report.json"
    visible_rows = rows[:top_n]
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key in _table_keys(visible_rows)
        )
        + "</tr>"
        for row in visible_rows
    )
    keys = _table_keys(visible_rows)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ECloudFlow report</title>
<style>body{{font-family:Arial,sans-serif;color:#18222b;background:#f5f7f8;margin:0;padding:24px}}main{{max-width:1280px;margin:auto}}section{{background:white;border:1px solid #ccd7dd;padding:16px;margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7dd;padding:6px;text-align:left}}img{{max-width:100%}}</style></head>
<body><main><h1>ECloudFlow evaluation report</h1><section><h2>Top molecules and metrics</h2><table><thead><tr>{"".join(f"<th>{html.escape(key)}</th>" for key in keys)}</tr></thead><tbody>{table}</tbody></table></section>
<section><h2>Publication figures</h2><img src="metric_distribution.svg" alt="metric distribution"><img src="quality_speed_pareto.svg" alt="quality-speed Pareto"></section>
<section id="electron-cloud"><h2>Electron cloud and provenance</h2><pre>{html.escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></section></main></body></html>"""
    _atomic_text(report_path, document)
    _atomic_text(summary_path, json.dumps(_safe(payload), indent=2, sort_keys=True))
    paths = (report_path, summary_path, *figure_paths)
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    return ReportBundle(paths=paths, hashes=hashes)


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    """Find a list of metric rows in common evaluation payload shapes."""
    if isinstance(payload, dict):
        for key in ("rows", "ranked", "molecules", "valid", "per_item"):
            value = payload.get(key)
            if isinstance(value, list):
                return [
                    dict(item) if isinstance(item, dict) else {"value": item}
                    for item in value
                ]
        return [{"metric": key, "value": value} for key, value in payload.items()]
    return [{"value": payload}]


def _context_molecule_rows(context: Any) -> list[dict[str, Any]]:
    """Serialize direct evaluation context molecules without losing metrics.

    :param context: Optional :class:`EvaluationContext` retained by an
        :class:`EvaluationResult`.
    :return: JSON-safe molecule rows in the context's ranked-or-generated
        ordering, or an empty list when no serializable molecular records exist.
    :rtype: list[dict[str, typing.Any]]

    ``EvaluationResult.as_dict`` deliberately serializes only aggregate metric
    results. A direct Python caller, however, supplied molecular records that
    may already contain docking, QED, SA, and timing values. Extracting their
    ``as_dict`` representation here keeps the report figures and top-molecule
    table equivalent to the CLI artifact workflow without mutating evaluation
    state or recomputing any scientific metric.
    """
    if context is None:
        return []
    values = getattr(context, "molecules", ())
    rows: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            rows.append(_safe(value))
            continue
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            serialized = as_dict()
            if isinstance(serialized, dict):
                rows.append(_safe(serialized))
    return rows


def _table_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable table columns."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys or ["value"]


def _atomic_text(path: Path, content: str) -> None:
    """Write text through a sibling partial path."""
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _safe(value: Any) -> Any:
    """Recursively convert report values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "as_dict"):
        return _safe(value.as_dict())
    return str(value)


__all__ = ["ReportBundle", "build_report"]
