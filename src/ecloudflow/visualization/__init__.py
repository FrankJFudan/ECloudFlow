"""Interactive molecular/electron viewers and publication report helpers."""

from ecloudflow.visualization.fields import render_electron_field_html
from ecloudflow.visualization.molecule import render_complex_html
from ecloudflow.visualization.plots import (
    plot_ecdf,
    plot_geometry_heatmap,
    plot_metric_distribution,
    plot_property_distribution,
    plot_quality_speed_pareto,
    plot_violin,
)
from ecloudflow.visualization.report import ReportBundle, build_report

__all__ = [
    "ReportBundle",
    "build_report",
    "plot_ecdf",
    "plot_geometry_heatmap",
    "plot_metric_distribution",
    "plot_property_distribution",
    "plot_quality_speed_pareto",
    "plot_violin",
    "render_complex_html",
    "render_electron_field_html",
]
