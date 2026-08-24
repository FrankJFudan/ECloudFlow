"""Named metric registry and seven-domain evaluation orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ecloudflow.evaluation.types import (
    EvaluationContext,
    EvaluationResult,
    Metric,
    MetricResult,
)

GROUPS = (
    "chemistry",
    "distribution",
    "geometry",
    "binding",
    "ecloud",
    "conditional",
    "efficiency",
)


@dataclass(frozen=True)
class MetricRegistry:
    """Resolve stable metric names and group membership.

    :param metrics: Named metric implementations.
    :return: Immutable registry.  ``groups`` always exposes all seven domain
        names, including domains whose optional metrics are unavailable.
    :rtype: MetricRegistry
    """

    metrics: Mapping[str, Metric]

    def __post_init__(self) -> None:
        """Freeze the metric mapping and reject duplicate names implicitly."""
        object.__setattr__(self, "metrics", dict(self.metrics))

    @property
    def groups(self) -> dict[str, tuple[Metric, ...]]:
        """Return metrics grouped under every canonical evaluation domain."""
        grouped: dict[str, list[Metric]] = {group: [] for group in GROUPS}
        for metric in self.metrics.values():
            if metric.group not in grouped:
                raise ValueError(f"unknown metric group: {metric.group!r}")
            grouped[metric.group].append(metric)
        return {group: tuple(values) for group, values in grouped.items()}

    def get(self, name: str) -> Metric:
        """Return a named metric or raise a clear configuration error."""
        try:
            return self.metrics[name]
        except KeyError as error:
            raise KeyError(f"unknown evaluation metric: {name}") from error

    @classmethod
    def default(cls) -> MetricRegistry:
        """Build the standard registry without requiring optional binaries."""
        from ecloudflow.evaluation.binding import (
            DockingSuccessMetric,
            PLIFMetric,
            ShapeContactMetric,
            VinaScoreMetric,
        )
        from ecloudflow.evaluation.chemistry import (
            PoseBustersMetric,
            RDKitValidityMetric,
            UniquenessMetric,
        )
        from ecloudflow.evaluation.conditional import (
            FragmentSuccessMetric,
            PocketConditionMetric,
            PropertySuccessMetric,
        )
        from ecloudflow.evaluation.distribution import (
            DescriptorJSDMetric,
            FCDMetric,
            InternalDiversityMetric,
            NoveltyMetric,
        )
        from ecloudflow.evaluation.ecloud import (
            DipoleErrorMetric,
            ElectronCountMetric,
            ElectronCycleMetric,
            ElectronDensityMetric,
        )
        from ecloudflow.evaluation.efficiency import (
            GenerationEfficiencyMetric,
            MemoryMetric,
            NFEMetric,
            ScalingMetric,
        )
        from ecloudflow.evaluation.geometry import (
            AngleJSDMetric,
            BondLengthJSDMetric,
            ClashMetric,
            DihedralJSDMetric,
            PoseGeometryMetric,
            RawRelaxedRMSDMetric,
        )

        values = (
            RDKitValidityMetric(),
            PoseBustersMetric(),
            UniquenessMetric(),
            NoveltyMetric(),
            InternalDiversityMetric(),
            DescriptorJSDMetric(),
            FCDMetric(),
            PoseGeometryMetric(),
            ClashMetric(),
            RawRelaxedRMSDMetric(),
            BondLengthJSDMetric(),
            AngleJSDMetric(),
            DihedralJSDMetric(),
            VinaScoreMetric(),
            DockingSuccessMetric(),
            PLIFMetric(),
            ShapeContactMetric(),
            ElectronCountMetric(),
            DipoleErrorMetric(),
            ElectronDensityMetric(),
            ElectronCycleMetric(),
            FragmentSuccessMetric(),
            PocketConditionMetric(),
            PropertySuccessMetric(),
            GenerationEfficiencyMetric(),
            NFEMetric(),
            MemoryMetric(),
            ScalingMetric(),
        )
        return cls({metric.name: metric for metric in values})


def evaluate_run(
    context: EvaluationContext,
    *,
    registry: MetricRegistry | None = None,
    groups: Iterable[str] | None = None,
    config: Any = None,
) -> EvaluationResult:
    """Compute selected metrics while retaining unavailable statuses.

    :param context: Read-only evaluation inputs.
    :param registry: Custom registry; defaults to :meth:`MetricRegistry.default`.
    :param groups: Optional subset of canonical groups.  Unknown groups fail
        before any metric is run.
    :param config: Optional :class:`EvaluationConfig`; selected groups and
        raw/relaxed policy are read when explicit ``groups`` are absent.
    :return: Named metric results in deterministic metric-name order.
    :rtype: EvaluationResult
    """
    if not isinstance(context, EvaluationContext):
        from ecloudflow.evaluation.types import coerce_context

        context = coerce_context(context)
    if groups is not None:
        selected = tuple(groups)
    elif config is not None:
        selected = tuple(config.groups)
    else:
        selected = GROUPS
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown evaluation groups: {unknown}")
    active = registry or MetricRegistry.default()
    results: dict[str, MetricResult] = {}
    for group in selected:
        for metric in active.groups[group]:
            results[metric.name] = metric.compute(context)
    return EvaluationResult(results=results, context=context)


__all__ = ["GROUPS", "MetricRegistry", "evaluate_run"]
