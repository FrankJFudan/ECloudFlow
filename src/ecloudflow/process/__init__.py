"""Hybrid continuous and categorical stochastic interpolant processes."""

from ecloudflow.process.categorical import CategoricalPath, CategoricalSample
from ecloudflow.process.continuous import ContinuousPath, ContinuousSample
from ecloudflow.process.schedules import CosineBridge, InterpolantSchedule, LinearBridge

__all__ = [
    "CategoricalPath",
    "CategoricalSample",
    "ContinuousPath",
    "ContinuousSample",
    "CosineBridge",
    "InterpolantSchedule",
    "LinearBridge",
]
