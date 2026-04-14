"""Systems benchmarks: latency, small-model, data loading.

These tasks share the tracking physics baseline but add operational
constraints on top (wall-clock time, parameter budget, I/O throughput).
"""

from colliderml.tasks.systems.dataloading import DataLoadingTask
from colliderml.tasks.systems.latency import TrackingLatencyTask
from colliderml.tasks.systems.smallmodel import TrackingSmallModelTask

__all__ = [
    "TrackingLatencyTask",
    "TrackingSmallModelTask",
    "DataLoadingTask",
]
