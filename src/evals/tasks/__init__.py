from __future__ import annotations

from typing import Any, Dict

from .base import Task, CaptionEvalTask
from .distance import DistanceTask, CaptionDistanceTask, DistanceClassificationTask
from .source import SourceTask
from .subclass import SubclassTask
from .emission_lines import EmissionLineTask, CaptionEmissionLineTask

TASK_REGISTRY = {
    "distance": DistanceTask,
    "caption_distance": DistanceTask,
    "distance_classification": DistanceTask,
    "source": SourceTask,
    "caption_source": SourceTask,
    "source_classification": SourceTask,
    "subclass": SubclassTask,
    "caption_subclass": SubclassTask,
    "subclass_classification": SubclassTask,
    "emission_lines": EmissionLineTask,
    "caption_emission_lines": EmissionLineTask,
}


def get_task(task_type: str, **kwargs: Any) -> Task:
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_type}'. Available tasks: {sorted(list(TASK_REGISTRY.keys()))}"
        )
    return TASK_REGISTRY[task_type](**kwargs)


get_caption_task = get_task

__all__ = [
    "Task",
    "CaptionEvalTask",
    "DistanceTask",
    "CaptionDistanceTask",
    "DistanceClassificationTask",
    "SourceTask",
    "SubclassTask",
    "EmissionLineTask",
    "CaptionEmissionLineTask",
    "get_task",
    "get_caption_task",
]
