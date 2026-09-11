from __future__ import annotations

from typing import Any, Dict

from .base import CaptionEvalTask
from .distance import CaptionDistanceTask
from .emission_lines import CaptionEmissionLineTask
from .source import CategoricalCaptionTask


TASK_REGISTRY = {
    "caption_distance": CaptionDistanceTask,
    "caption_emission_lines": CaptionEmissionLineTask,
    "caption_source": lambda **kw: CategoricalCaptionTask(
        name="caption_source", target_column="class", **kw
    ),
    "caption_subclass": lambda **kw: CategoricalCaptionTask(
        name="caption_subclass", target_column="subclass", **kw
    ),
}


def get_caption_task(task_type: str, **kwargs) -> CaptionEvalTask:
    """Instantiate a caption evaluation task."""
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown caption task '{task_type}'. Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_type](**kwargs)


__all__ = [
    "CaptionEvalTask",
    "CaptionDistanceTask",
    "CaptionEmissionLineTask",
    "CategoricalCaptionTask",
    "get_caption_task",
]

