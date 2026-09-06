from typing import Protocol, Any, Dict, List
from dataclasses import dataclass

class EvalTask(Protocol):
    name: str

    def get_prompt_spec(self) -> Any:
        """Return structured prompt components for responders to use."""
        ...

    def default_prompt(self, **kwargs) -> str:
        """Return a default prompt for this task."""
        ...

    def default_parse(self, raw_text: str) -> Any:
        """Default parser to extract structured output from raw model text."""
        ...

    def extract_ground_truth(self, item: Any) -> Any:
        """Extract ground truth for this task from a data row/item."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return serializable configuration/metadata for this task."""
        ...

def get_task(task_type: str, **kwargs) -> EvalTask:
    from .distance_classification import DistanceClassificationTask
    from .emission_lines import EmissionLineTask

    TASK_REGISTRY = {
        "distance_classification": DistanceClassificationTask,
        "emission_lines": EmissionLineTask,
    }

    if task_type not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_type}'. Available tasks: {list(TASK_REGISTRY.keys())}")

    return TASK_REGISTRY[task_type](**kwargs)

