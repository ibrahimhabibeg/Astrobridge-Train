from typing import Protocol, Any, Dict, Optional


class EvalTask(Protocol):
    name: str

    def build_prompt(
        self, *, image_mode: bool, spectrum_text: Optional[str] = None
    ) -> str:
        """Build the complete evaluation prompt for this task.

        Three calling patterns:
          task.build_prompt(image_mode=True)                       # Vision models (base_qwen, gemini)
          task.build_prompt(image_mode=False, spectrum_text=data)   # Text model (base_qwen_text)
          task.build_prompt(image_mode=False)                       # Native model (astrobridge)
        """
        ...

    def fallback_tag(self) -> str:
        """Return the fallback tag to append when retrying failed parses."""
        ...

    def default_parse(self, raw_text: str) -> Any:
        """Extract structured output from raw model text."""
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
    from .source_classification import CategoricalClassificationTask

    TASK_REGISTRY = {
        "distance_classification": DistanceClassificationTask,
        "emission_lines": EmissionLineTask,
        "source_classification": lambda **kw: CategoricalClassificationTask(
            name="source_classification", target_column="class", **kw
        ),
        "subclass_classification": lambda **kw: CategoricalClassificationTask(
            name="subclass_classification", target_column="subclass", **kw
        ),
    }

    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_type}'. Available tasks: {list(TASK_REGISTRY.keys())}"
        )

    return TASK_REGISTRY[task_type](**kwargs)
