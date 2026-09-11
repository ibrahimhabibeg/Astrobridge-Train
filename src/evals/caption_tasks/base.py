from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class CaptionEvalTask(Protocol):
    """Protocol defining a caption evaluation task where a Frontier model predicts properties from a caption."""
    name: str

    def build_frontier_prompt(self, caption: str) -> str:
        """Construct the prompt sent to the Frontier VLM containing the caption and task instructions."""
        ...

    def fallback_tag(self) -> str:
        """Tag to append when re-prompting the frontier model on formatting failures."""
        ...

    def default_parse(self, raw_text: str) -> Any:
        """Extract structured prediction from raw text returned by the frontier model."""
        ...

    def extract_ground_truth(self, item: Any) -> Any:
        """Extract ground truth label/value for this task from a dataset row/item."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return serializable task metadata and configuration."""
        ...

