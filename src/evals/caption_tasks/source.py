from __future__ import annotations

import re
from typing import Any, Dict, Optional
import pandas as pd

from .base import CaptionEvalTask


class CategoricalCaptionTask(CaptionEvalTask):
    """Evaluates the model's ability to describe features identifying astronomical source class or subclass."""

    def __init__(
        self,
        active_classes: Dict[str, str],
        name: str = "caption_source",
        target_column: str = "class",
        **kwargs,
    ):
        self.name = name
        self.target_column = target_column
        self.active_classes = active_classes
        self.categories = list(self.active_classes.values())
        self._options_text = ", ".join(self.categories)

    def build_frontier_prompt(self, caption: str) -> str:
        target_desc = "source class" if self.target_column == "class" else "source subclass"
        return (
            "You are an expert astrophysicist. You will be provided with a scientific description of an astronomical spectrum.\n"
            f"Based ONLY on the description of the spectrum, classify the astronomical {target_desc} into one of the following categories.\n\n"
            f"Allowed categories:\n{self._options_text}\n\n"
            f"Spectrum Description:\n\"\"\"\n{caption.strip()}\n\"\"\"\n\n"
            "Think step-by-step, but you MUST conclude your response with the exact format:\n"
            "FINAL ANSWER: [Category]"
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        labels_str = "|".join(re.escape(cat) for cat in self.categories)
        match = re.search(r"FINAL ANSWER:\s*(" + labels_str + r")", raw_text, re.IGNORECASE)
        if match:
            matched_lower = match.group(1).lower()
            for cat in self.categories:
                if cat.lower() == matched_lower:
                    return cat

        return None

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, str):
            raw_class = item
        elif isinstance(item, (dict, pd.Series)):
            raw_class = item[self.target_column]
        else:
            raise ValueError(f"Cannot extract ground truth from item of type {type(item)}")

        if raw_class in self.active_classes:
            return self.active_classes[raw_class]

        return "UNKNOWN"

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "target_column": self.target_column,
            "active_classes": self.active_classes,
            "categories": self.categories,
        }

