import re
from typing import Dict, Any, Optional
import pandas as pd

from ..prompts import render_prompt


class CategoricalClassificationTask:
    """Unified task for categorical classification (source type, subclass, etc.)."""

    def __init__(
        self,
        active_classes: Dict[str, str],
        name: str = "source_classification",
        target_column: str = "class",
        **kwargs,
    ):
        self.name = name
        self.target_column = target_column
        self.active_classes = active_classes
        self.categories = list(self.active_classes.values())
        self._options_text = ", ".join(self.categories)

    def build_prompt(
        self, *, image_mode: bool = False, spectrum_text: Optional[str] = None
    ) -> str:
        is_subclass = self.target_column not in ("class", "source_class")
        if spectrum_text is not None:
            template = (
                "direct_eval/source_subclass_text.jinja2"
                if is_subclass
                else "direct_eval/source_class_text.jinja2"
            )
            return render_prompt(
                template, options=self._options_text, spectrum_text=spectrum_text
            )
        else:
            template = (
                "direct_eval/source_subclass_image.jinja2"
                if is_subclass
                else "direct_eval/source_class_image.jinja2"
            )
            return render_prompt(template, options=self._options_text)

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        labels_str = "|".join(self.categories)
        match = re.search(
            r"FINAL ANSWER:\s*(" + labels_str + r")", raw_text, re.IGNORECASE
        )
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
            raise ValueError(
                f"Cannot extract ground truth from item of type {type(item)}"
            )

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
