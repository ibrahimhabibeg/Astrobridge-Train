from __future__ import annotations

import re
from typing import Any, Dict, Optional
import pandas as pd

from ..prompts import render_prompt
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
        self.benchmark_name = "subclass" if "subclass" in target_column.lower() else "source_class"
        self.active_classes = active_classes
        self.categories = list(dict.fromkeys(self.active_classes.values()))
        self._options_text = ", ".join(self.categories)

    def build_frontier_prompt(self, caption: str, item: Optional[Dict[str, Any]] = None) -> str:
        template_name = (
            "caption_eval/source_class.jinja2"
            if self.target_column in ("class", "source_class")
            else "caption_eval/subclass.jinja2"
        )
        return render_prompt(
            template_name,
            options=self._options_text,
            caption=caption.strip(),
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
            gt_obj = item["ground_truth"] if "ground_truth" in item else item
            raw_class = "UNKNOWN"
            if isinstance(gt_obj, dict):
                if self.target_column in gt_obj:
                    raw_class = gt_obj[self.target_column]
                elif "source_class" in gt_obj and self.target_column in ("class", "source_class"):
                    raw_class = gt_obj["source_class"]
                elif "subclass" in gt_obj and self.target_column in ("subclass", "source_subclass"):
                    raw_class = gt_obj["subclass"]
                elif "class" in gt_obj:
                    raw_class = gt_obj["class"]

            if raw_class == "UNKNOWN":
                if self.target_column in item:
                    raw_class = item[self.target_column]
                elif "source_class" in item and self.target_column in ("class", "source_class"):
                    raw_class = item["source_class"]
                elif "subclass" in item and self.target_column in ("subclass", "source_subclass"):
                    raw_class = item["subclass"]
        else:
            raise ValueError(f"Cannot extract ground truth from item of type {type(item)}")

        if raw_class in self.active_classes:
            return self.active_classes[raw_class]

        for k, v in self.active_classes.items():
            if k.lower() == str(raw_class).lower():
                return v

        for v in self.categories:
            if v.lower() == str(raw_class).lower():
                return v

        return "UNKNOWN"

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "target_column": self.target_column,
            "active_classes": self.active_classes,
            "categories": self.categories,
        }
