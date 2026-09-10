import re
from typing import List, Dict, Any, Optional
import pandas as pd


class CategoricalClassificationTask:
    """Unified task for categorical classification (source type, subclass, etc.).

    Replaces the old SourceClassificationTask and SubclassClassificationTask
    which were near-identical copies differing only in `name` and `target_column`.
    """

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

    def build_prompt(self, *, image_mode: bool, spectrum_text: Optional[str] = None) -> str:
        parts = [
            "Briefly analyze and describe the given spectrum and then classify "
            "the astronomical source into one of the following categories.\n\n"
            f"Allowed categories:\n{self._options_text}\n",
        ]

        if spectrum_text is not None:
            parts.append(f"\n{spectrum_text}\n")

        parts.append(
            "\nYou MUST conclude your response with the exact format:\n"
            "FINAL ANSWER: [Category]"
        )

        return "".join(parts)

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> str:
        if not raw_text:
            return "UNKNOWN"

        labels_str = "|".join(self.categories)
        match = re.search(
            r"FINAL ANSWER:\s*(" + labels_str + r")", raw_text, re.IGNORECASE
        )
        if match:
            matched_lower = match.group(1).lower()
            for cat in self.categories:
                if cat.lower() == matched_lower:
                    return cat
        return "UNKNOWN"

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
