import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union
import pandas as pd

@dataclass
class SourceClassPromptSpec:
    categories: List[str]
    output_format_tag: str
    options_text: str

class SourceClassificationTask:
    name: str = "source_classification"

    def __init__(self, active_classes: Dict[str, str], **kwargs):
        self.active_classes = active_classes
        self.categories = list(self.active_classes.values())
        self.spec = SourceClassPromptSpec(
            categories=self.categories,
            output_format_tag="FINAL ANSWER",
            options_text=", ".join(self.categories),
        )

    def get_prompt_spec(self) -> SourceClassPromptSpec:
        return self.spec

    def default_prompt(self, **kwargs) -> str:
        return (
            "Briefly analyze and describe the given spectrum and then classify the astronomical source into one of the following categories.\n\n"
            f"Allowed categories:\n{self.spec.options_text}\n\n"
            "You MUST conclude your response with the exact format:\n"
            "FINAL ANSWER: [Category]"
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> str:
        if not raw_text:
            return "UNKNOWN"
            
        labels_str = "|".join(self.categories)
        # Search for FINAL ANSWER: <Category> (case-insensitive for the prefix, we will allow case-insensitive for categories too)
        match = re.search(r"FINAL ANSWER:\s*(" + labels_str + r")", raw_text, re.IGNORECASE)
        if match:
            # We want to return the exact case from self.categories
            matched_category_lower = match.group(1).lower()
            for cat in self.categories:
                if cat.lower() == matched_category_lower:
                    return cat
        return "UNKNOWN"

    def extract_ground_truth(self, item: Any) -> str:
        # Assuming item is the raw dataset class string (e.g., 'GALAXY')
        if isinstance(item, str):
            raw_class = item
        elif isinstance(item, (dict, pd.Series)):
            raw_class = item["class"]
        else:
            raise ValueError(f"Cannot extract ground truth from item of type {type(item)}")
            
        # Check if the raw_class is in active_classes
        if raw_class in self.active_classes:
            return self.active_classes[raw_class]
            
        return "UNKNOWN"

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "active_classes": self.active_classes,
            "categories": self.categories,
        }

