import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union
import pandas as pd

from ..buckets import BucketScheme, get_bucket_scheme

@dataclass
class DistanceClassPromptSpec:
    categories: List[str]
    category_descriptions: List[str]
    output_format_tag: str
    options_text: str
    options_multiline: str
    scheme: BucketScheme

class DistanceClassificationTask:
    name: str = "distance_classification"

    def __init__(self, scheme: Union[BucketScheme, str] = "3-group", **kwargs):
        if isinstance(scheme, str):
            self.scheme = get_bucket_scheme(scheme)
        else:
            self.scheme = scheme
        
        self.spec = DistanceClassPromptSpec(
            categories=self.scheme.labels,
            category_descriptions=self.scheme.descriptions,
            output_format_tag="FINAL ANSWER",
            options_text=self.scheme.format_options(),
            options_multiline=self.scheme.format_options_multiline(),
            scheme=self.scheme,
        )

    def get_prompt_spec(self) -> DistanceClassPromptSpec:
        return self.spec

    def default_prompt(self, **kwargs) -> str:
        return (
            "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
            f"{self.spec.options_text}. "
            "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> str:
        if not raw_text:
            return "UNKNOWN"
        labels_str = "".join(self.spec.categories)
        match = re.search(r"FINAL ANSWER:\s*([" + labels_str + r"])", raw_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "UNKNOWN"

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, (int, float)):
            return self.scheme.classify(float(item))
        elif isinstance(item, (dict, pd.Series)):
            z = item["Z"]
            return self.scheme.classify(float(z))
        raise ValueError(f"Cannot extract ground truth Z from item of type {type(item)}")

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "bucket_scheme": self.scheme.get_config(),
        }

