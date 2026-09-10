import re
from typing import Dict, Any, Union, Optional
import pandas as pd

from ..buckets import BucketScheme, get_bucket_scheme


class DistanceClassificationTask:
    name: str = "distance_classification"

    def __init__(self, scheme: Union[BucketScheme, str] = "3-group", **kwargs):
        if isinstance(scheme, str):
            self.scheme = get_bucket_scheme(scheme)
        else:
            self.scheme = scheme

    def build_prompt(
        self, *, image_mode: bool, spectrum_text: Optional[str] = None
    ) -> str:
        options = self.scheme.format_options()

        if image_mode:
            intro = (
                "Briefly analyze and describe the given spectrum and then classify "
                "the distance of the observed astronomical object into one of the "
                f"following categories:\n{options}.\n"
            )
        elif spectrum_text is not None:
            intro = (
                "Briefly analyze and describe the given spectrum and then classify "
                "the distance of the observed astronomical object into one of the "
                f"following categories:\n{options}.\n\n"
                f"{spectrum_text}\n\n"
            )
        else:
            intro = (
                "Based on the spectrum provided, classify the distance of the "
                "observed astronomical object into one of the following categories: "
                f"{options}. "
                "Think step-by-step, but you MUST conclude with the exact phrase "
            )
            return intro + "'FINAL ANSWER: [Letter]'"

        return (
            intro
            + "You MUST conclude your response with the exact format:\nFINAL ANSWER: [Letter]"
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None
        labels_str = "".join(self.scheme.labels)
        match = re.search(
            r"FINAL ANSWER:\s*([" + labels_str + r"])", raw_text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        return None

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, (int, float)):
            return self.scheme.classify(float(item))
        elif isinstance(item, (dict, pd.Series)):
            z = item["Z"]
            return self.scheme.classify(float(z))
        raise ValueError(
            f"Cannot extract ground truth Z from item of type {type(item)}"
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "bucket_scheme": self.scheme.get_config(),
        }
