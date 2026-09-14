from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union
import pandas as pd

from ..buckets import BucketScheme, get_bucket_scheme
from .base import CaptionEvalTask

BIN_TO_LABEL = {
    "<0.1": "A",
    "0.1-0.5": "B",
    ">0.5": "C",
}


class CaptionDistanceTask(CaptionEvalTask):
    """Evaluates the model's ability to describe distance/redshift features in spectra."""
    name: str = "caption_distance"

    def __init__(self, scheme: Union[BucketScheme, str] = "3-group", **kwargs):
        if isinstance(scheme, str):
            self.scheme = get_bucket_scheme(scheme)
        else:
            self.scheme = scheme

    def build_frontier_prompt(self, caption: str, item: Optional[Dict[str, Any]] = None) -> str:
        options = self.scheme.format_options()
        return (
            "You are an expert astrophysicist. You will be provided with a scientific description of an astronomical spectrum.\n"
            "Based ONLY on the description of the spectrum, classify the distance of the observed astronomical object "
            "into one of the following categories.\n\n"
            f"Allowed categories:\n{options}\n\n"
            f"Spectrum Description:\n\"\"\"\n{caption.strip()}\n\"\"\"\n\n"
            "Think step-by-step, but you MUST conclude your response with the exact format:\n"
            "FINAL ANSWER: [Letter]"
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

        if isinstance(item, (dict, pd.Series)):
            gt_obj = item["ground_truth"] if "ground_truth" in item else item
            if isinstance(gt_obj, dict):
                if "redshift_bin" in gt_obj and str(gt_obj["redshift_bin"]) in BIN_TO_LABEL:
                    return BIN_TO_LABEL[str(gt_obj["redshift_bin"])]
                if "z" in gt_obj:
                    return self.scheme.classify(float(gt_obj["z"]))
                if "Z" in gt_obj:
                    return self.scheme.classify(float(gt_obj["Z"]))

            if "redshift_bin" in item and str(item["redshift_bin"]) in BIN_TO_LABEL:
                return BIN_TO_LABEL[str(item["redshift_bin"])]
            if "z" in item:
                return self.scheme.classify(float(item["z"]))
            if "Z" in item:
                return self.scheme.classify(float(item["Z"]))

        raise ValueError(f"Cannot extract ground truth Z from item of type {type(item)}: {item}")

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "bucket_scheme": self.scheme.get_config(),
        }
