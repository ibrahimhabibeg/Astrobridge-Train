from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union
import pandas as pd

from ..buckets import BucketScheme, get_bucket_scheme
from ..prompts import render_prompt
from .base import Task

BIN_TO_LABEL = {
    "<0.1": "A",
    "0.1-0.5": "B",
    ">0.5": "C",
}


class DistanceTask(Task):
    name: str = "distance"
    benchmark_name: str = "redshift"

    def __init__(self, scheme: Union[BucketScheme, str] = "3-group", **kwargs):
        self.scheme = get_bucket_scheme(scheme) if isinstance(scheme, str) else scheme

    def build_frontier_prompt(self, caption: str, item: Optional[Dict[str, Any]] = None) -> str:
        return render_prompt(
            "caption_eval/distance.jinja2",
            options=self.scheme.format_options(),
            caption=caption.strip(),
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str, **kwargs: Any) -> Optional[str]:
        if not raw_text:
            return None
        labels_str = "".join(self.scheme.labels)
        match = re.search(r"FINAL ANSWER:\s*([" + labels_str + r"])", raw_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"(?:answer is|category)\s*(?:is\s*)?[:\s]*([ " + labels_str + r"])", raw_text, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, (int, float)):
            return self.scheme.classify(float(item))

        if isinstance(item, (dict, pd.Series)):
            gt = item.get("ground_truth", item)
            if isinstance(gt, dict):
                if "redshift_bin" in gt and str(gt["redshift_bin"]) in BIN_TO_LABEL:
                    return BIN_TO_LABEL[str(gt["redshift_bin"])]
                for k in ("z", "Z", "redshift"):
                    if k in gt and gt[k] is not None:
                        return self.scheme.classify(float(gt[k]))

            if "redshift_bin" in item and str(item["redshift_bin"]) in BIN_TO_LABEL:
                return BIN_TO_LABEL[str(item["redshift_bin"])]
            for k in ("z", "Z", "redshift"):
                if k in item and item[k] is not None:
                    return self.scheme.classify(float(item[k]))

        raise ValueError(f"Cannot extract ground truth redshift from: {item}")

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "benchmark_name": self.benchmark_name,
            "bucket_scheme": self.scheme.get_config(),
        }


CaptionDistanceTask = DistanceTask
DistanceClassificationTask = DistanceTask
