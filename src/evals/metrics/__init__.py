from __future__ import annotations

from .classification import accuracy, confusion_matrix_dict
from .multilabel import mean_jaccard_index
from .reporter import compute_caption_metrics

compute_metrics = compute_caption_metrics

__all__ = [
    "accuracy",
    "confusion_matrix_dict",
    "mean_jaccard_index",
    "compute_caption_metrics",
    "compute_metrics",
]
