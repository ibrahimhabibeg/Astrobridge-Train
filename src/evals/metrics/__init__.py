"""Evaluation metrics package.

Provides modular, reusable metric functions for single-label classification
and multi-label detection tasks.

Quick reference:

    # Use individual metric functions
    from evals.metrics.classification import accuracy, macro_f1, ordinal_mae
    from evals.metrics.multilabel import sample_precision_recall_f1, snr_weighted_metrics

    # Or use the full pipeline (compute everything + save to disk)
    from evals.metrics import compute_and_save_metrics
"""

# Re-export the pipeline entry point for backward compatibility
from .pipeline import compute_and_save_metrics

__all__ = ["compute_and_save_metrics"]

