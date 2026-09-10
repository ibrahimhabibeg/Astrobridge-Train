"""Single-label classification metrics.

Pure functions operating on arrays/lists. No file I/O, no pandas, no task objects.
Every function takes ground-truth and prediction arrays and returns a scalar or dict.
"""

from typing import List, Optional, Dict, Any, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    classification_report as sklearn_classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Fraction of predictions that exactly match the ground truth.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (same length as y_true).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Optional[List[str]] = None) -> float:
    """Unweighted mean of per-class F1 scores.

    Each class contributes equally to the average, regardless of how many
    samples belong to it. Useful for evaluating performance on rare classes.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of valid label names. If None, inferred from data.

    Returns:
        Macro-averaged F1 as a float in [0, 1].
    """
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def weighted_f1(y_true: Sequence, y_pred: Sequence, labels: Optional[List[str]] = None) -> float:
    """Support-weighted mean of per-class F1 scores.

    Each class's F1 is weighted by its number of true instances (support).
    Reflects overall performance more faithfully when classes are imbalanced.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of valid label names. If None, inferred from data.

    Returns:
        Weighted F1 as a float in [0, 1].
    """
    return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))


def ordinal_mae(y_true: Sequence, y_pred: Sequence, labels: List[str]) -> float:
    """Mean absolute error on the ordinal index of each label.

    Only meaningful when labels represent an ordered scale (e.g.,
    A=nearby, B=intermediate, C=far). An error of A→C costs 2, while
    A→B costs 1. Do NOT use for nominal categories where label order
    is arbitrary.

    Args:
        y_true: Ground-truth labels (must all be in ``labels``).
        y_pred: Predicted labels (must all be in ``labels``).
        labels: Ordered list of labels defining the ordinal scale.

    Returns:
        MAE in label-index units (e.g., 0.5 means predictions are off
        by half a category on average).
    """
    label_map = {label: i for i, label in enumerate(labels)}
    y_true_idx = [label_map[y] for y in y_true]
    y_pred_idx = [label_map[y] for y in y_pred]
    return float(mean_absolute_error(y_true_idx, y_pred_idx))


def format_error_rate(y_pred: Sequence, valid_labels: List[str]) -> float:
    """Fraction of predictions that are not in the set of valid labels.

    A format error occurs when the model's output could not be parsed
    into any recognized category (e.g., the model rambled instead of
    following the "FINAL ANSWER: X" format).

    Args:
        y_pred: Raw predicted labels (may contain invalid entries).
        valid_labels: The set of acceptable label values.

    Returns:
        Format error rate as a float in [0, 1].
    """
    valid_set = set(valid_labels)
    n_errors = sum(1 for p in y_pred if p not in valid_set)
    return n_errors / len(y_pred) if len(y_pred) > 0 else 0.0


def confusion_matrix_dict(
    y_true: Sequence, y_pred: Sequence, labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Confusion matrix as a nested dict: ``{true_label: {pred_label: count}}``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of label names.

    Returns:
        Nested dict where ``result[true][pred]`` is the count.
    """
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    return {
        t: {p: int(cm[i][j]) for j, p in enumerate(labels)}
        for i, t in enumerate(labels)
    }


def per_class_report(
    y_true: Sequence, y_pred: Sequence, labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-class precision, recall, F1, and support.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of label names.

    Returns:
        Dict mapping each label to ``{precision, recall, f1, support}``.
    """
    report = sklearn_classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    result = {}
    for label in labels:
        entry = dict(report[label])
        entry["f1"] = entry.pop("f1-score", 0.0)
        result[label] = entry
    return result


def classification_report(
    y_true: Sequence,
    y_pred: Sequence,
    labels: List[str],
    *,
    ordinal: bool = False,
) -> Dict[str, Any]:
    """Full classification report bundle.

    Computes accuracy, macro/weighted F1, format error stats, confusion
    matrix, and per-class metrics. Optionally includes ordinal MAE when
    ``ordinal=True`` (only for tasks where labels have a natural order).

    Args:
        y_true: Ground-truth labels (may contain values outside ``labels``
            if the ground truth itself has unknowns, but typically all valid).
        y_pred: Predicted labels (may contain invalid values — these are
            counted as format errors and excluded from metric computation).
        labels: Ordered list of valid label names.
        ordinal: If True, compute and include ordinal MAE.

    Returns:
        Dict with keys: total_samples, format_errors, format_error_rate,
        global_accuracy, macro_f1, weighted_f1, confusion_matrix,
        per_class_metrics. If ordinal=True, also includes
        mean_absolute_error.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    total = len(y_true)

    valid_set = set(labels)
    valid_indices = [i for i in range(total) if y_pred[i] in valid_set]
    n_format_errors = total - len(valid_indices)

    result: Dict[str, Any] = {
        "total_samples": total,
        "format_errors": n_format_errors,
        "format_error_rate": n_format_errors / total if total > 0 else 0.0,
    }

    if not valid_indices:
        # All predictions were format errors — can't compute anything
        result.update({
            "global_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": {},
            "per_class_metrics": {},
        })
        if ordinal:
            result["mean_absolute_error"] = float("nan")
        return result

    yt = [y_true[i] for i in valid_indices]
    yp = [y_pred[i] for i in valid_indices]

    result["global_accuracy"] = accuracy(yt, yp)
    result["macro_f1"] = macro_f1(yt, yp, labels=labels)
    result["weighted_f1"] = weighted_f1(yt, yp, labels=labels)
    result["confusion_matrix"] = confusion_matrix_dict(yt, yp, labels)
    result["per_class_metrics"] = per_class_report(yt, yp, labels)

    if ordinal:
        result["mean_absolute_error"] = ordinal_mae(yt, yp, labels)

    return result

