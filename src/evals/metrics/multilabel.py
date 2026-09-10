"""Multi-label metrics for emission line detection.

Terminology used throughout this module:

- **gt_sets**: List of ground-truth label sets, one per sample.
  Example: [{"Hα", "[O III] 5007"}, {"Hβ"}, set()]

- **pred_sets**: List of predicted label sets, one per sample.
  Example: [{"Hα"}, {"Hβ", "Hγ"}, set()]

- **gt_dicts**: List of ground-truth dicts mapping label → SNR value.
  Example: [{"Hα": 42.3, "[O III] 5007": 12.1}, {"Hβ": 3.5}, {}]

The four F1 variants computed here measure different things:

1. **Sample-mean F1**: Average of per-sample F1. Every spectrum
   contributes equally regardless of how many lines it contains.
   This is the primary headline metric.

2. **Micro F1**: Pools all TP/FP/FN across samples before computing
   F1. Samples with many lines contribute more. Biased toward
   common lines.

3. **Macro-label F1**: Average of per-line F1 scores. Measures
   whether the model is equally good at detecting all lines, vs.
   only common ones.

4. **SNR-weighted F1**: Like sample-mean F1, but recall is weighted
   by log(1 + SNR). Missing a bright line hurts more than missing
   a faint one.
"""

from typing import List, Dict, Set, Any, Optional
import numpy as np
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.preprocessing import MultiLabelBinarizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    """Divide num/den, returning fallback when den is zero."""
    return num / den if den > 0 else fallback


# ---------------------------------------------------------------------------
# Sample-level metrics
# ---------------------------------------------------------------------------

def sample_precision_recall_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    """Per-sample precision, recall, and F1, averaged across samples.

    For each sample independently:
      - Precision = |predicted ∩ ground_truth| / |predicted|
      - Recall    = |predicted ∩ ground_truth| / |ground_truth|
      - F1        = harmonic mean of precision and recall

    Then each metric is averaged across all samples. Every sample
    contributes equally, regardless of how many labels it has.

    Edge cases per sample:
      - No predictions AND no ground truth -> P=1, R=1, F1=1 (perfect empty match)
      - No predictions but ground truth exists -> P=1, R=0, F1=0
      - Predictions but no ground truth -> P=0, R=1, F1=0

    Returns:
        Dict with keys: mean_precision, mean_recall, mean_f1.
    """
    precisions = []
    recalls = []
    f1s = []

    for gt, pred in zip(gt_sets, pred_sets):
        tp = len(gt & pred)
        n_pred = len(pred)
        n_gt = len(gt)

        if n_pred == 0 and n_gt == 0:
            p, r, f1 = 1.0, 1.0, 1.0
        elif n_pred == 0:
            p, r, f1 = 1.0, 0.0, 0.0
        elif n_gt == 0:
            p, r, f1 = 0.0, 1.0, 0.0
        else:
            p = tp / n_pred
            r = tp / n_gt
            f1 = _safe_div(2 * p * r, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


# ---------------------------------------------------------------------------
# Micro-level metrics
# ---------------------------------------------------------------------------

def micro_precision_recall_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    """Pool all TPs, FPs, and FNs across every sample, then compute F1.

    Unlike sample-mean F1, samples with more labels contribute more
    to the final score. This metric is biased toward frequently
    occurring labels.

    Returns:
        Dict with keys: micro_precision, micro_recall, micro_f1.
    """
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for gt, pred in zip(gt_sets, pred_sets):
        tp = len(gt & pred)
        total_tp += tp
        total_pred += len(pred)
        total_gt += len(gt)

    mic_p = _safe_div(total_tp, total_pred)
    mic_r = _safe_div(total_tp, total_gt)
    mic_f1 = _safe_div(2 * mic_p * mic_r, mic_p + mic_r)

    return {
        "micro_precision": float(mic_p),
        "micro_recall": float(mic_r),
        "micro_f1": float(mic_f1),
    }


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def exact_match_rate(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> float:
    """Fraction of samples where predicted set == ground truth set exactly.

    A strict metric: a sample only counts as correct if every ground-truth
    line is detected AND no extra lines are predicted.

    Returns:
        Exact match rate as a float in [0, 1].
    """
    if not gt_sets:
        return 0.0
    matches = sum(1 for gt, pred in zip(gt_sets, pred_sets) if gt == pred)
    return matches / len(gt_sets)


# ---------------------------------------------------------------------------
# SNR-weighted metrics
# ---------------------------------------------------------------------------

def snr_weighted_metrics(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    """Recall and F1 weighted by signal-to-noise ratio (SNR).

    Standard recall treats every missed line equally. SNR-weighted recall
    penalizes missing bright (high-SNR) lines more than faint ones:

        recall_snr = SUM log(1 + SNR_i) for detected lines
                     ------------------------------------
                     SUM log(1 + SNR_i) for all GT lines

    Precision remains unweighted (|TP| / |predicted|), since predictions
    don't carry a "strength" value.

    F1_snr is the harmonic mean of unweighted precision and SNR-weighted recall.

    Args:
        gt_dicts: List of {line_name: snr} dicts (ground truth with SNR).
        pred_sets: List of sets of predicted line names.

    Returns:
        Dict with keys: mean_snr_weighted_recall, mean_snr_weighted_f1.
        Also includes mean_precision (unweighted, same as sample-level).
    """
    precisions = []
    snr_recalls = []
    snr_f1s = []

    for gt_dict, pred in zip(gt_dicts, pred_sets):
        gt = set(gt_dict.keys())
        tp = gt & pred
        n_pred = len(pred)
        n_gt = len(gt)

        # Unweighted precision (same as sample-level)
        if n_pred == 0 and n_gt == 0:
            p = 1.0
        elif n_pred == 0:
            p = 1.0  # true: no false positives
        elif n_gt == 0:
            p = 0.0
        else:
            p = len(tp) / n_pred

        # SNR-weighted recall
        gt_snr_total = sum(np.log1p(gt_dict.get(l, 0)) for l in gt)
        tp_snr_total = sum(np.log1p(gt_dict.get(l, 0)) for l in tp)

        if n_gt == 0 and n_pred == 0:
            rw = 1.0
        elif gt_snr_total > 0:
            rw = tp_snr_total / gt_snr_total
        else:
            rw = 0.0

        f1w = _safe_div(2 * p * rw, p + rw)

        precisions.append(p)
        snr_recalls.append(rw)
        snr_f1s.append(f1w)

    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_snr_weighted_recall": float(np.mean(snr_recalls)) if snr_recalls else 0.0,
        "mean_snr_weighted_f1": float(np.mean(snr_f1s)) if snr_f1s else 0.0,
    }


# ---------------------------------------------------------------------------
# Micro-level SNR-weighted metrics
# ---------------------------------------------------------------------------

def micro_snr_weighted_metrics(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    """Micro-aggregated SNR-weighted recall and F1.

    Pools SNR-weighted recall across all samples (total detected SNR /
    total ground-truth SNR), then combines with micro precision.

    Returns:
        Dict with keys: micro_snr_weighted_recall, micro_snr_weighted_f1.
    """
    total_tp = 0
    total_pred = 0
    total_gt_snr = 0.0
    total_tp_snr = 0.0

    for gt_dict, pred in zip(gt_dicts, pred_sets):
        gt = set(gt_dict.keys())
        tp = gt & pred
        total_tp += len(tp)
        total_pred += len(pred)
        total_gt_snr += sum(np.log1p(gt_dict.get(l, 0)) for l in gt)
        total_tp_snr += sum(np.log1p(gt_dict.get(l, 0)) for l in tp)

    mic_p = _safe_div(total_tp, total_pred)
    mic_rw = _safe_div(total_tp_snr, total_gt_snr)
    mic_f1w = _safe_div(2 * mic_p * mic_rw, mic_p + mic_rw)

    return {
        "micro_snr_weighted_recall": float(mic_rw),
        "micro_snr_weighted_f1": float(mic_f1w),
    }


# ---------------------------------------------------------------------------
# Per-label metrics
# ---------------------------------------------------------------------------

def per_label_report(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    labels: List[str],
    gt_dicts: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-label precision, recall, F1, TP/FP/FN counts, and optional SNR stats.

    Treats each label as an independent binary classification problem
    (present vs. absent in a sample), then reports metrics for each
    label individually.

    Args:
        gt_sets: List of ground-truth label sets.
        pred_sets: List of predicted label sets.
        labels: Canonical list of all possible labels.
        gt_dicts: Optional. If provided, also computes mean_snr and
            mean_detected_snr for each label.

    Returns:
        Dict mapping each label to {precision, recall, f1, support,
        tp, fp, fn, [mean_snr, mean_detected_snr]}.
    """
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([set(labels)])  # ensure all labels appear
    yt = mlb.transform(gt_sets)
    yp = mlb.transform(pred_sets)

    report = sklearn_classification_report(
        yt, yp, target_names=labels, output_dict=True, zero_division=0
    )

    result = {}
    for i, label in enumerate(labels):
        t_mask = yt[:, i] == 1
        p_mask = yp[:, i] == 1

        entry = {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label].get("f1-score", 0.0),
            "support": int(t_mask.sum()),
            "tp": int((t_mask & p_mask).sum()),
            "fp": int((~t_mask & p_mask).sum()),
            "fn": int((t_mask & ~p_mask).sum()),
        }

        if gt_dicts is not None:
            import pandas as pd
            gt_series = pd.Series(gt_dicts)
            snrs = gt_series.loc[t_mask].apply(lambda d: d.get(label, 0))
            det_snrs = gt_series.loc[t_mask & p_mask].apply(lambda d: d.get(label, 0))
            entry["mean_snr"] = float(snrs.mean()) if not snrs.empty else 0.0
            entry["mean_detected_snr"] = float(det_snrs.mean()) if not det_snrs.empty else 0.0

        result[label] = entry

    return result


def macro_label_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    labels: List[str],
) -> float:
    """Unweighted mean of per-label F1 scores.

    Measures whether the model is equally good at detecting all line types.

    Args:
        gt_sets: List of ground-truth label sets.
        pred_sets: List of predicted label sets.
        labels: Canonical list of all possible labels.

    Returns:
        Macro-label F1 as a float in [0, 1].
    """
    report = per_label_report(gt_sets, pred_sets, labels)
    f1s = [report[label]["f1"] for label in labels]
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Full report bundle
# ---------------------------------------------------------------------------

def multilabel_report(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
    labels: List[str],
    *,
    n_format_errors: int = 0,
) -> Dict[str, Any]:
    """Full multi-label report bundle for emission line detection.

    Combines all metric families into a single dict:
      - Sample-level: mean precision, recall, F1
      - Micro-level: pooled precision, recall, F1
      - SNR-weighted: sample-mean and micro SNR-weighted recall and F1
      - Macro-label: mean of per-label F1
      - Per-label: individual label metrics with SNR stats
      - Exact match rate
      - Format error count

    Args:
        gt_dicts: List of {line_name: snr} dicts (ground truth with SNR).
        pred_sets: List of sets of predicted line names.
        labels: Canonical list of all possible labels.
        n_format_errors: Number of samples where the parser completely
            failed (returned None). These should already be excluded
            from gt_dicts/pred_sets before calling this function.

    Returns:
        Nested dict with all computed metrics.
    """
    gt_sets = [set(d.keys()) for d in gt_dicts]

    total_samples = len(gt_dicts) + n_format_errors
    exact = exact_match_rate(gt_sets, pred_sets)
    sample = sample_precision_recall_f1(gt_sets, pred_sets)
    micro = micro_precision_recall_f1(gt_sets, pred_sets)
    snr = snr_weighted_metrics(gt_dicts, pred_sets)
    micro_snr = micro_snr_weighted_metrics(gt_dicts, pred_sets)
    plr = per_label_report(gt_sets, pred_sets, labels, gt_dicts=gt_dicts)
    macro_f1 = macro_label_f1(gt_sets, pred_sets, labels)

    n_exact = int(exact * len(gt_sets)) if gt_sets else 0

    return {
        "total_samples": total_samples,
        "evaluated_samples": len(gt_dicts),
        "exact_matches": n_exact,
        "exact_match_rate": exact,
        "format_errors": n_format_errors,
        "sample_level": sample,
        "dataset_micro_level": {
            **micro,
            **micro_snr,
        },
        "snr_weighted": {
            "mean_snr_weighted_recall": snr["mean_snr_weighted_recall"],
            "mean_snr_weighted_f1": snr["mean_snr_weighted_f1"],
        },
        "dataset_macro_level": {
            "mean_line_f1": macro_f1,
        },
        "per_line_metrics": plr,
    }

