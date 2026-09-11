from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..metrics.classification import (
    accuracy,
    macro_f1,
    weighted_f1,
    ordinal_mae,
    confusion_matrix_dict,
)
from ..metrics.multilabel import (
    sample_precision_recall_f1,
    micro_precision_recall_f1,
    snr_weighted_metrics,
    micro_snr_weighted_metrics,
    per_label_report,
    macro_label_f1,
    exact_match_rate,
)


def compute_caption_metrics(results_dir: str, task: Any) -> Dict[str, Any]:
    """Compute task metrics and caption diagnostics from predictions.jsonl and save metrics.json & report.md."""
    preds_file = os.path.join(results_dir, "predictions.jsonl")
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"predictions.jsonl not found in {results_dir}")

    records: List[Dict[str, Any]] = []
    with open(preds_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"Warning: predictions.jsonl in {results_dir} is empty.")
        return {}

    task_name = getattr(task, "name", "unknown")
    is_multilabel = "emission_lines" in task_name

    # 1. Caption Diagnostics
    caption_lengths_chars = [len(r.get("caption", {}).get("text", "")) for r in records]
    caption_word_counts = [len(r.get("caption", {}).get("text", "").split()) for r in records]
    fallbacks = [1 if r.get("frontier_evaluation", {}).get("forced_fallback", False) else 0 for r in records]

    parse_success = []
    for r in records:
        pred = r.get("frontier_evaluation", {}).get("prediction")
        if pred is None or pred == "UNKNOWN":
            parse_success.append(0)
        else:
            parse_success.append(1)

    diagnostics = {
        "avg_caption_char_len": float(np.mean(caption_lengths_chars)) if caption_lengths_chars else 0.0,
        "std_caption_char_len": float(np.std(caption_lengths_chars)) if caption_lengths_chars else 0.0,
        "avg_caption_word_count": float(np.mean(caption_word_counts)) if caption_word_counts else 0.0,
        "std_caption_word_count": float(np.std(caption_word_counts)) if caption_word_counts else 0.0,
        "frontier_parse_success_rate": float(np.mean(parse_success)) if parse_success else 0.0,
        "frontier_fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
    }

    # 2. Task Performance Metrics
    task_metrics: Dict[str, Any] = {}

    if is_multilabel:
        # Multilabel emission line detection
        gt_sets = []
        pred_sets = []
        gt_dicts = []

        for r in records:
            gt_val = r.get("ground_truth", {})
            if isinstance(gt_val, dict):
                gt_sets.append(set(gt_val.keys()))
                gt_dicts.append(gt_val)
            elif isinstance(gt_val, (list, set)):
                gt_sets.append(set(gt_val))
                gt_dicts.append({k: 1.0 for k in gt_val})
            else:
                gt_sets.append(set())
                gt_dicts.append({})

            p_val = r.get("frontier_evaluation", {}).get("prediction", [])
            pred_sets.append(set(p_val) if isinstance(p_val, (list, set)) else set())

        prf = sample_precision_recall_f1(gt_sets, pred_sets)
        micro_prf = micro_precision_recall_f1(gt_sets, pred_sets)
        snr_m = snr_weighted_metrics(gt_dicts, pred_sets)
        micro_snr = micro_snr_weighted_metrics(gt_dicts, pred_sets)
        exact = exact_match_rate(gt_sets, pred_sets)

        canonical_lines = getattr(task, "canonical_lines", [])
        if not canonical_lines:
            all_found = set().union(*gt_sets, *pred_sets)
            canonical_lines = sorted(list(all_found)) if all_found else []
        line_m = per_label_report(gt_sets, pred_sets, labels=canonical_lines, gt_dicts=gt_dicts)
        macro_f1_val = macro_label_f1(gt_sets, pred_sets, labels=canonical_lines)

        task_metrics = {
            # Sample-level metrics (mean across samples)
            "sample_precision": prf.get("mean_precision", 0.0),
            "sample_recall": prf.get("mean_recall", 0.0),
            "sample_f1": prf.get("mean_f1", 0.0),
            "snr_weighted_recall": snr_m.get("mean_snr_weighted_recall", 0.0),
            "snr_weighted_f1": snr_m.get("mean_snr_weighted_f1", 0.0),
            # Dataset-level (micro / pooled across dataset)
            "dataset_precision": micro_prf.get("micro_precision", 0.0),
            "dataset_recall": micro_prf.get("micro_recall", 0.0),
            "dataset_f1": micro_prf.get("micro_f1", 0.0),
            "dataset_micro_snr_weighted_recall": micro_snr.get("micro_snr_weighted_recall", 0.0),
            "dataset_micro_snr_weighted_f1": micro_snr.get("micro_snr_weighted_f1", 0.0),
            # Macro-line and exact match
            "dataset_macro_f1": macro_f1_val,
            "exact_match_rate": exact,
            # Structured groupings matching classic eval suite
            "sample_level": {
                **prf,
                "mean_snr_weighted_recall": snr_m.get("mean_snr_weighted_recall", 0.0),
                "mean_snr_weighted_f1": snr_m.get("mean_snr_weighted_f1", 0.0),
            },
            "dataset_micro_level": {
                **micro_prf,
                **micro_snr,
            },
            "dataset_macro_level": {
                "mean_line_f1": macro_f1_val,
            },
            "per_line": line_m,
        }
    else:
        # Single-label classification
        y_true = [str(r.get("ground_truth", "UNKNOWN")) for r in records]
        y_pred = [str(r.get("frontier_evaluation", {}).get("prediction", "UNKNOWN")) for r in records]

        acc = accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        wf1 = weighted_f1(y_true, y_pred)

        # Check if ordinal (distance classification labels like A, B, C)
        ord_mae = None
        if hasattr(task, "scheme") and hasattr(task.scheme, "labels"):
            labels = list(task.scheme.labels)
            cm = confusion_matrix_dict(y_true, y_pred, labels=labels)
            try:
                ord_mae = ordinal_mae(y_true, y_pred, labels=labels)
            except Exception:
                ord_mae = None
        elif hasattr(task, "categories"):
            labels = list(task.categories)
            cm = confusion_matrix_dict(y_true, y_pred, labels=labels)
        else:
            labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix_dict(y_true, y_pred, labels=labels)

        task_metrics = {
            "accuracy": acc,
            "macro_f1": mf1,
            "weighted_f1": wf1,
            "confusion_matrix": cm,
        }
        if ord_mae is not None:
            task_metrics["ordinal_mae"] = ord_mae

    # Combine into full summary
    metrics_summary = {
        "task_name": task_name,
        "total_samples": len(records),
        "task_metrics": task_metrics,
        "caption_diagnostics": diagnostics,
    }

    # Save metrics.json
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Generate report.md
    _generate_markdown_report(results_dir, records, metrics_summary, is_multilabel)

    return metrics_summary


def _generate_markdown_report(
    results_dir: str,
    records: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    is_multilabel: bool,
):
    """Generate human-readable markdown evaluation report with confusion matrix and examples."""
    task_name = metrics.get("task_name", "Task")
    tm = metrics.get("task_metrics", {})
    diag = metrics.get("caption_diagnostics", {})

    lines = [
        f"# Evaluation Report: {task_name}",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Samples** | {metrics.get('total_samples')} |",
    ]

    if is_multilabel:
        lines.extend([
            f"| **Sample-Mean Precision** | {tm.get('sample_precision', 0):.4f} |",
            f"| **Sample-Mean Recall** | {tm.get('sample_recall', 0):.4f} |",
            f"| **Sample-Mean F1** | {tm.get('sample_f1', 0):.4f} |",
            f"| **Dataset-Level (Micro) Precision** | {tm.get('dataset_precision', 0):.4f} |",
            f"| **Dataset-Level (Micro) Recall** | {tm.get('dataset_recall', 0):.4f} |",
            f"| **Dataset-Level (Micro) F1** | {tm.get('dataset_f1', 0):.4f} |",
            f"| **Dataset-Level Macro F1** | {tm.get('dataset_macro_f1', 0):.4f} |",
            f"| **Exact Match Rate** | {tm.get('exact_match_rate', 0) * 100:.2f}% |",
            f"| **SNR-Weighted Recall (Sample)** | {tm.get('snr_weighted_recall', 0):.4f} |",
            f"| **SNR-Weighted F1 (Sample)** | {tm.get('snr_weighted_f1', 0):.4f} |",
            f"| **SNR-Weighted F1 (Dataset Micro)** | {tm.get('dataset_micro_snr_weighted_f1', 0):.4f} |",
        ])
    else:
        lines.extend([
            f"| **Accuracy** | {tm.get('accuracy', 0):.4f} |",
            f"| **Macro F1** | {tm.get('macro_f1', 0):.4f} |",
            f"| **Weighted F1** | {tm.get('weighted_f1', 0):.4f} |",
        ])
        if "ordinal_mae" in tm:
            lines.append(f"| **Ordinal MAE** | {tm.get('ordinal_mae', 0):.4f} |")

    lines.extend([
        "",
        "## 2. Caption Quality Diagnostics",
        "",
        "| Diagnostic Indicator | Value |",
        "| :--- | :--- |",
        f"| **Avg Caption Length (Chars)** | {diag.get('avg_caption_char_len', 0):.1f} ± {diag.get('std_caption_char_len', 0):.1f} |",
        f"| **Avg Caption Word Count** | {diag.get('avg_caption_word_count', 0):.1f} ± {diag.get('std_caption_word_count', 0):.1f} |",
        f"| **Frontier Parse Success Rate** | {diag.get('frontier_parse_success_rate', 0) * 100:.1f}% |",
        f"| **Frontier Fallback Retry Rate** | {diag.get('frontier_fallback_rate', 0) * 100:.1f}% |",
    ])

    # Confusion matrix or per-line table
    if not is_multilabel and "confusion_matrix" in tm:
        cm = tm["confusion_matrix"]
        if isinstance(cm, dict) and cm:
            labels = list(cm.keys())
            lines.extend([
                "",
                "## 3. Confusion Matrix",
                "",
                "| True \\ Pred | " + " | ".join(labels) + " |",
                "| :--- | " + " | ".join([":---:"] * len(labels)) + " |",
            ])
            for true_lbl in labels:
                row_str = f"| **{true_lbl}** | " + " | ".join(str(cm[true_lbl].get(p, 0)) for p in labels) + " |"
                lines.append(row_str)

    elif is_multilabel and "per_line" in tm:
        lines.extend([
            "",
            "## 3. Per-Line Detection Statistics",
            "",
            "| Emission Line | Precision | Recall | F1 | Support |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for line_name, stats in tm["per_line"].items():
            lines.append(
                f"| {line_name} | {stats.get('precision', 0):.3f} | {stats.get('recall', 0):.3f} | {stats.get('f1', 0):.3f} | {stats.get('support', 0)} |"
            )

    # 4. Spotlights / Case Studies
    successes = [r for r in records if r.get("frontier_evaluation", {}).get("is_correct", False)]
    failures = [r for r in records if not r.get("frontier_evaluation", {}).get("is_correct", False)]

    lines.extend([
        "",
        "## 4. Qualitative Spotlights",
        "",
        "### Top Success Examples",
    ])
    for i, s in enumerate(successes[:2]):
        cap = s.get("caption", {}).get("text", "").strip()
        raw = s.get("frontier_evaluation", {}).get("raw_response", "").strip()
        lines.extend([
            f"#### Success Case #{i+1} (ID: `{s.get('sample_id')}`)",
            f"- **Ground Truth**: `{s.get('ground_truth')}` | **Predicted**: `{s.get('frontier_evaluation', {}).get('prediction')}`",
            f"- **Generated Caption**:\n> {cap}",
            f"- **Frontier Reasoning**:\n```\n{raw}\n```",
            "",
        ])

    lines.append("### Top Failure Examples")
    for i, f_item in enumerate(failures[:2]):
        cap = f_item.get("caption", {}).get("text", "").strip()
        raw = f_item.get("frontier_evaluation", {}).get("raw_response", "").strip()
        lines.extend([
            f"#### Failure Case #{i+1} (ID: `{f_item.get('sample_id')}`)",
            f"- **Ground Truth**: `{f_item.get('ground_truth')}` | **Predicted**: `{f_item.get('frontier_evaluation', {}).get('prediction')}`",
            f"- **Generated Caption**:\n> {cap}",
            f"- **Frontier Reasoning**:\n```\n{raw}\n```",
            "",
        ])

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
