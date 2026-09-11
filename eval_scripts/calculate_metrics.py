#!/usr/bin/env python3
"""Calculate updated evaluation metrics from any evaluation JSONL file or directory.

Supports:
- Caption evaluations (predictions.jsonl)
- Classic evaluations (results.jsonl)
- Multilabel emission line detection:
    * Sample-level metrics (Sample Precision, Recall, F1)
    * Dataset-level micro metrics (Micro Precision, Recall, F1 pooled over dataset)
    * Dataset-level macro F1 (unweighted mean of per-line F1)
    * Exact match rate
    * SNR-weighted metrics (sample mean & dataset micro)
    * Per-line detection statistics table
- Single-label classification (distance, source, subclass):
    * Accuracy, Macro F1, Weighted F1
    * Ordinal MAE (for ordinal distance schemes)
    * Confusion Matrix
- Caption quality diagnostics (char length, word count, parse success rate, retry rate)

Usage:
    uv run eval_scripts/calculate_metrics.py path/to/predictions.jsonl
    uv run eval_scripts/calculate_metrics.py path/to/results_dir/
    uv run eval_scripts/calculate_metrics.py eval_results/caption_eval/*/predictions.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure src/ is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from evals.metrics.classification import (
        accuracy,
        macro_f1,
        weighted_f1,
        ordinal_mae,
        confusion_matrix_dict,
    )
    from evals.metrics.multilabel import (
        sample_precision_recall_f1,
        micro_precision_recall_f1,
        snr_weighted_metrics,
        micro_snr_weighted_metrics,
        per_label_report,
        macro_label_f1,
        exact_match_rate,
    )
except ImportError:
    # Fallback implementations in case package isn't installed in editable mode
    print("Notice: evals.metrics imports loaded via direct fallback.")
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    def accuracy(y_true, y_pred):
        return float(accuracy_score(y_true, y_pred)) if y_true else 0.0

    def macro_f1(y_true, y_pred):
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0

    def weighted_f1(y_true, y_pred):
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0

    def ordinal_mae(y_true, y_pred, labels):
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        diffs = [abs(l2i[t] - l2i[p]) for t, p in zip(y_true, y_pred) if t in l2i and p in l2i]
        return float(np.mean(diffs)) if diffs else 0.0

    def confusion_matrix_dict(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {lbl: {labels[j]: int(cm[i][j]) for j in range(len(labels))} for i, lbl in enumerate(labels)}


# Standard default emission lines
CANONICAL_EMISSION_LINES = [
    "[O II] 3727", "[Ne III] 3869", "Hδ", "Hγ", "[O III] 4363",
    "Hβ", "[O III] 5007", "He I 5876", "[O I] 6300", "[N II] 6583",
    "Hα", "[S II] 6720", "[Ar III] 7135", "[O II] 7325", "[S III] 9069"
]


def resolve_jsonl_paths(inputs: List[str]) -> List[Path]:
    """Resolve input strings (files, directories, glob patterns) to existing JSONL files."""
    resolved: List[Path] = []
    for raw in inputs:
        matches = glob.glob(raw)
        if not matches:
            matches = [raw]
        for m in matches:
            p = Path(m)
            if p.is_file():
                if p.suffix in [".jsonl", ".json"]:
                    resolved.append(p)
            elif p.is_dir():
                # Check for standard file names in priority order
                cand_pred = p / "predictions.jsonl"
                cand_res = p / "results.jsonl"
                if cand_pred.exists():
                    resolved.append(cand_pred)
                elif cand_res.exists():
                    resolved.append(cand_res)
                else:
                    # Find any jsonl in the directory
                    all_jsonls = list(p.glob("*.jsonl"))
                    if all_jsonls:
                        resolved.append(all_jsonls[0])
                    else:
                        print(f"Warning: No .jsonl found in directory {p}")
            else:
                print(f"Warning: Path not found: {raw}")
    return sorted(list(set(resolved)))


def extract_record_data(
    r: Dict[str, Any]
) -> Tuple[Any, Any, Optional[str], Optional[Dict[str, float]]]:
    """Extract (ground_truth, prediction, format_error, snr_dict) from a record."""
    gt: Any = None
    pred: Any = None
    gt_snr: Optional[Dict[str, float]] = None

    # 1. Ground truth
    if "ground_truth" in r:
        gt = r["ground_truth"]
    elif "ground_truth_lines" in r:
        gt = r["ground_truth_lines"]
    elif "correct_answer" in r:
        gt = r["correct_answer"]
    elif "target" in r:
        gt = r["target"]

    if isinstance(gt, dict):
        gt_snr = {k: float(v) for k, v in gt.items() if isinstance(v, (int, float))}
    elif isinstance(gt, (list, set)):
        gt_snr = {str(k): 1.0 for k in gt}

    # 2. Prediction
    if "frontier_evaluation" in r and isinstance(r["frontier_evaluation"], dict):
        pred = r["frontier_evaluation"].get("prediction")
    elif "predicted_lines" in r:
        pred = r["predicted_lines"]
    elif "model_answer" in r:
        pred = r["model_answer"]
    elif "prediction" in r:
        pred = r["prediction"]
    elif "response" in r and isinstance(r["response"], dict):
        pred = r["response"].get("parsed", r["response"].get("prediction"))

    return gt, pred, gt_snr


def infer_is_multilabel(records: List[Dict[str, Any]], user_task: Optional[str]) -> bool:
    """Infer whether the task is multilabel emission line detection or single-label."""
    if user_task:
        return "emission" in user_task.lower() or "line" in user_task.lower()

    # Check task attribute in records
    for r in records[:5]:
        t = str(r.get("task", "")).lower()
        if "emission" in t or "line" in t:
            return True
        if "distance" in t or "source" in t or "subclass" in t:
            return False

    # Check data types of ground truth
    for r in records[:10]:
        gt, _, _ = extract_record_data(r)
        if isinstance(gt, (dict, list, set)):
            return True
        if "ground_truth_lines" in r or "predicted_lines" in r:
            return True

    return False


def compute_metrics_for_records(
    records: List[Dict[str, Any]],
    is_multilabel: bool,
    task_name: str,
) -> Dict[str, Any]:
    """Compute all evaluation metrics and diagnostics for a set of records."""
    total_samples = len(records)
    if total_samples == 0:
        return {"total_samples": 0, "task_metrics": {}, "caption_diagnostics": {}}

    # 1. Caption diagnostics
    caption_lengths_chars = []
    caption_word_counts = []
    fallbacks = []
    parse_success = []

    has_captions = any("caption" in r for r in records)

    for r in records:
        if "caption" in r and isinstance(r["caption"], dict):
            text = str(r["caption"].get("text", ""))
            caption_lengths_chars.append(len(text))
            caption_word_counts.append(len(text.split()))

        # Check fallback
        if "frontier_evaluation" in r and isinstance(r["frontier_evaluation"], dict):
            fallbacks.append(1 if r["frontier_evaluation"].get("forced_fallback", False) else 0)
            p = r["frontier_evaluation"].get("prediction")
            parse_success.append(0 if p is None or p == "UNKNOWN" else 1)
        elif "forced_fallback" in r:
            fallbacks.append(1 if r.get("forced_fallback", False) else 0)

    diagnostics: Dict[str, Any] = {}
    if has_captions and caption_lengths_chars:
        diagnostics = {
            "avg_caption_char_len": float(np.mean(caption_lengths_chars)),
            "std_caption_char_len": float(np.std(caption_lengths_chars)),
            "avg_caption_word_count": float(np.mean(caption_word_counts)),
            "std_caption_word_count": float(np.std(caption_word_counts)),
            "frontier_parse_success_rate": float(np.mean(parse_success)) if parse_success else 1.0,
            "frontier_fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
        }

    task_metrics: Dict[str, Any] = {}

    if is_multilabel:
        gt_sets: List[Set[str]] = []
        pred_sets: List[Set[str]] = []
        gt_dicts: List[Dict[str, float]] = []
        n_format_errors = 0

        for r in records:
            gt, pred, gt_snr = extract_record_data(r)

            if isinstance(gt, dict):
                gt_sets.append(set(gt.keys()))
                gt_dicts.append({k: float(v) for k, v in gt.items() if isinstance(v, (int, float))})
            elif isinstance(gt, (list, set)):
                gt_sets.append(set(gt))
                gt_dicts.append({k: 1.0 for k in gt})
            else:
                gt_sets.append(set())
                gt_dicts.append({})

            if isinstance(pred, (list, set)):
                pred_sets.append(set(pred))
            elif pred is None or pred == "UNKNOWN":
                pred_sets.append(set())
                n_format_errors += 1
            else:
                pred_sets.append(set())

        prf = sample_precision_recall_f1(gt_sets, pred_sets)
        micro_prf = micro_precision_recall_f1(gt_sets, pred_sets)
        snr_m = snr_weighted_metrics(gt_dicts, pred_sets)
        micro_snr = micro_snr_weighted_metrics(gt_dicts, pred_sets)
        exact = exact_match_rate(gt_sets, pred_sets)

        # Candidate lines for per-label reporting
        all_gt_lines = set().union(*gt_sets)
        all_pred_lines = set().union(*pred_sets)
        combined_lines = all_gt_lines.union(all_pred_lines)

        # Standard canonical lines order, supplemented by any detected
        candidate_lines = [line for line in CANONICAL_EMISSION_LINES if line in combined_lines]
        extra_lines = sorted(list(combined_lines - set(candidate_lines)))
        canonical_lines = candidate_lines + extra_lines
        if not canonical_lines:
            canonical_lines = ["Hγ", "Hβ", "Hα"]

        line_m = per_label_report(gt_sets, pred_sets, labels=canonical_lines, gt_dicts=gt_dicts)
        macro_f1_val = macro_label_f1(gt_sets, pred_sets, labels=canonical_lines)

        task_metrics = {
            # Headline metrics
            "sample_precision": prf.get("mean_precision", 0.0),
            "sample_recall": prf.get("mean_recall", 0.0),
            "sample_f1": prf.get("mean_f1", 0.0),
            "snr_weighted_recall": snr_m.get("mean_snr_weighted_recall", 0.0),
            "snr_weighted_f1": snr_m.get("mean_snr_weighted_f1", 0.0),
            # Dataset-level micro metrics
            "dataset_precision": micro_prf.get("micro_precision", 0.0),
            "dataset_recall": micro_prf.get("micro_recall", 0.0),
            "dataset_f1": micro_prf.get("micro_f1", 0.0),
            "dataset_micro_snr_weighted_recall": micro_snr.get("micro_snr_weighted_recall", 0.0),
            "dataset_micro_snr_weighted_f1": micro_snr.get("micro_snr_weighted_f1", 0.0),
            # Macro & Exact Match
            "dataset_macro_f1": macro_f1_val,
            "exact_match_rate": exact,
            "format_errors": n_format_errors,
            # Structured dicts
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
        y_true = []
        y_pred = []
        for r in records:
            gt, pred, _ = extract_record_data(r)
            y_true.append(str(gt) if gt is not None else "UNKNOWN")
            y_pred.append(str(pred) if pred is not None else "UNKNOWN")

        acc = accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        wf1 = weighted_f1(y_true, y_pred)

        all_labels = sorted(list(set(y_true + y_pred) - {"UNKNOWN"}))
        cm = confusion_matrix_dict(y_true, y_pred, labels=all_labels)

        # Detect ordinal labels (like 'A', 'B', 'C', 'D', 'E')
        ord_mae = None
        if all_labels and all(len(lbl) == 1 and lbl.isupper() for lbl in all_labels):
            try:
                ord_mae = ordinal_mae(y_true, y_pred, labels=sorted(all_labels))
            except Exception:
                ord_mae = None

        task_metrics = {
            "accuracy": acc,
            "macro_f1": mf1,
            "weighted_f1": wf1,
            "confusion_matrix": cm,
        }
        if ord_mae is not None:
            task_metrics["ordinal_mae"] = ord_mae

    return {
        "task_name": task_name,
        "total_samples": total_samples,
        "evaluated_samples": total_samples,
        "task_metrics": task_metrics,
        "caption_diagnostics": diagnostics,
    }


def format_markdown_table(headers: List[str], rows: List[List[str]], alignments: Optional[List[str]] = None) -> str:
    """Format markdown table string."""
    if not alignments:
        alignments = [":---"] + [":---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_metrics_summary(filepath: Path, summary: Dict[str, Any], is_multilabel: bool):
    """Print clean formatted tables directly to stdout."""
    task_name = summary.get("task_name", "Evaluation")
    total = summary.get("total_samples", 0)
    tm = summary.get("task_metrics", {})
    diag = summary.get("caption_diagnostics", {})

    print("\n" + "=" * 80)
    print(f"📊 EVALUATION REPORT: {task_name.upper()}")
    print(f"📁 Source: {filepath}")
    print("=" * 80)
    print(f"Total Samples: {total}")

    if is_multilabel:
        exact = tm.get("exact_match_rate", 0.0) * 100
        fmt_err = tm.get("format_errors", 0)
        print(f"Exact Match Rate: {exact:.2f}% | Parse Errors: {fmt_err}\n")

        print("--- 1. Headline Metrics ---")
        headers = ["Metric Category", "Precision", "Recall", "F1-Score", "SNR-Weighted F1"]
        rows = [
            [
                "**Sample-Mean**",
                f"{tm.get('sample_precision', 0.0):.4f}",
                f"{tm.get('sample_recall', 0.0):.4f}",
                f"{tm.get('sample_f1', 0.0):.4f}",
                f"{tm.get('snr_weighted_f1', 0.0):.4f}",
            ],
            [
                "**Dataset-Level (Micro)**",
                f"{tm.get('dataset_precision', 0.0):.4f}",
                f"{tm.get('dataset_recall', 0.0):.4f}",
                f"{tm.get('dataset_f1', 0.0):.4f}",
                f"{tm.get('dataset_micro_snr_weighted_f1', 0.0):.4f}",
            ],
            [
                "**Dataset Macro F1**",
                "-",
                "-",
                f"{tm.get('dataset_macro_f1', 0.0):.4f}",
                "-",
            ],
        ]
        aligns = [":---", ":---:", ":---:", ":---:", ":---:"]
        print(format_markdown_table(headers, rows, aligns))

        # Per-line table
        per_line = tm.get("per_line", {})
        if per_line:
            print("\n--- 2. Per-Line Detection Statistics ---")
            p_headers = ["Line", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean SNR"]
            p_rows = []
            for line_name, s in per_line.items():
                mean_snr_str = f"{s.get('mean_snr', 0.0):.1f}" if s.get("mean_snr") is not None else "-"
                p_rows.append([
                    f"**{line_name}**",
                    str(s.get("support", 0)),
                    str(s.get("tp", 0)),
                    str(s.get("fp", 0)),
                    str(s.get("fn", 0)),
                    f"{s.get('precision', 0.0):.4f}",
                    f"{s.get('recall', 0.0):.4f}",
                    f"{s.get('f1', 0.0):.4f}",
                    mean_snr_str,
                ])
            p_aligns = [":---"] + [":---:"] * (len(p_headers) - 1)
            print(format_markdown_table(p_headers, p_rows, p_aligns))

    else:
        print(f"Accuracy:    {tm.get('accuracy', 0.0):.4f}")
        print(f"Macro F1:    {tm.get('macro_f1', 0.0):.4f}")
        print(f"Weighted F1: {tm.get('weighted_f1', 0.0):.4f}")
        if "ordinal_mae" in tm:
            print(f"Ordinal MAE: {tm.get('ordinal_mae', 0.0):.4f}")

        cm = tm.get("confusion_matrix")
        if isinstance(cm, dict) and cm:
            print("\n--- Confusion Matrix (True \\ Pred) ---")
            labels = list(cm.keys())
            cm_headers = ["True \\ Pred"] + labels
            cm_rows = []
            for row_lbl in labels:
                row_vals = [f"**{row_lbl}**"] + [str(cm[row_lbl].get(col_lbl, 0)) for col_lbl in labels]
                cm_rows.append(row_vals)
            print(format_markdown_table(cm_headers, cm_rows))

    if diag:
        print("\n--- Caption Diagnostics ---")
        diag_headers = ["Diagnostic Indicator", "Value"]
        diag_rows = [
            ["Avg Caption Length (Chars)", f"{diag.get('avg_caption_char_len', 0.0):.1f} ± {diag.get('std_caption_char_len', 0.0):.1f}"],
            ["Avg Caption Word Count", f"{diag.get('avg_caption_word_count', 0.0):.1f} ± {diag.get('std_caption_word_count', 0.0):.1f}"],
            ["Frontier Parse Success Rate", f"{diag.get('frontier_parse_success_rate', 0.0) * 100:.1f}%"],
            ["Frontier Fallback Retry Rate", f"{diag.get('frontier_fallback_rate', 0.0) * 100:.1f}%"],
        ]
        print(format_markdown_table(diag_headers, diag_rows))

    print("=" * 80 + "\n")


def generate_report_md(
    summary: Dict[str, Any],
    is_multilabel: bool,
    filepath: Path,
    records: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate Markdown report string."""
    task_name = summary.get("task_name", "Evaluation")
    tm = summary.get("task_metrics", {})
    diag = summary.get("caption_diagnostics", {})

    lines = [
        f"# Evaluation Report: {task_name}",
        "",
        f"**Source file:** `{filepath.name}`  ",
        f"**Total Samples:** {summary.get('total_samples')}  ",
        "",
        "## 1. Executive Summary",
        "",
    ]

    if is_multilabel:
        headers = ["Metric", "Value"]
        rows = [
            ["**Sample-Mean Precision**", f"{tm.get('sample_precision', 0.0):.4f}"],
            ["**Sample-Mean Recall**", f"{tm.get('sample_recall', 0.0):.4f}"],
            ["**Sample-Mean F1**", f"{tm.get('sample_f1', 0.0):.4f}"],
            ["**Dataset-Level (Micro) Precision**", f"{tm.get('dataset_precision', 0.0):.4f}"],
            ["**Dataset-Level (Micro) Recall**", f"{tm.get('dataset_recall', 0.0):.4f}"],
            ["**Dataset-Level (Micro) F1**", f"{tm.get('dataset_f1', 0.0):.4f}"],
            ["**Dataset-Level Macro F1**", f"{tm.get('dataset_macro_f1', 0.0):.4f}"],
            ["**Exact Match Rate**", f"{tm.get('exact_match_rate', 0.0) * 100:.2f}%"],
            ["**SNR-Weighted Recall (Sample)**", f"{tm.get('snr_weighted_recall', 0.0):.4f}"],
            ["**SNR-Weighted F1 (Sample)**", f"{tm.get('snr_weighted_f1', 0.0):.4f}"],
            ["**SNR-Weighted F1 (Dataset Micro)**", f"{tm.get('dataset_micro_snr_weighted_f1', 0.0):.4f}"],
        ]
        lines.append(format_markdown_table(headers, rows))

        per_line = tm.get("per_line", {})
        if per_line:
            lines.extend(["", "## 2. Per-Line Detection Statistics", ""])
            p_headers = ["Line", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean SNR"]
            p_rows = []
            for line_name, s in per_line.items():
                mean_snr_str = f"{s.get('mean_snr', 0.0):.1f}" if s.get("mean_snr") is not None else "-"
                p_rows.append([
                    f"**{line_name}**",
                    str(s.get("support", 0)),
                    str(s.get("tp", 0)),
                    str(s.get("fp", 0)),
                    str(s.get("fn", 0)),
                    f"{s.get('precision', 0.0):.4f}",
                    f"{s.get('recall', 0.0):.4f}",
                    f"{s.get('f1', 0.0):.4f}",
                    mean_snr_str,
                ])
            lines.append(format_markdown_table(p_headers, p_rows))

    else:
        headers = ["Metric", "Value"]
        rows = [
            ["**Accuracy**", f"{tm.get('accuracy', 0.0):.4f}"],
            ["**Macro F1**", f"{tm.get('macro_f1', 0.0):.4f}"],
            ["**Weighted F1**", f"{tm.get('weighted_f1', 0.0):.4f}"],
        ]
        if "ordinal_mae" in tm:
            rows.append(["**Ordinal MAE**", f"{tm.get('ordinal_mae', 0.0):.4f}"])
        lines.append(format_markdown_table(headers, rows))

        cm = tm.get("confusion_matrix")
        if isinstance(cm, dict) and cm:
            lines.extend(["", "## 2. Confusion Matrix", ""])
            labels = list(cm.keys())
            cm_headers = ["True \\ Pred"] + labels
            cm_rows = []
            for row_lbl in labels:
                cm_rows.append([f"**{row_lbl}**"] + [str(cm[row_lbl].get(col, 0)) for col in labels])
            lines.append(format_markdown_table(cm_headers, cm_rows))

    if diag:
        lines.extend([
            "",
            "## 3. Caption Quality Diagnostics",
            "",
            format_markdown_table(
                ["Diagnostic Indicator", "Value"],
                [
                    ["**Avg Caption Length (Chars)**", f"{diag.get('avg_caption_char_len', 0.0):.1f} ± {diag.get('std_caption_char_len', 0.0):.1f}"],
                    ["**Avg Caption Word Count**", f"{diag.get('avg_caption_word_count', 0.0):.1f} ± {diag.get('std_caption_word_count', 0.0):.1f}"],
                    ["**Frontier Parse Success Rate**", f"{diag.get('frontier_parse_success_rate', 0.0) * 100:.1f}%"],
                    ["**Frontier Fallback Retry Rate**", f"{diag.get('frontier_fallback_rate', 0.0) * 100:.1f}%"],
                ]
            ),
        ])

    # 4. Spotlights / Qualitative Examples
    if records:
        successes = [r for r in records if r.get("frontier_evaluation", {}).get("is_correct", False)]
        failures = [r for r in records if not r.get("frontier_evaluation", {}).get("is_correct", False)]

        if successes or failures:
            lines.extend(["", "## 4. Qualitative Spotlights", ""])

            if successes:
                lines.append("### Top Success Examples")
                for i, s in enumerate(successes[:2]):
                    cap = s.get("caption", {}).get("text", "").strip()
                    raw = s.get("frontier_evaluation", {}).get("raw_response", "").strip()
                    sample_id = s.get("sample_id") or s.get("wiki_entity_id") or f"Sample {i+1}"
                    gt_val = s.get("ground_truth", s.get("ground_truth_lines", s.get("correct_answer")))
                    pred_val = s.get("frontier_evaluation", {}).get("prediction", s.get("predicted_lines", s.get("model_answer")))
                    lines.extend([
                        f"#### Success Case #{i+1} (ID: `{sample_id}`)",
                        f"- **Ground Truth**: `{gt_val}` | **Predicted**: `{pred_val}`",
                        f"- **Generated Caption**:\n> {cap}" if cap else "",
                        f"- **Frontier Reasoning**:\n```\n{raw}\n```" if raw else "",
                        "",
                    ])

            if failures:
                lines.append("### Top Failure Examples")
                for i, f_item in enumerate(failures[:2]):
                    cap = f_item.get("caption", {}).get("text", "").strip()
                    raw = f_item.get("frontier_evaluation", {}).get("raw_response", "").strip()
                    sample_id = f_item.get("sample_id") or f_item.get("wiki_entity_id") or f"Sample {i+1}"
                    gt_val = f_item.get("ground_truth", f_item.get("ground_truth_lines", f_item.get("correct_answer")))
                    pred_val = f_item.get("frontier_evaluation", {}).get("prediction", f_item.get("predicted_lines", f_item.get("model_answer")))
                    lines.extend([
                        f"#### Failure Case #{i+1} (ID: `{sample_id}`)",
                        f"- **Ground Truth**: `{gt_val}` | **Predicted**: `{pred_val}`",
                        f"- **Generated Caption**:\n> {cap}" if cap else "",
                        f"- **Frontier Reasoning**:\n```\n{raw}\n```" if raw else "",
                        "",
                    ])

    return "\n".join(lines) + "\n"


def process_file(
    filepath: Path,
    user_task: Optional[str],
    output_dir: Optional[str],
    no_save: bool,
):
    """Process a single JSONL file and compute metrics."""
    records: List[Dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print(f"Error: {filepath} contains no valid JSON lines.")
        return

    is_multilabel = infer_is_multilabel(records, user_task)

    # Determine task name
    if user_task:
        task_name = user_task
    else:
        # Infer from first record
        first_task = records[0].get("task")
        if first_task:
            task_name = str(first_task)
        elif is_multilabel:
            task_name = "emission_lines"
        else:
            task_name = "classification"

    summary = compute_metrics_for_records(records, is_multilabel, task_name)

    # Print summary to stdout
    print_metrics_summary(filepath, summary, is_multilabel)

    # Save to disk if requested
    if not no_save:
        save_dir = Path(output_dir) if output_dir else filepath.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = save_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"💾 Saved JSON metrics to: {metrics_file}")

        report_file = save_dir / "report.md"
        report_content = generate_report_md(summary, is_multilabel, filepath, records=records)
        with open(report_file, "w") as f:
            f.write(report_content)
        print(f"📝 Saved Markdown report to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute updated evaluation metrics from JSONL files (caption or classic evals)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSONL files or directories (e.g. predictions.jsonl, results.jsonl, or eval directory)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Explicitly specify task type ('emission_lines', 'distance', 'source', 'subclass')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Custom output directory to save metrics.json and report.md (defaults to input file's directory)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print metrics to stdout only without writing metrics.json or report.md",
    )

    args = parser.parse_args()

    resolved_files = resolve_jsonl_paths(args.inputs)
    if not resolved_files:
        print("Error: No valid JSONL files found matching inputs.")
        sys.exit(1)

    for p in resolved_files:
        process_file(p, args.task, args.output_dir, args.no_save)


if __name__ == "__main__":
    main()
