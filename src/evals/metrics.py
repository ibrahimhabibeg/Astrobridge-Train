import os
import json
import pandas as pd
import numpy as np
from typing import Any
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from .buckets import BucketScheme

def compute_and_save_metrics(results_dir: str, task_or_scheme: Any):
    if hasattr(task_or_scheme, "name") and task_or_scheme.name == "emission_lines":
        _compute_emission_line_metrics(results_dir, task_or_scheme)
    else:
        scheme = getattr(task_or_scheme, "scheme", task_or_scheme)
        _compute_classification_metrics(results_dir, scheme)

def _load_metadata(results_dir: str) -> dict:
    path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _compute_classification_metrics(results_dir: str, scheme: BucketScheme):
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path): return
        
    df = pd.read_json(results_path, lines=True)
    if df.empty: return
        
    df["pred"] = df.get("model_answer", df.get("parsed", "UNKNOWN")).fillna("UNKNOWN")
    
    valid_mask = df["pred"].isin(scheme.labels)
    format_errors = len(df) - valid_mask.sum()
    df_valid = df[valid_mask]
    
    if df_valid.empty: return
        
    y_true, y_pred = df_valid["correct_answer"], df_valid["pred"]
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=scheme.labels, output_dict=True, zero_division=0)
    
    label_map = {label: i for i, label in enumerate(scheme.labels)}
    mae = mean_absolute_error(y_true.map(label_map), y_pred.map(label_map))
    
    cm = confusion_matrix(y_true, y_pred, labels=scheme.labels)
    cm_dict = {t: {p: int(cm[i][j]) for j, p in enumerate(scheme.labels)} for i, t in enumerate(scheme.labels)}
    
    per_class = {}
    for label in scheme.labels:
        per_class[label] = report[label]
        per_class[label]["f1"] = per_class[label].pop("f1-score", 0.0)

    metrics = {
        "metadata": _load_metadata(results_dir),
        "total_samples": len(df),
        "format_errors": int(format_errors),
        "format_error_rate": float(format_errors / len(df)),
        "global_accuracy": acc,
        "mean_absolute_error": mae,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm_dict,
        "per_class_metrics": per_class
    }
    
    print(f"\n=== Classification Metrics ===")
    print(f"Total Samples: {len(df)} | Format Errors: {format_errors}")
    print(f"Accuracy: {acc*100:.1f}% | MAE: {mae:.3f} classes | Macro F1: {metrics['macro_f1']:.3f}")
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def _compute_emission_line_metrics(results_dir: str, task: Any):
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path): return

    df = pd.read_json(results_path, lines=True)
    if df.empty: return

    # Standardize columns
    df["gt"] = df.get("ground_truth_lines", pd.Series([{}])).apply(lambda x: x if isinstance(x, dict) else {})
    df["pred"] = df.get("predicted_lines", df.get("parsed", pd.Series([[]]))).apply(lambda x: x if isinstance(x, list) else [])
    
    df["gt_set"] = df["gt"].apply(set)
    df["pred_set"] = df["pred"].apply(set)
    df["tp_set"] = [g & p for g, p in zip(df["gt_set"], df["pred_set"])]

    # Format errors
    has_raw = df.get("raw_response", pd.Series([""]*len(df))).fillna("")
    is_none = has_raw.str.upper().str.contains("NONE")
    format_errors = int(((df["pred_set"].apply(len) == 0) & ~is_none & (has_raw != "")).sum())

    # Sample-level stats
    df["len_tp"] = df["tp_set"].apply(len)
    df["len_gt"] = df["gt_set"].apply(len)
    df["len_pred"] = df["pred_set"].apply(len)
    
    df["gt_snr"] = [sum(np.log1p(g.get(l, 0)) for l in g) for g in df["gt"]]
    df["tp_snr"] = [sum(np.log1p(g.get(l, 0)) for l in tp) for g, tp in zip(df["gt"], df["tp_set"])]

    # Safe division helpers
    def safe_div(num, den, empty_cond=None):
        res = num / den.replace(0, np.nan)
        if empty_cond is not None:
            res = np.where((den == 0) & empty_cond, 1.0, res)
        return pd.Series(res).fillna(0.0)

    df["p"] = safe_div(df["len_tp"], df["len_pred"], df["len_gt"] == 0)
    df["r"] = safe_div(df["len_tp"], df["len_gt"], df["len_pred"] == 0)
    df["f1"] = safe_div(2 * df["p"] * df["r"], df["p"] + df["r"])
    
    df["rw"] = safe_div(df["tp_snr"], df["gt_snr"], df["len_pred"] == 0)
    df["f1w"] = safe_div(2 * df["p"] * df["rw"], df["p"] + df["rw"])
    
    means = df[["p", "r", "f1", "rw", "f1w"]].mean().to_dict()
    
    # Micro metrics
    tot = df[["len_tp", "len_pred", "len_gt", "tp_snr", "gt_snr"]].sum()
    mic_p = tot["len_tp"] / tot["len_pred"] if tot["len_pred"] > 0 else 0.0
    mic_r = tot["len_tp"] / tot["len_gt"] if tot["len_gt"] > 0 else 0.0
    mic_f1 = 2 * mic_p * mic_r / (mic_p + mic_r) if (mic_p + mic_r) > 0 else 0.0
    mic_rw = tot["tp_snr"] / tot["gt_snr"] if tot["gt_snr"] > 0 else 0.0
    mic_f1w = 2 * mic_p * mic_rw / (mic_p + mic_rw) if (mic_p + mic_rw) > 0 else 0.0

    # Per-line metrics
    lines = getattr(task, "canonical_lines", [])
    mlb = MultiLabelBinarizer(classes=lines)
    mlb.fit(df["gt_set"])
    yt = mlb.transform(df["gt_set"])
    yp = mlb.transform(df["pred_set"])
    
    report = classification_report(yt, yp, target_names=lines, output_dict=True, zero_division=0)
    
    per_line_metrics = {}
    line_f1s = []
    for i, line in enumerate(lines):
        t_mask = yt[:, i] == 1
        p_mask = yp[:, i] == 1
        
        snrs = df.loc[t_mask, "gt"].apply(lambda d: d.get(line, 0))
        det_snrs = df.loc[t_mask & p_mask, "gt"].apply(lambda d: d.get(line, 0))
        
        report[line].update({
            "f1": report[line].pop("f1-score"),
            "mean_snr": float(snrs.mean()) if not snrs.empty else 0.0,
            "mean_detected_snr": float(det_snrs.mean()) if not det_snrs.empty else 0.0,
            "tp": int((t_mask & p_mask).sum()),
            "fp": int((~t_mask & p_mask).sum()),
            "fn": int((t_mask & ~p_mask).sum()),
            "support": int(t_mask.sum())
        })
        per_line_metrics[line] = report[line]
        line_f1s.append(report[line]["f1"])

    macro_line_f1 = float(np.mean(line_f1s)) if line_f1s else 0.0
    exact_matches = int((df["gt_set"] == df["pred_set"]).sum())
    
    metrics = {
        "metadata": _load_metadata(results_dir),
        "total_samples": len(df),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(df),
        "format_errors": format_errors,
        "sample_level": {
            "mean_precision": means["p"],
            "mean_recall": means["r"],
            "mean_f1": means["f1"],
            "mean_snr_weighted_recall": means["rw"],
            "mean_snr_weighted_f1": means["f1w"],
        },
        "dataset_micro_level": {
            "micro_precision": mic_p,
            "micro_recall": mic_r,
            "micro_f1": mic_f1,
            "micro_snr_weighted_recall": mic_rw,
            "micro_snr_weighted_f1": mic_f1w,
        },
        "dataset_macro_level": {
            "mean_line_f1": macro_line_f1,
        },
        "per_line_metrics": per_line_metrics,
    }
    
    print(f"\n=== Emission Line Metrics ===")
    print(f"Samples: {len(df)} | Exact Match: {exact_matches/len(df)*100:.1f}% | Format Errors: {format_errors}")
    print(f"Sample Means -> Prec: {means['p']*100:.1f}% | Rec: {means['r']*100:.1f}% | F1: {means['f1']:.3f} | SNR-F1: {means['f1w']:.3f}")
    print(f"Micro Totals -> Prec: {mic_p*100:.1f}% | Rec: {mic_r*100:.1f}% | F1: {mic_f1:.3f} | SNR-F1: {mic_f1w:.3f}")
    print(f"Macro Means  -> Line F1: {macro_line_f1:.3f}")
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
