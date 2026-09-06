import os
import json
from typing import Any, Dict, List, Set
from .buckets import BucketScheme

def compute_and_save_metrics(results_dir: str, task_or_scheme: Any):
    """
    Dispatcher to compute and save metrics based on task or bucket scheme.
    """
    if hasattr(task_or_scheme, "name") and task_or_scheme.name == "emission_lines":
        _compute_emission_line_metrics(results_dir, task_or_scheme)
    else:
        # Distance classification task or legacy BucketScheme
        scheme = getattr(task_or_scheme, "scheme", task_or_scheme)
        _compute_classification_metrics(results_dir, scheme)

def _compute_classification_metrics(results_dir: str, scheme: BucketScheme):
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"No results.jsonl found in {results_dir}")
        return
        
    y_true = []
    y_pred = []
    format_errors = 0
    total = 0
    
    label_map = {label: i+1 for i, label in enumerate(scheme.labels)}
    labels_str = "".join(scheme.labels)
    
    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1
            true_label = data["correct_answer"]
            pred_label = data.get("model_answer", data.get("parsed", "UNKNOWN"))
            
            if pred_label == "UNKNOWN" or pred_label not in label_map:
                format_errors += 1
                continue
                
            y_true.append(true_label)
            y_pred.append(pred_label)
            
    if not y_true and format_errors == 0:
        print("No valid predictions to compute metrics on.")
        return

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0
    
    mae = sum(abs(label_map[t] - label_map[p]) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    
    cm = {t: {p: 0 for p in labels_str} for t in labels_str}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
        
    per_class_metrics = {}
    for t in labels_str:
        row_total = sum(cm[t].values())
        recall = cm[t][t] / row_total if row_total > 0 else 0.0
        
        col_total = sum(cm[p][t] for p in labels_str)
        precision = cm[t][t] / col_total if col_total > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[t] = {
            "support": row_total,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / float(len(labels_str))
    
    total_valid = sum(m["support"] for m in per_class_metrics.values())
    if total_valid > 0:
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_valid
    else:
        weighted_f1 = 0.0

    metadata_path = os.path.join(results_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    metrics = {
        "metadata": metadata,
        "total_samples": total,
        "format_errors": format_errors,
        "format_error_rate": format_errors / total if total > 0 else 0.0,
        "global_accuracy": accuracy,
        "mean_absolute_error": mae,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics
    }
    
    print("\n=== Evaluation Metrics ===")
    print(f"Total Samples: {total}")
    print(f"Format Errors (UNKNOWN): {format_errors} ({(metrics['format_error_rate'])*100:.1f}%)")
    print(f"Global Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"Mean Absolute Error (Ordinal off-by-X): {mae:.3f} classes")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print("         Predicted")
    print("       " + "   ".join(list(labels_str)))
    for t in labels_str:
        row = [f"{cm[t][p]:3d}" for p in labels_str]
        print(f"True {t} " + " ".join(row))
        
    print("\n--- Per-Class Metrics ---")
    for t in labels_str:
        m = per_class_metrics[t]
        print(f"{t}: Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f} (Support: {m['support']})")
            
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

def _compute_emission_line_metrics(results_dir: str, task: Any):
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"No results.jsonl found in {results_dir}")
        return

    canonical_lines = getattr(task, "canonical_lines", [])

    total = 0
    format_errors = 0
    sample_precisions: List[float] = []
    sample_recalls: List[float] = []
    sample_f1s: List[float] = []
    sample_weighted_recalls: List[float] = []
    sample_weighted_f1s: List[float] = []
    exact_matches = 0

    total_tp = 0
    total_pred = 0
    total_truth = 0
    total_detected_snr = 0.0
    total_truth_snr = 0.0

    # Per-line tracking
    line_stats: Dict[str, Dict[str, Any]] = {
        line: {
            "support": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "snrs": [],
            "detected_snrs": [],
        }
        for line in canonical_lines
    }

    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1

            gt_dict: Dict[str, float] = data.get("ground_truth_lines", {})
            pred_list: List[str] = data.get("predicted_lines", data.get("parsed", []))
            if pred_list is None:
                pred_list = []

            # Check format error
            if "raw_response" in data and data["raw_response"] and not pred_list and "NONE" not in data["raw_response"].upper():
                format_errors += 1

            gt_set: Set[str] = set(gt_dict.keys())
            pred_set: Set[str] = set(pred_list)

            tp_set = gt_set & pred_set
            fp_set = pred_set - gt_set
            fn_set = gt_set - pred_set

            # Sample level unweighted precision & recall
            if len(pred_set) > 0:
                p = len(tp_set) / len(pred_set)
            else:
                p = 1.0 if len(gt_set) == 0 else 0.0

            if len(gt_set) > 0:
                r = len(tp_set) / len(gt_set)
            else:
                r = 1.0 if len(pred_set) == 0 else 0.0

            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            # Sample level SNR-weighted recall & F1
            sample_truth_snr = sum(gt_dict[line] for line in gt_set)
            sample_detected_snr = sum(gt_dict[line] for line in tp_set)

            if sample_truth_snr > 0:
                r_weighted = sample_detected_snr / sample_truth_snr
            else:
                r_weighted = 1.0 if len(pred_set) == 0 else 0.0

            f1_weighted = 2 * p * r_weighted / (p + r_weighted) if (p + r_weighted) > 0 else 0.0

            sample_precisions.append(p)
            sample_recalls.append(r)
            sample_f1s.append(f1)
            sample_weighted_recalls.append(r_weighted)
            sample_weighted_f1s.append(f1_weighted)

            if gt_set == pred_set:
                exact_matches += 1

            total_tp += len(tp_set)
            total_pred += len(pred_set)
            total_truth += len(gt_set)
            total_detected_snr += sample_detected_snr
            total_truth_snr += sample_truth_snr

            # Track per-line
            for line_name in canonical_lines:
                stats = line_stats[line_name]
                in_gt = line_name in gt_set
                in_pred = line_name in pred_set

                if in_gt:
                    stats["support"] += 1
                    snr_val = gt_dict[line_name]
                    stats["snrs"].append(snr_val)
                    if in_pred:
                        stats["tp"] += 1
                        stats["detected_snrs"].append(snr_val)
                    else:
                        stats["fn"] += 1
                else:
                    if in_pred:
                        stats["fp"] += 1

    if total == 0:
        print("No valid results found in results.jsonl")
        return

    # Aggregate metrics
    mean_precision = sum(sample_precisions) / total
    mean_recall = sum(sample_recalls) / total
    mean_f1 = sum(sample_f1s) / total
    mean_weighted_recall = sum(sample_weighted_recalls) / total
    mean_weighted_f1 = sum(sample_weighted_f1s) / total
    exact_match_rate = exact_matches / total

    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall = total_tp / total_truth if total_truth > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_weighted_recall = total_detected_snr / total_truth_snr if total_truth_snr > 0 else 0.0
    micro_weighted_f1 = 2 * micro_precision * micro_weighted_recall / (micro_precision + micro_weighted_recall) if (micro_precision + micro_weighted_recall) > 0 else 0.0

    # Per-line calculations
    per_line_metrics = {}
    for line_name, stats in line_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        support = stats["support"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mean_snr = sum(stats["snrs"]) / support if support > 0 else 0.0
        detected_mean_snr = sum(stats["detected_snrs"]) / tp if tp > 0 else 0.0

        per_line_metrics[line_name] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mean_snr": round(mean_snr, 2),
            "mean_detected_snr": round(detected_mean_snr, 2),
        }

    metadata_path = os.path.join(results_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    metrics = {
        "metadata": metadata,
        "total_samples": total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "format_errors": format_errors,
        "sample_level": {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "mean_snr_weighted_recall": mean_weighted_recall,
            "mean_snr_weighted_f1": mean_weighted_f1,
        },
        "dataset_micro_level": {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "micro_snr_weighted_recall": micro_weighted_recall,
            "micro_snr_weighted_f1": micro_weighted_f1,
        },
        "per_line_metrics": per_line_metrics,
    }

    print("\n=== Emission Line Evaluation Metrics ===")
    print(f"Total Samples: {total}")
    print(f"Exact Match Rate: {exact_match_rate * 100:.2f}% ({exact_matches}/{total})")
    print(f"Format Errors: {format_errors} ({format_errors / total * 100:.1f}%)")
    print("\n--- Sample-Level Averages ---")
    print(f"Mean Precision:               {mean_precision * 100:.2f}%")
    print(f"Mean Recall:                  {mean_recall * 100:.2f}%")
    print(f"Mean F1:                      {mean_f1:.4f}")
    print(f"Mean SNR-Weighted Recall:     {mean_weighted_recall * 100:.2f}%")
    print(f"Mean SNR-Weighted F1:         {mean_weighted_f1:.4f}")
    print("\n--- Micro-Averaged Totals ---")
    print(f"Micro Precision:              {micro_precision * 100:.2f}%")
    print(f"Micro Recall:                 {micro_recall * 100:.2f}%")
    print(f"Micro SNR-Weighted Recall:    {micro_weighted_recall * 100:.2f}%")
    print(f"Micro SNR-Weighted F1:        {micro_weighted_f1:.4f}")

    print("\n--- Per-Line Performance (Top active lines) ---")
    print(f"{'Line':16s} | {'Supp':>4s} | {'SNR':>5s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}")
    print("-" * 55)
    sorted_lines = sorted(per_line_metrics.items(), key=lambda item: item[1]["support"], reverse=True)
    for line_name, m in sorted_lines:
        if m["support"] > 0 or (m["tp"] + m["fp"]) > 0:
            print(f"{line_name:16s} | {m['support']:4d} | {m['mean_snr']:5.1f} | {m['precision']*100:5.1f}% | {m['recall']*100:5.1f}% | {m['f1']:6.3f}")

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
