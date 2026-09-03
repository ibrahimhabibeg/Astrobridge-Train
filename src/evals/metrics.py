import os
import json
from .buckets import BucketScheme

def compute_and_save_metrics(results_dir: str, scheme: BucketScheme):
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
            pred_label = data["model_answer"]
            
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

    metrics = {
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
