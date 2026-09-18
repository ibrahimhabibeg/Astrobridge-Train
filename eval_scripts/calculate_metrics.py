from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.metrics.reporter import compute_caption_metrics


def _resolve_predictions_files(inputs: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in inputs:
        matches = [Path(p) for p in glob.glob(raw)] or [Path(raw)]
        for p in matches:
            if p.is_dir():
                cand = p / "predictions.jsonl"
                if cand.is_file():
                    resolved.append(cand)
            elif p.is_file():
                resolved.append(p)
    return sorted(list(set(resolved)))


def _print_metrics_summary(bundle: dict) -> None:
    task_name = bundle.get("task_name", "unknown")
    total = bundle.get("total_samples", 0)
    metrics = bundle.get("task_metrics", {})
    diag = bundle.get("caption_diagnostics", {})

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name} | Total Samples: {total}")
    print(f"{'=' * 60}")

    if "emission_lines" in task_name.lower():
        print(f"  Mean Jaccard Index:     {metrics.get('mean_jaccard', 0.0):.4f}")
    else:
        acc = metrics.get("accuracy", 0.0)
        corr = metrics.get("correct_samples", int(round(acc * total)))
        print(f"  Accuracy:               {acc * 100:.2f}% ({corr}/{total})")

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, dict) and cm:
            print("\nConfusion Matrix:")
            true_labels = list(cm.keys())
            pred_cols = list(next(iter(cm.values())).keys())
            col_width = max(max(len(str(c)) for c in pred_cols), 10) + 2
            row_width = max(max(len(str(r)) for r in true_labels), 12) + 2

            header = f"{'True \\ Pred':<{row_width}}" + "".join(f"{c:>{col_width}}" for c in pred_cols)
            print(f"  {header}")
            for r in true_labels:
                row_str = f"{r:<{row_width}}" + "".join(f"{cm[r].get(c, 0):>{col_width}}" for c in pred_cols)
                print(f"  {row_str}")

    if diag:
        print("\nDiagnostics:")
        print(f"  Parse Success Rate:     {diag.get('frontier_parse_success_rate', 0.0) * 100:.1f}%")
        print(f"  Fallback Retry Rate:    {diag.get('frontier_fallback_rate', 0.0) * 100:.1f}%")
        print(f"  Avg Caption Words:      {diag.get('avg_caption_word_count', 0.0):.1f}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate metrics from predictions.jsonl.")
    parser.add_argument("inputs", nargs="+", help="Path(s) to predictions.jsonl or result directories.")
    args = parser.parse_args()

    files = _resolve_predictions_files(args.inputs)
    if not files:
        print("No valid predictions.jsonl files found.")
        sys.exit(1)

    for file_path in files:
        print(f"Processing: {file_path}")
        bundle = compute_caption_metrics(file_path)
        _print_metrics_summary(bundle)
        print(f"Updated {file_path.parent / 'metrics.json'}")


if __name__ == "__main__":
    main()
