import os
import sys
import json
import argparse

# Ensure src is in the python path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from evals.tasks import get_task
from evals.metrics import compute_and_save_metrics

def main():
    parser = argparse.ArgumentParser(description="Recompute metrics for an evaluation run.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory (e.g. eval_results/20260907_133755_distance_eval)")
    args = parser.parse_args()

    results_dir = args.results_dir
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        sys.exit(1)

    metadata_path = os.path.join(results_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {results_dir}")
        sys.exit(1)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Determine the task
    task_info = metadata.get("task", {})
    task_name = task_info.get("task_name")
    
    if not task_name:
        # Fallback if task_name is not in metadata
        print("Warning: Could not determine task_name from metadata.json.")
        task_name = input("Enter task name (e.g. 'emission_lines' or 'distance_classification'): ").strip()

    print(f"Detected task: {task_name}")

    if task_name == "distance_classification":
        # Check old metadata format vs new format
        scheme_info = task_info.get("bucket_scheme", {})
        scheme = scheme_info.get("name") if isinstance(scheme_info, dict) else metadata.get("run_config", {}).get("bucket_scheme", "3-group")
        task = get_task(task_name, scheme=scheme)
    elif task_name in ["source_classification", "subclass_classification"]:
        active_classes = task_info.get("active_classes", {})
        task = get_task(task_name, active_classes=active_classes)
    else:
        task = get_task(task_name)

    print(f"Recomputing metrics for {results_dir}...")
    compute_and_save_metrics(results_dir, task)
    print(f"Done! Check {os.path.join(results_dir, 'metrics.json')} for the updated metrics.")

if __name__ == "__main__":
    main()
