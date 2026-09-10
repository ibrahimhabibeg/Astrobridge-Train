#!/usr/bin/env python
"""Local evaluation script — runs everything on the local machine.

Use this instead of run_eval.py when your local device has a GPU.
No Modal dependency required.

Usage:
    uv run eval_scripts/run_eval_local.py eval_configs/distance.yaml
    uv run eval_scripts/run_eval_local.py eval_configs/distance.yaml --limit 5
    uv run eval_scripts/run_eval_local.py eval_configs/distance.yaml --gemini-model gemini-3.5-flash-lite
"""

import argparse
import json
import os
from datetime import datetime

import dotenv
import numpy as np
import yaml

from evals.data import (
    load_test_spectra,
    load_test_spectra_by_category,
    load_test_spectra_emission_lines,
)
from evals.metrics import compute_and_save_metrics
from evals.responders import EvalSample, get_responder
from evals.tasks import get_task


def run_evaluation(config: dict, output_dir: str):
    """Run the full evaluation pipeline locally."""
    import torch
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected. Running on CPU — this will be slow for model-based responders.")

    task_config = config.get("task", {})
    task_name = task_config["name"]
    task_kwargs = task_config.get("kwargs", {})

    run_config = config.get("run", {})

    task = get_task(task_name, **task_kwargs)
    responder = get_responder(run_config, device)

    # Load appropriate dataset
    if task_name == "emission_lines":
        df_test = load_test_spectra_emission_lines()
    elif task_name == "source_classification":
        df_test = load_test_spectra_by_category("class", list(task_kwargs["active_classes"].keys()))
    elif task_name == "subclass_classification":
        df_test = load_test_spectra_by_category("subclass", list(task_kwargs["active_classes"].keys()))
    else:
        df_test = load_test_spectra()

    limit = run_config.get("limit")
    if limit is not None:
        print(f"Limiting evaluation to first {limit} samples.")
        df_test = df_test.head(limit)

    batch_size = run_config.get("batch_size", 64)
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "run_config": run_config,
        "responder": responder.get_config(),
        "task": task.get_config(),
        "batch_size": batch_size,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    for batch_df in tqdm(list(chunker(df_test, batch_size)), desc=f"Evaluating {task_name}"):
        wiki_entity_ids = batch_df["wiki_entity_id"].tolist()
        ground_truths = [task.extract_ground_truth(row) for _, row in batch_df.iterrows()]
        surveys = batch_df["survey"].tolist() if "survey" in batch_df.columns else ["sdss"] * len(batch_df)

        samples = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            spec_data = row["spectrum"]
            flux = np.array(spec_data["flux"])
            wavelength = np.array(spec_data["lambda"])
            mask = (
                np.array(spec_data["mask"]).astype(bool)
                if "mask" in spec_data
                else np.zeros_like(flux, dtype=bool)
            )
            ivar = np.array(spec_data["ivar"]) if "ivar" in spec_data else None
            samples.append(EvalSample(wavelength, flux, mask, surveys[i], ivar=ivar))

        try:
            answers = responder.respond_batch(samples, task)
            batch_results = []

            for wiki_entity_id, gt, resp in zip(wiki_entity_ids, ground_truths, answers):
                result = {
                    "wiki_entity_id": wiki_entity_id,
                    "correct_answer": gt if task_name != "emission_lines" else None,
                    "ground_truth_lines": gt if task_name == "emission_lines" else None,
                    "model_answer": resp.parsed if task_name != "emission_lines" else None,
                    "predicted_lines": resp.parsed if task_name == "emission_lines" else None,
                    "full_response": resp.raw_text,
                    "forced_fallback": getattr(resp, "forced_fallback", False),
                    "responder": metadata["responder"]["type"],
                    "task": metadata["task"]["task_name"],
                }
                result = {k: v for k, v in result.items() if v is not None}
                batch_results.append(result)

                tqdm.write(f"[{wiki_entity_id}] Truth: {gt} | Pred: {resp.parsed}")

            with open(os.path.join(output_dir, "results.jsonl"), "a") as f:
                for res in batch_results:
                    f.write(json.dumps(res) + "\n")

        except Exception as e:
            tqdm.write(f"Error on batch starting with {wiki_entity_ids[0] if wiki_entity_ids else 'unknown'}: {e}")

    print("Evaluation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation locally (assumes local GPU is available)."
    )
    parser.add_argument("config", type=str, help="Path to evaluation YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.limit is not None:
        config["run"]["limit"] = args.limit
    if args.gemini_model is not None:
        config["run"]["gemini_model"] = args.gemini_model

    run_config = config["run"]
    suffix = run_config.get("suffix_tag", "eval")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")

    output_base = os.path.join(os.getcwd(), "eval_results")
    output_dir = os.path.join(output_base, timestamp_dir)

    print(f"Starting LOCAL evaluation from {args.config}. Results -> {timestamp_dir}")
    run_evaluation(config, output_dir)

    print("Computing metrics...")
    task = get_task(config["task"]["name"], **config["task"].get("kwargs", {}))
    compute_and_save_metrics(output_dir, task)
    print(f"All done! Check {output_dir} for results and metrics.")


if __name__ == "__main__":
    main()

