#!/usr/bin/env python
"""Local caption-based evaluation script.

Runs two-stage evaluation:
  Stage 1: Generates captions via specialized CaptionResponder (or loads from --captions)
  Stage 2: Queries Frontier VLM (Gemini / mock) to predict physical properties from captions
  Stage 3: Computes task metrics, caption diagnostics, and generates a rich markdown report.

Usage:
    # Full end-to-end evaluation
    uv run eval_scripts/run_caption_eval_local.py \
        --task eval_configs/caption_tasks/distance.yaml \
        --responder eval_configs/caption_responders/astrobridge.yaml \
        --frontier eval_configs/frontier/gemini.yaml \
        --limit 10

    # Fast evaluation from cached captions
    uv run eval_scripts/run_caption_eval_local.py \
        --task eval_configs/caption_tasks/distance.yaml \
        --captions eval_results/cached_captions/astrobridge_captions.jsonl \
        --frontier eval_configs/frontier/gemini.yaml
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

from evals.caption_responders import CaptionSample, get_caption_responder
from evals.caption_tasks import get_caption_task
from evals.frontier import get_frontier_model
from evals.caption_eval import compute_caption_metrics
from evals.data import (
    load_test_spectra,
    load_test_spectra_by_category,
    load_test_spectra_emission_lines,
)


def run_caption_evaluation(
    task_config: dict,
    frontier_config: dict,
    responder_config: Optional[dict],
    captions_file: Optional[str],
    output_dir: str,
    split: str = "legacy",
    limit: Optional[int] = None,
    caption_prompt_override: Optional[str] = None,
    batch_size: int = 32,
):
    """Executes the caption evaluation pipeline locally."""
    import torch

    os.makedirs(output_dir, exist_ok=True)
    task_name = task_config["name"]
    task_kwargs = task_config.get("kwargs", {})
    task = get_caption_task(task_name, **task_kwargs)

    frontier = get_frontier_model(frontier_config)

    # Load appropriate dataset for ground truth mapping
    print(f"Loading ground truth dataset for {task_name}...")
    print(f"Loading ground truth dataset for {task_name} (split='{split}')...")
    if "emission_lines" in task_name:
        df_test = load_test_spectra_emission_lines()
        df_test = load_test_spectra_emission_lines(split_version=split)
    elif "source" in task_name:
        active_keys = list(task_kwargs.get("active_classes", {"GALAXY": "Galaxy", "QSO": "Quasar"}).keys())
        df_test = load_test_spectra_by_category("class", active_keys)
        df_test = load_test_spectra_by_category("class", active_keys, split_version=split)
    elif "subclass" in task_name:
        active_keys = list(task_kwargs.get("active_classes", {}).keys())
        df_test = load_test_spectra_by_category("subclass", active_keys)
        df_test = load_test_spectra_by_category("subclass", active_keys, split_version=split)
    else:
        df_test = load_test_spectra()
        df_test = load_test_spectra(split_version=split)

    if limit is not None:
        df_test = df_test.head(limit)

    gt_by_id: Dict[str, Any] = {}
    survey_by_id: Dict[str, str] = {}
    for _, row in df_test.iterrows():
        eid = str(row["wiki_entity_id"])
        gt_by_id[eid] = task.extract_ground_truth(row)
        survey_by_id[eid] = str(row.get("survey", "sdss"))

    # Stage 1: Obtain Captions
    captions_by_id: Dict[str, Dict[str, Any]] = {}

    if captions_file and os.path.exists(captions_file):
        print(f"Loading pre-generated captions from {captions_file}...")
        total_in_file = 0
        with open(captions_file, "r") as f:
            for line in f:
                if line.strip():
                    total_in_file += 1
                    item = json.loads(line)
                    eid = str(item["wiki_entity_id"])
                    if eid in gt_by_id:
                        captions_by_id[eid] = item

        matched_count = len(captions_by_id)
        print(f"Matched {matched_count}/{total_in_file} captions against ground truth (split='{split}').")
        if total_in_file > 0 and matched_count < 0.5 * total_in_file and split == "legacy":
            print(
                f"[Notice] Low match rate ({matched_count}/{total_in_file}). If these captions were "
                f"generated on the v7 split, please pass '--split v7'."
            )
    elif responder_config:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing caption responder '{responder_config.get('responder_type')}' on {device}...")
        responder = get_caption_responder(responder_config, device)

        def chunker(seq, size):
            return (seq[pos : pos + size] for pos in range(0, len(seq), size))

        print(f"Generating captions for {len(df_test)} test spectra...")
        with open(os.path.join(output_dir, "captions.jsonl"), "w") as cap_f:
            for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Stage 1: Caption Generation"):
                samples = []
                surveys = batch_df["survey"].tolist() if "survey" in batch_df.columns else ["sdss"] * len(batch_df)

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
                    samples.append(
                        CaptionSample(
                            sample_id=str(row["wiki_entity_id"]),
                            wavelength=wavelength,
                            flux=flux,
                            mask=mask,
                            survey=surveys[i],
                            ivar=ivar,
                        )
                    )

                gen_captions = responder.generate_captions(samples, prompt_override=caption_prompt_override)
                for c in gen_captions:
                    rec = {
                        "wiki_entity_id": c.sample_id,
                        "survey": c.survey,
                        "caption": c.caption,
                        "responder_type": c.responder_type,
                        "model_id": c.model_id,
                        "caption_prompt": c.caption_prompt,
                    }
                    captions_by_id[c.sample_id] = rec
                    cap_f.write(json.dumps(rec) + "\n")
    else:
        raise ValueError("Must provide either responder configuration or --captions file.")

    # Filter to samples where we have both ground truth and captions
    eval_ids = [eid for eid in gt_by_id if eid in captions_by_id]
    print(f"Stage 2: Evaluating {len(eval_ids)} samples using Frontier VLM...")

    frontier_prompts = []
    for eid in eval_ids:
        cap_text = captions_by_id[eid]["caption"]
        prompt = task.build_frontier_prompt(cap_text)
        frontier_prompts.append(prompt)

    frontier_responses = frontier.predict_batch(
        prompts=frontier_prompts,
        parse_fn=task.default_parse,
        fallback_tag=task.fallback_tag(),
    )

    # Compile predictions.jsonl
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w") as pred_f:
        for eid, prompt, f_resp in zip(eval_ids, frontier_prompts, frontier_responses):
            gt = gt_by_id[eid]
            pred = f_resp.parsed
            is_correct = False

            if "emission_lines" in task_name:
                # Set comparison for multilabel
                gt_set = set(gt.keys()) if isinstance(gt, dict) else set(gt)
                pred_set = set(pred) if isinstance(pred, (list, set)) else set()
                is_correct = (gt_set == pred_set)
            else:
                is_correct = (str(pred).strip().lower() == str(gt).strip().lower())

            record = {
                "sample_id": eid,
                "survey": survey_by_id.get(eid, "unknown"),
                "task": task_name,
                "ground_truth": gt,
                "caption": {
                    "text": captions_by_id[eid]["caption"],
                    "responder_type": captions_by_id[eid].get("responder_type", "unknown"),
                    "model_id": captions_by_id[eid].get("model_id", "unknown"),
                    "prompt_used": captions_by_id[eid].get("caption_prompt", ""),
                },
                "frontier_evaluation": {
                    "frontier_model": frontier_config.get("gemini_model", frontier_config.get("frontier_type")),
                    "prompt": prompt,
                    "raw_response": f_resp.raw_text,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "forced_fallback": f_resp.forced_fallback,
                },
            }
            pred_f.write(json.dumps(record) + "\n")

    # Save run_config.json
    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(
            {
                "task": task.get_config(),
                "frontier": frontier.get_config(),
                "responder": responder_config if responder_config else {"source": captions_file},
                "split": split,
                "total_samples": len(eval_ids),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=4,
        )

    # Stage 3: Compute metrics and generate report.md
    print("Stage 3: Computing metrics and generating report.md...")
    metrics = compute_caption_metrics(output_dir, task)
    print("All done!")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run local caption-based evaluation.")
    parser.add_argument("--task", type=str, required=True, help="Path to caption task YAML config.")
    parser.add_argument("--frontier", type=str, required=True, help="Path to frontier model YAML config.")
    parser.add_argument("--responder", type=str, default=None, help="Path to caption responder YAML config.")
    parser.add_argument("--captions", type=str, default=None, help="Path to pre-generated captions.jsonl.")
    parser.add_argument("--split", type=str, default="legacy", choices=["legacy", "v7"], help="Dataset split: 'legacy' (default, 400 test) or 'v7' (641 val+test).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for captioning.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Override caption prompt.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.task, "r") as f:
        task_config = yaml.safe_load(f)
    with open(args.frontier, "r") as f:
        frontier_config = yaml.safe_load(f)

    responder_config = None
    if args.responder:
        with open(args.responder, "r") as f:
            responder_config = yaml.safe_load(f)

    if args.gemini_model:
        frontier_config["gemini_model"] = args.gemini_model

    resp_tag = responder_config.get("responder_type") if responder_config else "cached_captions"
    task_name = task_config.get("name", "task")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{task_name}")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{args.split}_{task_name}")

    output_dir = os.path.join(os.getcwd(), "eval_results", "caption_eval", timestamp_dir)
    print(f"Starting caption evaluation -> Output dir: {output_dir}")

    run_caption_evaluation(
        task_config=task_config,
        frontier_config=frontier_config,
        responder_config=responder_config,
        captions_file=args.captions,
        output_dir=output_dir,
        split=args.split,
        limit=args.limit,
        caption_prompt_override=args.caption_prompt,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

