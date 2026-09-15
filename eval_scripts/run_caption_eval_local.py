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
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.caption_responders import CaptionSample, get_caption_responder
from evals.caption_tasks import get_caption_task
from evals.frontier import get_frontier_model
from evals.caption_eval import compute_caption_metrics
from evals.data import load_benchmark_dataset


def run_caption_evaluation(
    task_config: dict,
    frontier_config: dict,
    responder_config: Optional[dict],
    captions_file: Optional[str],
    output_dir: str,
    limit: Optional[int] = None,
    caption_prompt_override: Optional[str] = None,
    batch_size: int = 32,
    split: Optional[str] = None,
):
    """Executes the caption evaluation pipeline locally."""
    import torch

    os.makedirs(output_dir, exist_ok=True)
    task_name = task_config["name"]
    task_kwargs = task_config.get("kwargs", {})
    task = get_caption_task(task_name, **task_kwargs)

    frontier = get_frontier_model(frontier_config)

    # Load dedicated benchmark dataset
    benchmark_name = task_config.get("benchmark") or getattr(task, "benchmark_name", None)
    if not benchmark_name:
        raise ValueError(
            f"Task '{task_name}' does not specify a canonical benchmark dataset. "
            f"Please specify 'benchmark' in the task YAML or on the CaptionEvalTask."
        )
    print(f"Loading benchmark dataset '{benchmark_name}' for task '{task_name}'...")
    df_test = load_benchmark_dataset(benchmark_name)

    if limit is not None:
        df_test = df_test.head(limit)

    gt_by_id: Dict[str, Any] = {}
    survey_by_id: Dict[str, str] = {}
    rows_by_id: Dict[str, Any] = {}

    for _, row in df_test.iterrows():
        sid = str(row["sample_id"])
        gt_by_id[sid] = task.extract_ground_truth(row)
        survey_by_id[sid] = str(row.get("survey", "unknown"))
        rows_by_id[sid] = row

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
                    sid = str(item.get("sample_id") or item.get("object_id") or item.get("wiki_entity_id"))
                    if sid in gt_by_id:
                        captions_by_id[sid] = item

        matched_count = len(captions_by_id)
        print(f"Matched {matched_count}/{total_in_file} captions against benchmark ({len(gt_by_id)} target samples).")
    elif responder_config:
        captions_path = os.path.join(output_dir, "captions.jsonl")
        from eval_scripts.generate_captions import run_caption_generation
        run_caption_generation(
            responder_config=responder_config,
            output_path=captions_path,
            benchmark=benchmark_name,
            limit=limit,
            batch_size=batch_size,
            caption_prompt=caption_prompt_override,
            df=df_test,
        )
        with open(captions_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = str(item.get("sample_id") or item.get("object_id") or item.get("wiki_entity_id"))
                    if sid in gt_by_id:
                        captions_by_id[sid] = item
    else:
        raise ValueError("Must provide either responder configuration or --captions file.")

    # Filter to samples where we have both ground truth and captions
    eval_ids = [sid for sid in gt_by_id if sid in captions_by_id]
    print(f"Stage 2: Evaluating {len(eval_ids)} samples using Frontier VLM...")

    frontier_prompts = []
    for sid in eval_ids:
        cap_text = captions_by_id[sid]["caption"]
        row_item = rows_by_id[sid]
        prompt = task.build_frontier_prompt(cap_text, item=row_item)
        frontier_prompts.append(prompt)

    frontier_responses = frontier.predict_batch(
        prompts=frontier_prompts,
        parse_fn=task.default_parse,
        fallback_tag=task.fallback_tag(),
    )

    # Compile predictions.jsonl
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w") as pred_f:
        for sid, prompt, f_resp in zip(eval_ids, frontier_prompts, frontier_responses):
            gt = gt_by_id[sid]
            pred = f_resp.parsed
            row_item = rows_by_id[sid]
            is_correct = False

            if "emission_lines" in task_name:
                candidates = list(row_item.get("candidate_query_lines", []))
                if candidates and isinstance(pred, (list, set)):
                    cand_set = {str(k) for k in candidates}
                    pred = [l for l in pred if l in cand_set]

                gt_set = set(gt.keys()) if isinstance(gt, dict) else set(gt)
                pred_set = set(pred) if isinstance(pred, (list, set)) else set()
                is_correct = (gt_set == pred_set)
            else:
                is_correct = (str(pred).strip().lower() == str(gt).strip().lower())

            record = {
                "sample_id": sid,
                "object_id": sid,
                "survey": survey_by_id.get(sid, "unknown"),
                "task": task_name,
                "ground_truth": gt,
                "caption": {
                    "text": captions_by_id[sid]["caption"],
                    "responder_type": captions_by_id[sid].get("responder_type", "unknown"),
                    "model_id": captions_by_id[sid].get("model_id", "unknown"),
                    "prompt_used": captions_by_id[sid].get("caption_prompt", ""),
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

            if "regime" in row_item:
                record["regime"] = str(row_item["regime"])
            if "candidate_query_lines" in row_item:
                record["candidate_query_lines"] = [str(k) for k in row_item["candidate_query_lines"]]

            pred_f.write(json.dumps(record) + "\n")

    # Save run_config.json
    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(
            {
                "task": task.get_config(),
                "frontier": frontier.get_config(),
                "responder": responder_config if responder_config else {"source": captions_file},
                "benchmark": task_name,
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
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for captioning.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Override caption prompt.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model.")
    parser.add_argument("--split", type=str, default=None, help="Deprecated (datasets are pre-stratified benchmarks).")
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

    output_dir = os.path.join(os.getcwd(), "eval_results", "caption_eval", timestamp_dir)
    print(f"Starting caption evaluation -> Output dir: {output_dir}")

    run_caption_evaluation(
        task_config=task_config,
        frontier_config=frontier_config,
        responder_config=responder_config,
        captions_file=args.captions,
        output_dir=output_dir,
        limit=args.limit,
        caption_prompt_override=args.caption_prompt,
        batch_size=args.batch_size,
        split=args.split,
    )


if __name__ == "__main__":
    main()
