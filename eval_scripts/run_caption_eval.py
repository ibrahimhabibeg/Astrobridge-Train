#!/usr/bin/env python
"""Unified Modal/Local caption-based evaluation script.

Usage:
    # Run full pipeline (Modal GPU for captioning + Local Gemini for property prediction)
    modal run eval_scripts/run_caption_eval.py \
        --task eval_configs/caption_tasks/distance.yaml \
        --responder eval_configs/caption_responders/astrobridge.yaml \
        --frontier eval_configs/frontier/gemini.yaml \
        --limit 100

    # Or run purely from cached captions locally:
    python eval_scripts/run_caption_eval.py \
        --task eval_configs/caption_tasks/distance.yaml \
        --captions eval_results/cached_captions/astrobridge_captions.jsonl \
        --frontier eval_configs/frontier/gemini.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import dotenv
import yaml
import numpy as np

# Ensure repo root and src/ are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.config import (
    DEFAULT_BASE_LLM_REPO,
    DEFAULT_MODEL_REPO,
)
from evals.data import ensure_all_benchmark_files

# Modal setup
try:
    import modal
    volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
    app = modal.App("astrobridge-caption-evaluation")

    def download_models():
        import sys
        from pathlib import Path
        sys.path.insert(0, "/root/src")
        from huggingface_hub import hf_hub_download, snapshot_download
        from evals.config import (
            DEFAULT_BASE_LLM_REPO,
            DEFAULT_MODEL_REPO,
        )
        from evals.data import ensure_all_benchmark_files

        print(f"Downloading AstroBridge extra weights: {DEFAULT_MODEL_REPO}")
        try:
            snapshot_download(DEFAULT_MODEL_REPO)
            hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename="middle.pt")
        except Exception as e:
            print(f"Notice: {e}")

        print(f"Downloading Base LLM: {DEFAULT_BASE_LLM_REPO}")
        snapshot_download(DEFAULT_BASE_LLM_REPO)

        print("Downloading benchmark evaluation datasets...")
        ensure_all_benchmark_files(target_dir="/root/data/benchmarks")

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install_from_pyproject("pyproject.toml")
        .pip_install("huggingface_hub", "pyyaml", "tqdm", "matplotlib", "Pillow", "torchvision")
        .run_function(download_models)
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("configs", remote_path="/root/configs")
        .add_local_dir("eval_scripts", remote_path="/root/eval_scripts")
        .add_local_dir("eval_configs", remote_path="/root/eval_configs")
        .add_local_dir("data", remote_path="/root/data")
    )

    @app.function(
        image=image,
        timeout=86400,
        volumes={"/outputs": volume},
    )
    def generate_captions_remote(
        responder_config: dict,
        timestamp_dir: str,
        benchmark: str = "all",
        limit: int = None,
        batch_size: int = 32,
        caption_prompt: str = None,
    ):
        """Remote function executing GPU caption generation on Modal by calling the local script logic."""
        import sys
        sys.path.insert(0, "/root")
        sys.path.insert(0, "/root/src")
        os.chdir("/root")

        from eval_scripts.generate_captions import run_caption_generation

        out_dir = os.path.join("/outputs", timestamp_dir)
        os.makedirs(out_dir, exist_ok=True)
        captions_file = os.path.join(out_dir, "captions.jsonl")

        total = run_caption_generation(
            responder_config=responder_config,
            output_path=captions_file,
            benchmark=benchmark or "all",
            limit=limit,
            batch_size=batch_size,
            caption_prompt=caption_prompt,
        )
        volume.commit()
        print(f"[Modal Remote] Done. Successfully generated and committed {total} captions in {captions_file}")
        return timestamp_dir

except ImportError:
    app = None


def main_local(
    task: str,
    frontier: str,
    responder: str = None,
    captions: str = None,
    limit: int = None,
    batch_size: int = 32,
    caption_prompt: str = None,
    gemini_model: str = None,
    gpu: str = None,
):
    dotenv.load_dotenv()
    with open(task, "r") as f:
        task_config = yaml.safe_load(f)
    with open(frontier, "r") as f:
        frontier_config = yaml.safe_load(f)

    responder_config = None
    if responder:
        with open(responder, "r") as f:
            responder_config = yaml.safe_load(f)

    if gemini_model:
        frontier_config["gemini_model"] = gemini_model

    resp_tag = responder_config.get("responder_type") if responder_config else "cached_captions"
    task_name = task_config.get("name", "task")
    benchmark_name = task_config.get("benchmark")
    if not benchmark_name:
        from evals.caption_tasks import get_caption_task
        task = get_caption_task(task_name, **task_config.get("kwargs", {}))
        benchmark_name = getattr(task, "benchmark_name", task_name)

    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{task_name}")

    local_output_dir = os.path.join(os.getcwd(), "eval_results", "caption_eval", timestamp_dir)
    os.makedirs(local_output_dir, exist_ok=True)

    captions_file = captions

    if not captions_file and responder_config:
        if responder_config.get("responder_type") == "mock" or os.environ.get("RUN_LOCAL_ONLY"):
            print("Running caption generation locally...")
            from eval_scripts.run_caption_eval_local import run_caption_evaluation
            run_caption_evaluation(
                task_config=task_config,
                frontier_config=frontier_config,
                responder_config=responder_config,
                captions_file=None,
                output_dir=local_output_dir,
                limit=limit,
                caption_prompt_override=caption_prompt,
                batch_size=batch_size,
            )
            return
        else:
            gpu_type = gpu or responder_config.get("gpu", "A100-80GB")
            print(f"Running Caption Generation REMOTELY on Modal GPU ({gpu_type})...")
            generate_captions_remote.with_options(gpu=gpu_type).remote(
                responder_config=responder_config,
                timestamp_dir=timestamp_dir,
                benchmark=benchmark_name,
                limit=limit,
                batch_size=batch_size,
                caption_prompt=caption_prompt,
            )
            print("Caption generation completed on Modal. Syncing captions to local output dir...")
            subprocess.run(
                ["modal", "volume", "get", "astrobridge-evals", f"{timestamp_dir}/captions.jsonl", local_output_dir],
                check=True,
            )
            captions_file = os.path.join(local_output_dir, "captions.jsonl")

    # Run Stage 2 & 3 locally using the Frontier VLM
    print("Evaluating captions with Frontier VLM locally...")
    from eval_scripts.run_caption_eval_local import run_caption_evaluation
    run_caption_evaluation(
        task_config=task_config,
        frontier_config=frontier_config,
        responder_config=None,
        captions_file=captions_file,
        output_dir=local_output_dir,
        limit=limit,
        caption_prompt_override=caption_prompt,
    )


if app is not None:
    main = app.local_entrypoint()(main_local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified caption evaluation script.")
    parser.add_argument("--task", type=str, required=True, help="Path to task YAML config.")
    parser.add_argument("--frontier", type=str, required=True, help="Path to frontier model YAML config.")
    parser.add_argument("--responder", type=str, default=None, help="Path to responder YAML config.")
    parser.add_argument("--captions", type=str, default=None, help="Path to captions.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for captioning.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Prompt override.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Gemini model override.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU override for Modal.")
    args = parser.parse_args()

    main_local(
        task=args.task,
        frontier=args.frontier,
        responder=args.responder,
        captions=args.captions,
        limit=args.limit,
        batch_size=args.batch_size,
        caption_prompt=args.caption_prompt,
        gemini_model=args.gemini_model,
        gpu=args.gpu,
    )
