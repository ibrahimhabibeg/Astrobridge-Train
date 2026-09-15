#!/usr/bin/env python
"""Generate and cache captions for astronomical spectra using Modal GPU across benchmarks.

Usage:
    modal run eval_scripts/generate_captions_modal.py \
        --responder eval_configs/caption_responders/astrobridge.yaml \
        --output eval_results/cached_captions/astrobridge_captions.jsonl

Options:
    --limit 50              # Limit to first N samples
    --gpu A100-80GB         # Override GPU type (e.g. A100-80GB, A10G)
    --batch-size 32         # Batch size for generation
    --caption-prompt "..."  # Override captioning prompt
    --benchmark all         # all | emission_lines | redshift | source_class | subclass
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
import yaml

import sys
from pathlib import Path

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

try:
    import modal
    volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
    app = modal.App("astrobridge-caption-generation")
except ImportError:
    modal = None
    volume = None
    app = None


def download_models():
    """Pre-bake required models and benchmark datasets into the Modal image cache."""
    import sys
    from pathlib import Path
    sys.path.insert(0, "/root/src")
    from huggingface_hub import hf_hub_download, snapshot_download
    from evals.config import (
        DEFAULT_BASE_LLM_REPO,
        DEFAULT_MODEL_REPO,
    )
    from evals.data import ensure_all_benchmark_files

    print(f"Downloading AstroBridge weights: {DEFAULT_MODEL_REPO}")
    try:
        snapshot_download(DEFAULT_MODEL_REPO)
        hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename="middle.pt")
    except Exception as e:
        print(f"Notice during AstroBridge download: {e}")

    print(f"Downloading Base LLM: {DEFAULT_BASE_LLM_REPO}")
    snapshot_download(DEFAULT_BASE_LLM_REPO)

    print("Downloading benchmark evaluation datasets...")
    ensure_all_benchmark_files(target_dir="/root/data/benchmarks")


if app is not None:
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
        split: str = None,
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


def main_local(
    responder: str,
    output: str = None,
    benchmark: str = "all",
    dataset_filter: str = None,
    limit: int = None,
    batch_size: int = None,
    caption_prompt: str = None,
    gpu: str = None,
    split: str = None,
):
    with open(responder, "r") as f:
        responder_config = yaml.safe_load(f)

    effective_batch_size = (
        batch_size
        if batch_size is not None
        else responder_config.get("batch_size", 32)
    )

    if caption_prompt:
        responder_config["caption_prompt"] = caption_prompt

    target_bench = dataset_filter or benchmark or "all"
    resp_tag = responder_config.get("responder_type", "model")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{target_bench}_captions")

    gpu_type = gpu or responder_config.get("gpu", "A100-80GB")
    print(f"Launching remote caption generation on Modal GPU ({gpu_type}) for benchmark '{target_bench}' with batch size {effective_batch_size}...")

    generate_captions_remote.with_options(gpu=gpu_type).remote(
        responder_config=responder_config,
        timestamp_dir=timestamp_dir,
        benchmark=target_bench,
        limit=limit,
        batch_size=effective_batch_size,
        caption_prompt=caption_prompt,
        split=split,
    )

    # Sync output from Modal volume to local machine
    temp_sync_dir = os.path.join(os.getcwd(), "eval_results", "temp_modal_sync")
    os.makedirs(temp_sync_dir, exist_ok=True)

    print(f"Syncing generated captions from Modal volume to local...")
    subprocess.run(
        ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, temp_sync_dir],
        check=True,
    )

    downloaded_file = os.path.join(temp_sync_dir, timestamp_dir, "captions.jsonl")

    # Determine final local output path
    if output:
        final_output_path = os.path.abspath(output)
    else:
        final_output_path = os.path.join(os.getcwd(), "eval_results", "cached_captions", f"{timestamp_dir}.jsonl")

    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    shutil.move(downloaded_file, final_output_path)

    # Clean up sync directory
    try:
        shutil.rmtree(os.path.join(temp_sync_dir, timestamp_dir))
    except Exception:
        pass

    print(f"\nSuccessfully generated and saved captions to:")
    print(f"  -> {final_output_path}")
    print(f"\nYou can now evaluate these captions with any task in seconds without using a GPU, e.g.:")
    print(f"  uv run eval_scripts/run_caption_eval_local.py \\")
    print(f"      --task eval_configs/caption_tasks/distance.yaml \\")
    print(f"      --captions {final_output_path} \\")
if app is not None:
    main = app.local_entrypoint()(main_local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal script to generate captions.")
    parser.add_argument("--responder", type=str, required=True, help="Path to responder config.")
    parser.add_argument("--output", type=str, default=None, help="Local output path.")
    parser.add_argument("--benchmark", type=str, default="all", help="Benchmark target (all, redshift, emission_lines, etc.).")
    parser.add_argument("--dataset-filter", type=str, default=None, help="Alias for --benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Prompt override.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU type override.")
    args, _ = parser.parse_known_args()

    bench = args.dataset_filter or args.benchmark
    print(
        f"Please run with Modal CLI:\n"
        f"  modal run eval_scripts/generate_captions_modal.py "
        f"--responder {args.responder} "
        f"--benchmark {bench} "
        + (f"--output {args.output} " if args.output else "")
        + (f"--limit {args.limit} " if args.limit else "")
    )
