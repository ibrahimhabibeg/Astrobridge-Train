#!/usr/bin/env python
"""Generate and cache captions for astronomical spectra using Modal GPU.

Usage:
    modal run eval_scripts/generate_captions_modal.py \
        --responder eval_configs/caption_responders/astrobridge.yaml \
        --output eval_results/cached_captions/astrobridge_captions.jsonl

Options:
    --limit 50              # Limit to first N samples
    --gpu A100-80GB         # Override GPU type (e.g. A100-80GB, A10G)
    --batch-size 32         # Batch size for generation
    --caption-prompt "..."  # Override captioning prompt
    --dataset-filter all    # all | emission_lines | source | subclass
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
import yaml

import modal

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-caption-generation")


def download_models():
    """Pre-bake required models and datasets into the Modal image cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    astrobridge_id = "UniverseTBD/astrobridge-model-v5"
    base_llm_id = "Qwen/Qwen3.5-9B"

    print(f"Downloading AstroBridge weights: {astrobridge_id}")
    try:
        snapshot_download(astrobridge_id)
        hf_hub_download(repo_id=astrobridge_id, filename="middle.pt")
    except Exception as e:
        print(f"Notice during AstroBridge download: {e}")

    print(f"Downloading Base LLM: {base_llm_id}")
    snapshot_download(base_llm_id)

    DATASET_CACHE_VERSION = "2026-09-11-v2-400samples"
    DATASET_CACHE_VERSION = "2026-09-12-v3-dual-splits"
    print(f"Downloading evaluation datasets (version {DATASET_CACHE_VERSION})...")
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset",
        force_download=True,
    )
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset",
        force_download=True,
    )
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/extracted_emission_lines.csv",
        repo_type="dataset",
        force_download=True,
    )
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/extracted_types.csv",
        repo_type="dataset",
        force_download=True,
    )


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("huggingface_hub", "pyyaml", "tqdm", "matplotlib", "Pillow", "torchvision")
    .run_function(download_models)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)


@app.function(
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def generate_captions_remote(
    responder_config: dict,
    timestamp_dir: str,
    dataset_filter: str = "all",
    split: str = "legacy",
    limit: int = None,
    batch_size: int = 32,
    caption_prompt: str = None,
):
    """Remote function executing GPU caption generation on Modal."""
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import torch
    from tqdm import tqdm

    from evals.caption_responders import CaptionSample, get_caption_responder
    from evals.data import (
        load_test_spectra,
        load_test_spectra_by_category,
        load_test_spectra_emission_lines,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Modal Remote] Initializing {responder_config.get('responder_type')} on {device}...")
    responder = get_caption_responder(responder_config, device)

    print(f"[Modal Remote] Loading dataset filter '{dataset_filter}'...")
    print(f"[Modal Remote] Loading dataset filter '{dataset_filter}' (split='{split}')...")
    if dataset_filter == "emission_lines":
        df_test = load_test_spectra_emission_lines()
        df_test = load_test_spectra_emission_lines(split_version=split)
    elif dataset_filter == "source":
        df_test = load_test_spectra_by_category("class", ["GALAXY", "QSO"])
        df_test = load_test_spectra_by_category("class", ["GALAXY", "QSO"], split_version=split)
    elif dataset_filter == "subclass":
        df_test = load_test_spectra_by_category("subclass", ["AGN", "STARBURST", "STARFORMING", "BROADLINE"])
        df_test = load_test_spectra_by_category("subclass", ["AGN", "STARBURST", "STARFORMING", "BROADLINE"], split_version=split)
    else:
        df_test = load_test_spectra()
        df_test = load_test_spectra(split_version=split)

    if limit is not None:
        print(f"[Modal Remote] Limiting to first {limit} samples.")
        df_test = df_test.head(limit)

    total_expected = len(df_test)
    print(f"[Modal Remote] Starting caption generation for {total_expected} spectra (dataset_filter='{dataset_filter}')...")
    print(f"[Modal Remote] Starting caption generation for {total_expected} spectra (dataset_filter='{dataset_filter}', split='{split}')...")

    out_dir = os.path.join("/outputs", timestamp_dir)
    os.makedirs(out_dir, exist_ok=True)
    captions_file = os.path.join(out_dir, "captions.jsonl")

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    total = 0
    with open(captions_file, "w") as out_f:
        for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="[Modal] Generating Captions"):
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

            captions = responder.generate_captions(samples, prompt_override=caption_prompt)
            for c in captions:
                record = {
                    "wiki_entity_id": c.sample_id,
                    "survey": c.survey,
                    "caption": c.caption,
                    "responder_type": c.responder_type,
                    "model_id": c.model_id,
                    "caption_prompt": c.caption_prompt,
                }
                out_f.write(json.dumps(record) + "\n")
                total += 1

            volume.commit()

    if total != total_expected:
        print(f"[Modal Remote] WARNING: Generated {total} captions but expected {total_expected}!")
    else:
        print(f"[Modal Remote] Done. Successfully generated all {total}/{total_expected} captions in {captions_file}")
    return timestamp_dir


@app.local_entrypoint()
def main(
    responder: str,
    output: str = None,
    dataset_filter: str = "all",
    split: str = "legacy",
    limit: int = None,
    batch_size: int = None,
    caption_prompt: str = None,
    gpu: str = None,
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

    resp_tag = responder_config.get("responder_type", "model")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_captions")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{split}_captions")

    gpu_type = gpu or responder_config.get("gpu", "A100-80GB")
    print(f"Launching remote caption generation on Modal GPU ({gpu_type}) with batch size {effective_batch_size}...")
    print(f"Launching remote caption generation on Modal GPU ({gpu_type}) with batch size {effective_batch_size} (split={split})...")

    generate_captions_remote.with_options(gpu=gpu_type).remote(
        responder_config=responder_config,
        timestamp_dir=timestamp_dir,
        dataset_filter=dataset_filter,
        split=split,
        limit=limit,
        batch_size=effective_batch_size,
        caption_prompt=caption_prompt,
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
    print(f"      --split {split} \\")
    print(f"      --frontier eval_configs/frontier/gemini.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal script to generate captions.")
    parser.add_argument("--responder", type=str, required=True, help="Path to responder config.")
    parser.add_argument("--output", type=str, default=None, help="Local output path.")
    parser.add_argument("--dataset-filter", type=str, default="all", help="Dataset filter.")
    parser.add_argument("--split", type=str, default="legacy", choices=["legacy", "v7"], help="Dataset split: 'legacy' (default, 400 test) or 'v7' (641 val+test).")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Prompt override.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU type override.")
    args, _ = parser.parse_known_args()

    print(
        f"Please run with Modal CLI:\n"
        f"  modal run eval_scripts/generate_captions_modal.py "
        f"--responder {args.responder} "
        f"--split {args.split} "
        + (f"--output {args.output} " if args.output else "")
        + (f"--limit {args.limit} " if args.limit else "")
    )

