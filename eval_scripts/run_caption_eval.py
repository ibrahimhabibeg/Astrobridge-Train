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
import subprocess
from datetime import datetime
import dotenv
import yaml
import numpy as np

# Modal setup
try:
    import modal
    volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
    app = modal.App("astrobridge-caption-evaluation")

    def download_models():
        from huggingface_hub import hf_hub_download, snapshot_download

        astrobridge_id = "UniverseTBD/astrobridge-model-v5"
        base_llm_id = "Qwen/Qwen3.5-9B"

        print(f"Downloading AstroBridge extra weights: {astrobridge_id}")
        try:
            snapshot_download(astrobridge_id)
            hf_hub_download(repo_id=astrobridge_id, filename="middle.pt")
        except Exception as e:
            print(f"Notice: {e}")

        print(f"Downloading Base LLM: {base_llm_id}")
        snapshot_download(base_llm_id)

        print("Downloading evaluation datasets...")
        hf_hub_download(
            repo_id="UniverseTBD/AstroBridge-Data",
            filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="UniverseTBD/AstroBridge-Data",
            filename="observations/spectra/extracted_emission_lines.csv",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="UniverseTBD/AstroBridge-Data",
            filename="observations/spectra/extracted_types.csv",
            repo_type="dataset",
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
    def generate_captions_remote(responder_config: dict, timestamp_dir: str, limit: int = None, caption_prompt: str = None):
        import sys
        sys.path.insert(0, "/root/src")
        os.chdir("/root")

        import torch
        from evals.caption_responders import CaptionSample, get_caption_responder
        from evals.data import load_test_spectra
        from tqdm import tqdm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        responder = get_caption_responder(responder_config, device)
        df_test = load_test_spectra()

        if limit is not None:
            df_test = df_test.head(limit)

        out_dir = os.path.join("/outputs", timestamp_dir)
        os.makedirs(out_dir, exist_ok=True)
        captions_path = os.path.join(out_dir, "captions.jsonl")

        batch_size = responder_config.get("batch_size", 32)
        def chunker(seq, size):
            return (seq[pos : pos + size] for pos in range(0, len(seq), size))

        with open(captions_path, "w") as out_f:
            for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Remote Caption Generation"):
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

        volume.commit()
        return timestamp_dir

except ImportError:
    app = None


def main_local(task: str, frontier: str, responder: str = None, captions: str = None, limit: int = None, caption_prompt: str = None, gemini_model: str = None, gpu: str = None):
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

    resp_tag = responder_config.get("responder_type") if responder_config else "cached"
    task_name = task_config.get("name", "task")
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
            )
            return
        else:
            gpu_type = gpu or responder_config.get("gpu", "A100-80GB")
            print(f"Running Caption Generation REMOTELY on Modal GPU ({gpu_type})...")
            generate_captions_remote.with_options(gpu=gpu_type).remote(
                responder_config, timestamp_dir, limit=limit, caption_prompt=caption_prompt
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified caption evaluation script.")
    parser.add_argument("--task", type=str, required=True, help="Path to task YAML config.")
    parser.add_argument("--frontier", type=str, required=True, help="Path to frontier model YAML config.")
    parser.add_argument("--responder", type=str, default=None, help="Path to responder YAML config.")
    parser.add_argument("--captions", type=str, default=None, help="Path to captions.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
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
        caption_prompt=args.caption_prompt,
        gemini_model=args.gemini_model,
        gpu=args.gpu,
    )

