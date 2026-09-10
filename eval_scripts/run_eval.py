import argparse
import json
import os
import subprocess
from datetime import datetime
import yaml
import dotenv
import modal
import numpy as np

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-evaluation")


def download_models():
    """Download required models and datasets on Modal boot."""
    from huggingface_hub import hf_hub_download, snapshot_download

    astrobridge_id = "UniverseTBD/astrobridge-model-v3_qwen"
    base_llm_id = "Qwen/Qwen3.5-9B"

    print(f"Downloading AstroBridge extra weights: {astrobridge_id}")
    snapshot_download(astrobridge_id)
    hf_hub_download(repo_id=astrobridge_id, filename="middle.pt")

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


def run_evaluation_core(config: dict, timestamp_dir: str, output_base_dir: str):
    """Core evaluation loop, runs either locally or on Modal."""
    import torch
    from tqdm import tqdm

    from evals.data import (
        load_test_spectra,
        load_test_spectra_by_category,
        load_test_spectra_emission_lines,
    )
    from evals.responders import EvalSample, get_responder
    from evals.tasks import get_task

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    output_dir = os.path.join(output_base_dir, timestamp_dir)
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
                # Remove None keys
                result = {k: v for k, v in result.items() if v is not None}
                batch_results.append(result)
                
                tqdm.write(f"[{wiki_entity_id}] Truth: {gt} | Pred: {resp.parsed}")

            with open(os.path.join(output_dir, "results.jsonl"), "a") as f:
                for res in batch_results:
                    f.write(json.dumps(res) + "\n")

        except Exception as e:
            tqdm.write(f"Error on batch starting with {wiki_entity_ids[0] if wiki_entity_ids else 'unknown'}: {e}")

        # Only commit if we're inside Modal and volume is bound to /outputs
        if output_base_dir == "/outputs":
            volume.commit()

    print("Evaluation complete.")
    return timestamp_dir


@app.function(
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def run_evaluation_remote(config: dict, timestamp_dir: str):
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    run_evaluation_core(config, timestamp_dir, "/outputs")
    return timestamp_dir


@app.local_entrypoint()
def main(config_path: str, limit: int = None, gemini_model: str = None, gpu: str = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if limit is not None:
        config["run"]["limit"] = limit
    if gemini_model is not None:
        config["run"]["gemini_model"] = gemini_model
    if gpu is not None:
        config["run"]["gpu"] = gpu

    run_config = config["run"]
    suffix = run_config.get("suffix_tag", "eval")
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")

    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Starting evaluation from {config_path}. Results -> {timestamp_dir}")

    if run_config["responder_type"] == "gemini":
        print("Running Gemini evaluator LOCALLY (bypassing Modal GPU).")
        dotenv.load_dotenv()
        run_evaluation_core(config, timestamp_dir, local_output_dir)
        results_dir = os.path.join(local_output_dir, timestamp_dir)
    else:
        gpu_type = run_config.get("gpu", "A100-80GB")
        print(f"Running model evaluator REMOTELY on Modal GPU ({gpu_type}).")
        run_evaluation_remote.with_options(gpu=gpu_type).remote(config, timestamp_dir)
        print("\nEvaluation finished!")
        print(f"Syncing Modal volume to local directory: {local_output_dir}")
        subprocess.run(
            ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, local_output_dir],
            check=True,
        )
        results_dir = os.path.join(local_output_dir, timestamp_dir)

    from evals.metrics import compute_and_save_metrics
    from evals.tasks import get_task

    print("Done generating results! Computing metrics locally...")
    task = get_task(config["task"]["name"], **config["task"].get("kwargs", {}))
    compute_and_save_metrics(results_dir, task)
    print(f"All done! Check {results_dir} for results and metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation script.")
    parser.add_argument("config", type=str, help="Path to evaluation YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model.")
    parser.add_argument("--gpu", type=str, default=None, help="Override GPU type (e.g. A100-80GB, A10G, H100).")
    args = parser.parse_args()

    # When run directly (not via modal run)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config["run"]["responder_type"] == "gemini":
        main(args.config, args.limit, args.gemini_model, args.gpu)
    else:
        print(f"Please use `modal run eval_scripts/run_eval.py --config-path {args.config}` to run on Modal GPUs.")

