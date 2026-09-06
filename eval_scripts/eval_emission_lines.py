import os
import json
import modal
from datetime import datetime
import subprocess
import dotenv

# ==========================================
# EVALUATION CONFIGURATION
# Edit these values before running the script
# ==========================================
RUN_CONFIG = {
    "responder_type": "base_qwen_text",                         # 'astrobridge', 'base_qwen', 'base_qwen_text', or 'gemini'
    "astrobridge_id": "UniverseTBD/astrobridge-model-v3_qwen", # The fine-tuned checkpoint
    "base_llm_id": "Qwen/Qwen3.5-9B",                   # The base language model
    "max_tokens": 4096,
    "fallback_max_tokens": 128,
    "gemini_model": "gemini-3.7-flash",
    "gemini_num_workers": 4, 
    "gemini_temperature": 1.0,                          # Set to 0.0 for greedy decoding (deterministic)
    "batch_size": 1,
    "gpu": "A10",                                 # 'A100-80GB' (for astrobridge) or 'A10' (for base)
    "limit": 5,                                      # Set to an int (e.g. 5) to test on a subset, or None for full test set
    "suffix_tag": None,                                 # Leave as None to auto-generate (responder_lines)
}
# ==========================================

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-emission-lines-evaluation")

def download_models():
    from huggingface_hub import snapshot_download, hf_hub_download
    
    astrobridge_id = "UniverseTBD/astrobridge-model-v3_qwen"
    base_llm_id = "Qwen/Qwen3.5-9B"
    
    print(f"Downloading AstroBridge extra weights: {astrobridge_id}")
    snapshot_download(astrobridge_id)
    hf_hub_download(repo_id=astrobridge_id, filename="middle.pt")
    
    print(f"Downloading Base LLM: {base_llm_id}")
    snapshot_download(base_llm_id)
        
    # Dataset
    print("Downloading evaluation datasets...")
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset"
    )
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/extracted_emission_lines.csv",
        repo_type="dataset"
    )

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("huggingface_hub", "pyyaml", "tqdm", "matplotlib", "Pillow", "torchvision")
    .run_function(download_models)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

def run_evaluation_core(timestamp_dir: str, output_base_dir: str):
    import numpy as np
    import torch
    from tqdm import tqdm
    
    from evals.tasks import get_task
    from evals.data import load_test_spectra_emission_lines
    from evals.responders import EvalSample, get_responder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = get_task("emission_lines")
    responder = get_responder(RUN_CONFIG, device)
    
    df_test = load_test_spectra_emission_lines()
    if RUN_CONFIG.get("limit") is not None:
        print(f"Limiting evaluation to first {RUN_CONFIG['limit']} samples.")
        df_test = df_test.head(RUN_CONFIG["limit"])
    batch_size = RUN_CONFIG["batch_size"]
    
    output_dir = os.path.join(output_base_dir, timestamp_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "run_config": RUN_CONFIG,
        "responder": responder.get_config(),
        "task": task.get_config(),
        "batch_size": batch_size
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Evaluating Emission Lines"):
        wiki_entity_ids = batch_df['wiki_entity_id'].tolist()
        ground_truths = [task.extract_ground_truth(row) for _, row in batch_df.iterrows()]
        surveys = batch_df['survey'].tolist() if 'survey' in batch_df.columns else ["sdss"] * len(batch_df)
        
        samples = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            spec_data = row["spectrum"]
            flux = np.array(spec_data["flux"])
            wavelength = np.array(spec_data["lambda"])
            mask = np.array(spec_data["mask"]).astype(bool) if "mask" in spec_data else np.zeros_like(flux, dtype=bool)
            ivar = np.array(spec_data["ivar"]) if "ivar" in spec_data else None
            samples.append(EvalSample(wavelength, flux, mask, surveys[i], ivar=ivar))
            
        try:
            answers = responder.respond_batch(samples, task)
            batch_results = []
            
            for wiki_entity_id, gt, resp in zip(wiki_entity_ids, ground_truths, answers):
                result = {
                    "wiki_entity_id": wiki_entity_id,
                    "ground_truth_lines": gt,
                    "predicted_lines": resp.parsed,
                    "raw_response": resp.raw_text,
                    "forced_fallback": getattr(resp, 'forced_fallback', False),
                    "responder": metadata["responder"]["type"],
                    "task": metadata["task"]["task_name"]
                }
                batch_results.append(result)
                gt_keys = list(gt.keys())
                tqdm.write(f"[{wiki_entity_id}] Truth: {gt_keys} | Pred: {resp.parsed}")
            
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
    gpu=RUN_CONFIG["gpu"],
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def run_evaluation_remote(timestamp_dir: str):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    run_evaluation_core(timestamp_dir, "/outputs")
    return timestamp_dir

@app.local_entrypoint()
def main():
    suffix = RUN_CONFIG.get("suffix_tag")
    if not suffix:
        suffix = f"{RUN_CONFIG['responder_type']}_lines"
        
    suffix_str = f"_{suffix}" if suffix else ""
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S{suffix_str}")
    
    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)
    
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))

    print(f"Starting emission line evaluation. Results will be saved to timestamped dir: {timestamp_dir}")
    
    if RUN_CONFIG["responder_type"] == "gemini":
        print("Running Gemini evaluator LOCALLY (bypassing Modal GPU).")
        dotenv.load_dotenv()
        run_evaluation_core(timestamp_dir, local_output_dir)
        results_dir = os.path.join(local_output_dir, timestamp_dir)
    else:
        print("Running model evaluator REMOTELY on Modal GPU.")
        run_evaluation_remote.remote(timestamp_dir)
        print("\nEvaluation finished!")
        print(f"Syncing Modal volume to local directory: {local_output_dir}")
        subprocess.run(
            ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, local_output_dir],
            check=True
        )
        results_dir = os.path.join(local_output_dir, timestamp_dir)
    
    from evals.tasks import get_task
    from evals.metrics import compute_and_save_metrics
    
    print("Done generating results! Computing metrics locally...")
    task = get_task("emission_lines")
    compute_and_save_metrics(results_dir, task)
    print(f"All done! Check {results_dir} for results and metrics.")

if __name__ == "__main__":
    if RUN_CONFIG["responder_type"] == "gemini":
        main()
    else:
        print("Please use `modal run eval_scripts/eval_emission_lines.py` to run on Modal GPUs.")

