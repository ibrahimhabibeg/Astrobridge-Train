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
    "responder_type": "gemini",                         # 'astrobridge', 'base_qwen', 'base_qwen_text', or 'gemini'
    "astrobridge_id": "UniverseTBD/astrobridge-captioner-v3", # The fine-tuned checkpoint
    "base_llm_id": "Qwen/Qwen3.5-9B",                   # The base language model
    "gemini_model": "gemini-3.7-flash",
    "gemini_num_workers": 4, 
    "gemini_max_output_tokens": 4096,
    "gemini_temperature": 1.0,                          # Set to 0.0 for greedy decoding (deterministic)
    "bucket_scheme": "3-group",                         # '5-group' or '3-group'
    "batch_size": 16,
    "gpu": "A100-80GB",                                 # 'A100-80GB' (for astrobridge) or 'A10' (for base)
    "suffix_tag": None,                                 # Leave as None to auto-generate (responder_scheme)
}
# ==========================================

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-evaluation")

def download_models():
    from huggingface_hub import snapshot_download, hf_hub_download
    
    astrobridge_id = "UniverseTBD/astrobridge-captioner-v3"
    base_llm_id = "Qwen/Qwen3.5-9B"
    
    print(f"Downloading AstroBridge extra weights: {astrobridge_id}")
    snapshot_download(astrobridge_id)
    hf_hub_download(repo_id=astrobridge_id, filename="middle.pt")
    
    print(f"Downloading Base LLM: {base_llm_id}")
    snapshot_download(base_llm_id)
        
    # Dataset
    print("Downloading evaluation dataset...")
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
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
    
    from evals.buckets import get_bucket_scheme
    from evals.data import load_test_spectra
    from evals.responders import EvalSample, get_responder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheme = get_bucket_scheme(RUN_CONFIG["bucket_scheme"])
    
    responder = get_responder(RUN_CONFIG, device)
    
    df_test = load_test_spectra()
    df_test = df_test.head(5)
    batch_size = RUN_CONFIG["batch_size"]
    
    output_dir = os.path.join(output_base_dir, timestamp_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "run_config": RUN_CONFIG,
        "responder": responder.get_config(),
        "bucket_scheme": scheme.get_config(),
        "batch_size": batch_size
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Evaluating"):
        wiki_entity_ids = batch_df['wiki_entity_id'].tolist()
        z_values = batch_df['Z'].tolist()
        correct_answers = [scheme.classify(z) for z in z_values]
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
            answers = responder.respond_batch(samples, scheme)
            batch_results = []
            
            for wiki_entity_id, z_value, correct, resp in zip(wiki_entity_ids, z_values, correct_answers, answers):
                result = {
                    "wiki_entity_id": wiki_entity_id,
                    "Z": z_value,
                    "correct_answer": correct,
                    "model_answer": resp.label,
                    "full_response": resp.raw_text,
                    "forced_fallback": getattr(resp, 'forced_fallback', False),
                    "responder": metadata["responder"]["type"],
                    "scheme": metadata["bucket_scheme"]["name"]
                }
                batch_results.append(result)
                tqdm.write(f"[{wiki_entity_id}] Correct: {correct} | Model: {resp.label}")
            
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
        suffix = f"{RUN_CONFIG['responder_type']}_{RUN_CONFIG['bucket_scheme']}"
        
    suffix_str = f"_{suffix}" if suffix else ""
    timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S{suffix_str}")
    
    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)
    
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))

    print(f"Starting evaluation. Results will be saved to timestamped dir: {timestamp_dir}")
    
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
    
    from evals.buckets import get_bucket_scheme
    from evals.metrics import compute_and_save_metrics
    
    print("Done generating results! Computing metrics locally...")
    compute_and_save_metrics(results_dir, get_bucket_scheme(RUN_CONFIG["bucket_scheme"]))
    print(f"All done! Check {results_dir} for results and metrics.")

# If someone runs `python eval_spectra.py` directly without `modal run`
if __name__ == "__main__":
    if RUN_CONFIG["responder_type"] == "gemini":
        main()
    else:
        print("Please use `modal run eval_scripts/eval_spectra.py` to run on Modal GPUs.")
