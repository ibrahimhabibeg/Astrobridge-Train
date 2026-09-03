import os
import json
import modal
from datetime import datetime
import subprocess

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-evaluation")

def download_models():
    from huggingface_hub import snapshot_download, hf_hub_download
    
    repo_id = "UniverseTBD/astrobridge-model-v2"
    snapshot_download(repo_id)
    hf_hub_download(repo_id=repo_id, filename="middle.pt")
    
    llm_name = "Qwen/Qwen3.5-9B"
    snapshot_download(llm_name)
    
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset"
    )

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("huggingface_hub", "pyyaml", "tqdm")
    .run_function(download_models)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def run_evaluation(repo_id: str, timestamp_dir: str, batch_size: int = 4):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import torch
    from tqdm import tqdm
    
    from evals.buckets import three_bucket_scheme
    from evals.data import load_test_spectra
    from evals.responders import EvalSample
    from evals.responders.astrobridge import AstroBridgeResponder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheme = three_bucket_scheme()
    responder = AstroBridgeResponder(repo_id, device)
    
    df_test = load_test_spectra()
    
    output_dir = f"/outputs/{timestamp_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
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
                    "full_response": resp.raw_text
                }
                batch_results.append(result)
                tqdm.write(f"[{wiki_entity_id}] Correct: {correct} | Model: {resp.label}")
            
            with open(f"{output_dir}/results.jsonl", "a") as f:
                for res in batch_results:
                    f.write(json.dumps(res) + "\n")
                    
        except Exception as e:
            tqdm.write(f"Error on batch starting with {wiki_entity_ids[0] if wiki_entity_ids else 'unknown'}: {e}")
            
        volume.commit()
        
    print("Evaluation complete. Syncing volume...")
    return timestamp_dir

@app.local_entrypoint()
def main():
    repo_id = "UniverseTBD/astrobridge-captioner-v3"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Starting evaluation. Results will be saved to timestamped dir: {timestamp_dir}")
    run_evaluation.remote(repo_id, timestamp_dir, batch_size=256)
    
    print(f"\nEvaluation finished!")
    print(f"Syncing Modal volume to local directory: {local_output_dir}")

    subprocess.run(
        ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, local_output_dir],
        check=True
    )
    
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from evals.buckets import three_bucket_scheme
    from evals.metrics import compute_and_save_metrics
    
    results_dir = os.path.join(local_output_dir, timestamp_dir)
    print(f"Done syncing! Computing metrics locally to save GPU time...")
    compute_and_save_metrics(results_dir, three_bucket_scheme())
    print(f"All done! Check {results_dir} for results and metrics.")
