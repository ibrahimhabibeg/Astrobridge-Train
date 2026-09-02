import os
import re
import json
import modal
from datetime import datetime
import subprocess

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-evaluation")

def download_models():
    import yaml
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

def get_correct_answer(z: float) -> str:
    if z < 0.1:
        return "A"
    elif 0.1 <= z < 0.5:
        return "B"
    elif 0.5 <= z < 1.0:
        return "C"
    elif 1.0 <= z < 2.0:
        return "D"
    else:
        return "E"

def extract_final_answer(response: str) -> str:
    match = re.search(r"FINAL ANSWER:\s*([A-E])", response)
    if match:
        return match.group(1)
    return "UNKNOWN"

def generate_caption_batch(
    model,
    tokenizer,
    encoders: dict,
    modality_out_dims: dict,
    modality_max_tokens: dict,
    prompt_template: str,
    device: str,
    raw_inputs_list: list[dict],
    max_new_tokens: int = 256,
    question: str = None,
) -> list[str]:
    import torch
    from captioner.utils.prompt import human_readable_subset
    
    with torch.no_grad():
        if not raw_inputs_list:
            raise ValueError("raw_inputs_list is empty")
    
        shown = frozenset(raw_inputs_list[0].keys())
        B = len(raw_inputs_list)
        
        modality_batch = {}
        for name, out_dim in modality_out_dims.items():
            T_m = modality_max_tokens[name]
            tokens = torch.zeros((B, T_m, out_dim), dtype=torch.float32, device=device)
            mask = torch.ones((B, T_m), dtype=torch.bool, device=device)
            
            for i, raw_inputs in enumerate(raw_inputs_list):
                if name in raw_inputs:
                    raw_tokens = encoders[name].encode(raw_inputs[name]).to(torch.float32)
                    n = min(raw_tokens.shape[1], T_m)
                    tokens[i, :n] = raw_tokens[0, :n].to(device)
                    mask[i, :n] = False
            modality_batch[name] = {"tokens": tokens, "mask": mask}
    
        prompt_text = question if question is not None else prompt_template.format(modalities=human_readable_subset(shown))
        prompt_ids = tokenizer([prompt_text] * B, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            prefix = model.fusion_stack(modality_batch)
            prompt_embeds = model.llm.get_input_embeddings()(prompt_ids)
            inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            gen = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        return tokenizer.batch_decode(gen, skip_special_tokens=True)

@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def run_evaluation(repo_id: str, prompt: str, timestamp_dir: str, batch_size: int = 4):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.utils.rnn as rnn
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from tqdm import tqdm
    
    from captioner.utils.config import load_config
    from captioner.inference import generate_caption
    from captioner.encoders.registry import build_encoder
    from captioner.model.captioner import Captioner, FusionStack
    from captioner.train.stage1 import build_llm, get_llm_hidden_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    print("Building base LLM...")
    llm, tokenizer = build_llm(cfg)
    print(f"Loading LoRA adapter from {repo_id}...")
    llm = PeftModel.from_pretrained(llm, repo_id)
    d_llm = get_llm_hidden_size(llm)
    print("Building FusionStack...")
    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
    )    
    print(f"Downloading middle.pt from {repo_id}...")
    middle_pt_path = hf_hub_download(repo_id=repo_id, filename="middle.pt")
    fusion_stack.load_state_dict(torch.load(middle_pt_path, map_location="cpu", weights_only=False))    
    print("Initializing full Captioner model and encoders...")
    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(device)
    model.eval()
    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in cfg.modalities}

    print("Loading dataset...")
    parquet_path = hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    
    print("Filtering and deduplicating data...")
    df_test = df[df['split'] == 'test']
    df_test = df_test.drop_duplicates(subset=['wiki_entity_id'])
    print(f"Found {len(df_test)} unique test samples.")
    
    output_dir = f"/outputs/{timestamp_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/prompt.txt", "w") as f:
        f.write(prompt)
        
    results = []
    
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Evaluating"):
        wiki_entity_ids = batch_df['wiki_entity_id'].tolist()
        z_values = batch_df['Z'].tolist()
        correct_answers = [get_correct_answer(z) for z in z_values]
        
        surveys = batch_df['survey'].tolist() if 'survey' in batch_df.columns else ["sdss"] * len(batch_df)
        
        raw_inputs_list = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            spec_data = row["spectrum"]
            f_tensor = torch.tensor(spec_data["flux"]).float().unsqueeze(0)
            wavelength = torch.tensor(spec_data["lambda"]).float().unsqueeze(0)
            
            spectrum_dict = {
                "flux": f_tensor,
                "wavelength": wavelength,
                "survey": [surveys[i]]
            }
            
            if "ivar" in spec_data:
                spectrum_dict["ivar"] = torch.tensor(spec_data["ivar"]).float().unsqueeze(0)
                
            if "mask" in spec_data:
                spectrum_dict["mask"] = torch.tensor(spec_data["mask"]).bool().unsqueeze(0)
            else:
                spectrum_dict["mask"] = torch.zeros_like(f_tensor, dtype=torch.bool)
                
            raw_inputs_list.append({"spectra": spectrum_dict})
        
        try:
            answers = generate_caption_batch(
                model, 
                tokenizer, 
                encoders, 
                out_dims, 
                max_tokens, 
                cfg.prompt.template, 
                device,
                raw_inputs_list, 
                question=prompt,
            )
            
            batch_results = []
            for wiki_entity_id, z_value, correct_answer, answer in zip(wiki_entity_ids, z_values, correct_answers, answers):
                final_answer = extract_final_answer(answer)
                
                result = {
                    "wiki_entity_id": wiki_entity_id,
                    "Z": z_value,
                    "correct_answer": correct_answer,
                    "model_answer": final_answer,
                    "full_response": answer
                }
                batch_results.append(result)
                tqdm.write(f"[{wiki_entity_id}] Correct: {correct_answer} | Model: {final_answer}")
            
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
    
    prompt = (
        "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
        "A: very close (z < 0.1), B: close (0.1 <= z < 0.5), C: intermediate (0.5 <= z < 1.0), D: far (1.0 <= z < 2.0), E: very far (z >= 2.0). "
        "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
    )
    
    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Starting evaluation. Results will be saved to timestamped dir: {timestamp_dir}")
    run_evaluation.remote(repo_id, prompt, timestamp_dir, batch_size=256)
    
    print(f"\nEvaluation finished!")
    print(f"Syncing Modal volume to local directory: {local_output_dir}")

    subprocess.run(
        ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, local_output_dir],
        check=True
    )
    
    results_dir = os.path.join(local_output_dir, timestamp_dir)
    print(f"Done syncing! Computing metrics locally to save GPU time...")
    compute_and_save_metrics(results_dir)
    print(f"All done! Check {results_dir} for results and metrics.")

def compute_and_save_metrics(results_dir: str):    
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"No results.jsonl found in {results_dir}")
        return
        
    y_true = []
    y_pred = []
    format_errors = 0
    total = 0
    
    label_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    
    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1
            true_label = data["correct_answer"]
            pred_label = data["model_answer"]
            
            if pred_label == "UNKNOWN" or pred_label not in label_map:
                format_errors += 1
                continue
                
            y_true.append(true_label)
            y_pred.append(pred_label)
            
    if not y_true and format_errors == 0:
        print("No valid predictions to compute metrics on.")
        return

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0
    
    mae = sum(abs(label_map[t] - label_map[p]) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    
    cm = {t: {p: 0 for p in "ABCDE"} for t in "ABCDE"}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
        
    per_class_metrics = {}
    for t in "ABCDE":
        row_total = sum(cm[t].values())
        recall = cm[t][t] / row_total if row_total > 0 else 0.0
        
        col_total = sum(cm[p][t] for p in "ABCDE")
        precision = cm[t][t] / col_total if col_total > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[t] = {
            "support": row_total,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / 5.0
    
    total_valid = sum(m["support"] for m in per_class_metrics.values())
    if total_valid > 0:
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_valid
    else:
        weighted_f1 = 0.0

    metrics = {
        "total_samples": total,
        "format_errors": format_errors,
        "format_error_rate": format_errors / total if total > 0 else 0.0,
        "global_accuracy": accuracy,
        "mean_absolute_error": mae,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics
    }
    
    print("\n=== Evaluation Metrics ===")
    print(f"Total Samples: {total}")
    print(f"Format Errors (UNKNOWN): {format_errors} ({(metrics['format_error_rate'])*100:.1f}%)")
    print(f"Global Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"Mean Absolute Error (Ordinal off-by-X): {mae:.3f} classes")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print("         Predicted")
    print("       A   B   C   D   E")
    for t in "ABCDE":
        row = [f"{cm[t][p]:3d}" for p in "ABCDE"]
        print(f"True {t} " + " ".join(row))
        
    print("\n--- Per-Class Metrics ---")
    for t in "ABCDE":
        m = per_class_metrics[t]
        print(f"{t}: Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f} (Support: {m['support']})")
            
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
