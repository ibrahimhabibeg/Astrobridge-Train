import os
import re
import json
import io
import modal
from datetime import datetime
import subprocess

volume = modal.Volume.from_name("astrobridge-evals", create_if_missing=True)
app = modal.App("astrobridge-evaluation-base")

def download_models():
    from huggingface_hub import snapshot_download, hf_hub_download
    
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
    .pip_install("matplotlib", "Pillow", "torchvision")
    .run_function(download_models)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

def render_spectrum_plot(wavelength, flux, mask=None, survey=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if mask is not None:
        valid = ~mask
        ax.plot(wavelength[valid], flux[valid], color='blue', lw=1, label='Valid')
        ax.plot(wavelength[mask], flux[mask], color='red', lw=1, alpha=0.5, label='Masked')
    else:
        ax.plot(wavelength, flux, color='blue', lw=1)
        
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title('Spectrum')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def build_mcq_prompt() -> str:
    prompt = """
    You are given a plot of an astronomical spectrum.
    Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories where z refers to the redshift of the object:
    A: very close (z < 0.1)
    B: close (0.1 <= z < 0.5)
    C: intermediate (0.5 <= z < 1.0)
    D: far (1.0 <= z < 2.0)
    E: very far (z >= 2.0)
    Your answer must be in the format 'FINAL ANSWER: [Letter]' where [Letter] is one of A, B, C, D, or E.
    """
    # prompt = (
    #     "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
    #     "A: very close (z < 0.1), B: close (0.1 <= z < 0.5), C: intermediate (0.5 <= z < 1.0), D: far (1.0 <= z < 2.0), E: very far (z >= 2.0). "
    #     "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
    # )
    return prompt

class EvalSample:
    def __init__(self, wavelength, flux, mask, survey):
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.survey = survey

class BaseEvaluator:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        import torch
        
        self._model_id = model_id
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self._model.eval()

    def predict_batch(self, samples: list[EvalSample], prompt: str) -> list[str]:
        from PIL import Image
        import torch
        
        messages_batch = []
        for sample in samples:
            png_bytes = render_spectrum_plot(
                sample.wavelength, sample.flux,
                mask=sample.mask, survey=sample.survey,
            )
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
            
        return self._generate_from_messages_batch(messages_batch)

    def _generate_from_messages_batch(self, messages_batch) -> list[str]:
        import torch
        
        texts = [self._processor.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False) for msgs in messages_batch]
                
        images = []
        for msgs in messages_batch:
            for msg in msgs:
                for content in msg.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        images.append(content["image"])
        
        self._processor.tokenizer.padding_side = 'left'
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token
            
        inputs = self._processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=2048)
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)


def get_correct_answer(z: float) -> str:
    if z < 0.1: return "A"
    elif 0.1 <= z < 0.5: return "B"
    elif 0.5 <= z < 1.0: return "C"
    elif 1.0 <= z < 2.0: return "D"
    else: return "E"

def extract_final_answer(response: str) -> str:
    match = re.search(r"FINAL ANSWER:\s*([A-E])", response)
    if match: return match.group(1)
    match = re.search(r"^\s*([A-E])\s*$", response.strip())
    if match: return match.group(1)
    return "UNKNOWN"

@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=86400,
    volumes={"/outputs": volume},
)
def run_evaluation(model_id: str, prompt: str, timestamp_dir: str, batch_size: int = 4):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = BaseEvaluator(model_id, device)

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

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Evaluating"):
        wiki_entity_ids = batch_df['wiki_entity_id'].tolist()
        z_values = batch_df['Z'].tolist()
        correct_answers = [get_correct_answer(z) for z in z_values]
        
        surveys = batch_df['survey'].tolist() if 'survey' in batch_df.columns else ["sdss"] * len(batch_df)
        
        samples = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            spec_data = row["spectrum"]
            flux = np.array(spec_data["flux"])
            wavelength = np.array(spec_data["lambda"])
            mask = np.array(spec_data["mask"]).astype(bool) if "mask" in spec_data else np.zeros_like(flux, dtype=bool)
            
            samples.append(EvalSample(wavelength, flux, mask, surveys[i]))
            
        try:
            answers = evaluator.predict_batch(samples, prompt)
            
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
    model_id = "Qwen/Qwen3.5-9B"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S_base")
    
    prompt = build_mcq_prompt()
    
    local_output_dir = os.path.join(os.getcwd(), "eval_results")
    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Starting evaluation. Results will be saved to timestamped dir: {timestamp_dir}")
    run_evaluation.remote(model_id, prompt, timestamp_dir, batch_size=16)
    
    print(f"\nEvaluation finished!")
    print(f"Syncing Modal volume to local directory: {local_output_dir}")

    subprocess.run(
        ["modal", "volume", "get", "astrobridge-evals", timestamp_dir, local_output_dir],
        check=True
    )
    
    results_dir = os.path.join(local_output_dir, timestamp_dir)
    print("Done syncing! Computing metrics locally to save GPU time...")
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

