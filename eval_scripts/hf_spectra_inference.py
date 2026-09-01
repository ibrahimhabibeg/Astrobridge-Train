import os
import modal

app = modal.App("astrobridge-inference")

def download_models():
    import yaml
    from huggingface_hub import snapshot_download, hf_hub_download
    
    repo_id = "UniverseTBD/astrobridge-model-v2"
    snapshot_download(repo_id)
    hf_hub_download(repo_id=repo_id, filename="middle.pt")
    
    # with open("/root/configs/model.yaml", "r") as f:
    #     model_cfg = yaml.safe_load(f)
    
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
    .pip_install("huggingface_hub", "pyyaml")
    .run_function(download_models)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("test_subjects", remote_path="/root/test_subjects")
)

@app.function(
    gpu="A10G",
    image=image,
    timeout=600,
)
def run_inference(repo_id: str, wiki_entity_id: str, prompt: str, survey: str = "desi"):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    
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

    print(f"Loading spectrum from UniverseTBD/AstroBridge-Data (with wiki_entity_id {wiki_entity_id})...")
    parquet_path = hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    df = df[df['wiki_entity_id']==wiki_entity_id]
    row = df.iloc[0]
    spec_data = row["spectrum"]
    
    flux = torch.tensor(spec_data["flux"]).unsqueeze(0).float()
        
    spectrum_batch = {
        "flux": flux,
        "wavelength": torch.tensor(spec_data["lambda"]).unsqueeze(0).float(),
        "survey": [survey],
    }
    if "ivar" in spec_data:
        spectrum_batch["ivar"] = torch.tensor(spec_data["ivar"]).unsqueeze(0).float()
    if "mask" in spec_data:
        spectrum_batch["mask"] = torch.tensor(spec_data["mask"]).unsqueeze(0).bool()
        
    raw_inputs = {"spectra": spectrum_batch}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}
    
    print(f"Running inference...")
    answer = generate_caption(
        model, 
        tokenizer, 
        encoders, 
        out_dims, 
        max_tokens, 
        cfg.prompt.template, 
        device,
        raw_inputs, 
        question=prompt,
    )
    
    return answer

@app.local_entrypoint()
def main():
    repo_id = "UniverseTBD/astrobridge-model-v2"
    # row_index = 0
    wiki_entity_id = 'gmw_00011334'
    survey = "sdss"
    
    prompt = (
        "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
        "A: very close (z < 0.1), B: close (0.1 <= z < 0.5), C: intermediate (0.5 <= z < 1.0), D: far (1.0 <= z < 2.0), E: very far (z >= 2.0). "
        "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
    )

    print("Submitting simple inference job to Modal...")
    answer = run_inference.remote(repo_id, wiki_entity_id, prompt, survey)
    
    print("\n" + "="*50)
    print("MODEL ANSWER:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")

