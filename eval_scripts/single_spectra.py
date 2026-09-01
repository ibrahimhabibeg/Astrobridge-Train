import os
import modal

app = modal.App("astrobridge-inference")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("test_subjects", remote_path="/root/test_subjects")
)

@app.function(
    gpu="A10G",
    image=image,
    timeout=600,
)
def run_inference(repo_id: str, spectrum_path: str, prompt: str, survey: str = "desi"):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
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

    print(f"Loading spectrum from {spectrum_path}...")
    npz = np.load(spectrum_path)
    flux = torch.from_numpy(npz["flux"]).unsqueeze(0).float()
        
    spectrum_batch = {
        "flux": flux,
        "wavelength": torch.from_numpy(npz["wavelength"]).unsqueeze(0).float(),
        "survey": [survey],
    }
    if "ivar" in npz:
        spectrum_batch["ivar"] = torch.from_numpy(npz["ivar"]).unsqueeze(0).float()
    if "mask" in npz:
        spectrum_batch["mask"] = torch.from_numpy(npz["mask"]).unsqueeze(0).bool()
        
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
        max_new_tokens=250,
        question=prompt,
    )
    
    return answer

@app.local_entrypoint()
def main():
    repo_id = "UniverseTBD/astrobridge-model-v2"
    spectrum_path = "test_subjects/spectrum_01.npz"
    survey = "desi"
    
    prompt = (
        "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
        "A: very close, B: close, C: intermediate, D: far, E: very far. "
        "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
    )

    print(f"Submitting simple inference job to Modal...")
    answer = run_inference.remote(repo_id, spectrum_path, prompt, survey)
    
    print("\n" + "="*50)
    print("MODEL ANSWER:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")

