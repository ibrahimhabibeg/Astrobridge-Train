#!/usr/bin/env python
"""Diagnostic test script to verify whether AstroBridge v6 conditions on input spectra.

Tests:
1. Real Quasar Spectrum (gmw_00000967, z=3.288)
2. All-Zeros Spectrum (flux = 0)
3. Gaussian Noise Spectrum (flux ~ N(0, 1))
4. Constant Flat Spectrum (flux = 50.0)
5. Zeroed LLM Prefix Vector (prefix = 0 tensor)
6. Real Absorption Galaxy Spectrum (gmw_00000415)
7. All-Zeros Spectrum for Galaxy

Usage:
    modal run eval_scripts/test_zero_spectrum_modal.py
"""

import argparse
import os
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
    print(f"Downloading evaluation datasets (version {DATASET_CACHE_VERSION})...")
    hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
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
    gpu="A10",
    timeout=3600,
)
def run_zero_spectrum_diagnostics(model_id: str = "UniverseTBD/astrobridge-model-v6"):
    import sys
    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import numpy as np
    import torch
    import torch.nn.functional as F

    from evals.caption_responders.astrobridge import AstroBridgeCaptionResponder
    from evals.caption_responders.base import CaptionSample, DEFAULT_SPECTRUM_CAPTION_PROMPT
    from evals.data import load_test_spectra
    from captioner.utils.prompt import build_wrapper_text

    print(f"\n========================================================")
    print(f"Running Diagnostic Zero-Spectrum Test on {model_id}")
    print(f"========================================================\n")

    responder_config = {
        "responder_type": "astrobridge",
        "astrobridge_id": model_id,
        "caption_prompt": DEFAULT_SPECTRUM_CAPTION_PROMPT,
        "max_tokens": 256,
    }
    responder = AstroBridgeCaptionResponder(responder_config, device="cuda")

    df_test = load_test_spectra()

    # Select target samples: Quasar (gmw_00000967) and Galaxy (gmw_00000415)
    quasar_rows = df_test[df_test["wiki_entity_id"] == "gmw_00000967"]
    if len(quasar_rows) == 0:
        quasar_row = df_test.iloc[0]
        q_id = quasar_row["wiki_entity_id"]
    else:
        quasar_row = quasar_rows.iloc[0]
        q_id = "gmw_00000967"

    galaxy_rows = df_test[df_test["wiki_entity_id"] == "gmw_00000415"]
    if len(galaxy_rows) == 0:
        galaxy_row = df_test.iloc[1]
        g_id = galaxy_row["wiki_entity_id"]
    else:
        galaxy_row = galaxy_rows.iloc[0]
        g_id = "gmw_00000415"

    q_flux = np.array(quasar_row["spectrum"]["flux"], dtype=np.float32)
    q_lambda = np.array(quasar_row["spectrum"]["lambda"], dtype=np.float32)
    q_mask = np.array(quasar_row["spectrum"].get("mask", np.zeros_like(q_flux, dtype=bool))).astype(bool)
    q_ivar = np.array(quasar_row["spectrum"].get("ivar", np.ones_like(q_flux, dtype=np.float32)))

    g_flux = np.array(galaxy_row["spectrum"]["flux"], dtype=np.float32)
    g_lambda = np.array(galaxy_row["spectrum"]["lambda"], dtype=np.float32)
    g_mask = np.array(galaxy_row["spectrum"].get("mask", np.zeros_like(g_flux, dtype=bool))).astype(bool)
    g_ivar = np.array(galaxy_row["spectrum"].get("ivar", np.ones_like(g_flux, dtype=np.float32)))

    conditions = [
        {
            "name": f"1. Real Quasar Spectrum ({q_id}, z=3.288)",
            "flux": q_flux,
            "lambda": q_lambda,
            "mask": q_mask,
            "ivar": q_ivar,
            "force_zero_prefix": False,
        },
        {
            "name": f"2. All-Zeros Flux (same grid as Quasar)",
            "flux": np.zeros_like(q_flux),
            "lambda": q_lambda,
            "mask": q_mask,
            "ivar": q_ivar,
            "force_zero_prefix": False,
        },
        {
            "name": f"3. Pure Gaussian Random Noise Flux N(0, 1)",
            "flux": np.random.normal(0, 1, size=q_flux.shape).astype(np.float32),
            "lambda": q_lambda,
            "mask": q_mask,
            "ivar": q_ivar,
            "force_zero_prefix": False,
        },
        {
            "name": f"4. Constant Flat Flux = 50.0",
            "flux": np.ones_like(q_flux) * 50.0,
            "lambda": q_lambda,
            "mask": q_mask,
            "ivar": q_ivar,
            "force_zero_prefix": False,
        },
        {
            "name": f"5. Zeroed LLM Prefix Vector (prefix = 0 tensor)",
            "flux": q_flux,
            "lambda": q_lambda,
            "mask": q_mask,
            "ivar": q_ivar,
            "force_zero_prefix": True,
        },
        {
            "name": f"6. Real Galaxy Spectrum ({g_id}, Absorption)",
            "flux": g_flux,
            "lambda": g_lambda,
            "mask": g_mask,
            "ivar": g_ivar,
            "force_zero_prefix": False,
        },
    ]

    print("\n--- Running Encoder & FusionStack Diagnostics ---")
    prefix_tensors = []

    with torch.no_grad():
        for cond in conditions:
            f_tensor = torch.tensor(cond["flux"]).float().unsqueeze(0)
            w_tensor = torch.tensor(cond["lambda"]).float().unsqueeze(0)
            spec_dict = {
                "flux": f_tensor,
                "wavelength": w_tensor,
                "survey": ["sdss"],
                "ivar": torch.tensor(cond["ivar"]).float().unsqueeze(0),
                "mask": torch.tensor(cond["mask"]).bool().unsqueeze(0),
            }
            raw_input = {"spectra": spec_dict}

            modality_batch = {}
            for name, out_dim in responder.out_dims.items():
                T_m = responder.max_tokens[name]
                tokens = torch.zeros((1, T_m, out_dim), dtype=torch.float32, device=responder.device)
                mask = torch.ones((1, T_m), dtype=torch.bool, device=responder.device)
                if name in raw_input:
                    raw_tokens = responder.encoders[name].encode(raw_input[name]).to(torch.float32)
                    n = min(raw_tokens.shape[1], T_m)
                    tokens[0, :n] = raw_tokens[0, :n].to(responder.device)
                    mask[0, :n] = False
                modality_batch[name] = {"tokens": tokens, "mask": mask}

            device_type = "cuda" if str(responder.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                prefix = responder.model.fusion_stack(modality_batch)  # (1, 64, d_llm)
                if cond["force_zero_prefix"]:
                    prefix = torch.zeros_like(prefix)
                prefix_tensors.append(prefix)

        p_quasar = prefix_tensors[0]
        p_zero = prefix_tensors[1]
        p_noise = prefix_tensors[2]
        p_galaxy = prefix_tensors[5]

        def cos_sim(t1, t2):
            return F.cosine_similarity(t1.flatten(), t2.flatten(), dim=0).item()

        print(f"Prefix Norm Quasar: {p_quasar.norm().item():.4f}")
        print(f"Prefix Norm Zeros:  {p_zero.norm().item():.4f}")
        print(f"Prefix Norm Noise:  {p_noise.norm().item():.4f}")
        print(f"Prefix Norm Galaxy: {p_galaxy.norm().item():.4f}")
        print(f"Cosine Sim (Quasar vs Galaxy): {cos_sim(p_quasar, p_galaxy):.4f}")
        print(f"Cosine Sim (Quasar vs Zeros):  {cos_sim(p_quasar, p_zero):.4f}")
        print(f"Cosine Sim (Quasar vs Noise):  {cos_sim(p_quasar, p_noise):.4f}")
        print("-" * 60)

        print("\n--- Generating Captions for Each Condition ---\n")
        results = []

        system = responder.cfg.prompt.system_variants[0]
        question = responder.caption_prompt
        embed_fn = responder.model.llm.get_input_embeddings()

        pre_text, post_text = build_wrapper_text(responder.cfg.prompt, system, question)
        pre_ids = responder.tokenizer(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(responder.device)
        post_ids = responder.tokenizer(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(responder.device)
        pre_embeds = embed_fn(pre_ids)
        post_embeds = embed_fn(post_ids)

        pad_token_id = responder.tokenizer.pad_token_id or responder.tokenizer.eos_token_id
        eos_ids = [responder.tokenizer.eos_token_id]
        for special_tok in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = responder.tokenizer.convert_tokens_to_ids(special_tok)
            if tok_id not in eos_ids:
                eos_ids.append(tok_id)

        for i, cond in enumerate(conditions):
            pref = prefix_tensors[i]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                seq_embeds = torch.cat([pre_embeds, pref, post_embeds], dim=1)
                attention_mask = torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=responder.device)

                gen = responder.model.llm.generate(
                    inputs_embeds=seq_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_ids,
                )
                decoded = responder.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

            results.append({
                "condition": cond["name"],
                "caption": decoded,
            })

            print(f"==================================================")
            print(f"Condition: {cond['name']}")
            print(f"Caption:\n{decoded}")
            print(f"==================================================\n")

    return results


@app.local_entrypoint()
def main(model: str = "UniverseTBD/astrobridge-model-v6"):
    print(f"Triggering diagnostic test on Modal for {model}...")
    run_zero_spectrum_diagnostics.remote(model_id=model)


if __name__ == "__main__":
    print("Run with: modal run eval_scripts/test_zero_spectrum_modal.py")
