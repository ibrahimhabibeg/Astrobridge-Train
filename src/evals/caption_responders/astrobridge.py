from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel

from .base import (
    BaseCaptionResponder,
    CaptionSample,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
)
from captioner.utils.config import load_config
from captioner.encoders.registry import build_encoder
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.data.spectra_dataset import trimmed_spectrum_arrays
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.prompt import build_wrapper_text


class AstroBridgeCaptionResponder(BaseCaptionResponder):
    """Specialized AstroBridge responder dedicated solely to spectrum caption generation."""

    def __init__(self, config: dict, device: str):
        assert "astrobridge_id" in config and config["astrobridge_id"], "Missing 'astrobridge_id' in config"
        self.device = device
        self.repo_id = config["astrobridge_id"]
        self.caption_prompt = config.get("caption_prompt", DEFAULT_SPECTRUM_CAPTION_PROMPT)
        self.cfg = load_config("base", "data", "modalities", "model", "stage2")

        print("Building base LLM...")
        llm, self.tokenizer = build_llm(self.cfg)
        print(f"Loading LoRA adapter from {self.repo_id}...")
        llm = PeftModel.from_pretrained(llm, self.repo_id)
        d_llm = get_llm_hidden_size(llm)

        print("Building FusionStack...")
        self.out_dims = {n: int(c.out_dim) for n, c in self.cfg.modalities.items()}
        fusion_stack = FusionStack(
            modality_out_dims=self.out_dims,
            d_shared=int(self.cfg.d_shared),
            d_llm=d_llm,
            qformer_cfg=dict(self.cfg.qformer),
            projector_hidden_mult=int(self.cfg.projector.hidden_mult),
            projector_dropout=float(self.cfg.projector.dropout),
            adapter_target_norm=llm_embedding_norm(llm),
        )
        print(f"Downloading middle.pt from {self.repo_id}...")
        middle_pt_path = hf_hub_download(repo_id=self.repo_id, filename="middle.pt")
        fusion_stack.load_state_dict(torch.load(middle_pt_path, map_location="cpu", weights_only=False))

        print("Initializing Captioner model and encoders...")
        self.model = Captioner(fusion_stack, llm, n_queries=int(self.cfg.qformer.n_queries))
        self.model.to(self.device)
        self.model.eval()
        self.encoders = {
            name: build_encoder(name, self.cfg.modalities[name], device=self.device)
            for name in self.cfg.modalities
        }
        self.max_tokens = {n: int(c.max_tokens) for n, c in self.cfg.modalities.items()}
        self.generate_max_tokens = config.get("max_tokens", 512)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "AstroBridgeCaptionResponder",
            "repo_id": self.repo_id,
            "caption_prompt": self.caption_prompt,
            "max_tokens": self.generate_max_tokens,
        }

    def generate_captions(
        self,
        samples: List[CaptionSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt

        raw_inputs_list = []
        for sample in samples:
            # Clean wavelength grid: drop non-positive / non-finite sentinels (-1) and sort ascending
            flux, wavelength, ivar, mask = trimmed_spectrum_arrays({
                "flux": sample.flux,
                "lambda": sample.wavelength,
                "ivar": sample.ivar if sample.ivar is not None else np.ones_like(sample.flux),
                "mask": sample.mask if sample.mask is not None else np.zeros_like(sample.flux, dtype=bool),
            })

            spectrum_dict = {
                "flux": torch.tensor(flux).float().unsqueeze(0),
                "wavelength": torch.tensor(wavelength).float().unsqueeze(0),
                "ivar": torch.tensor(ivar).float().unsqueeze(0),
                "mask": torch.tensor(mask).bool().unsqueeze(0),
                "survey": [sample.survey],
            }

            raw_inputs_list.append({"spectra": spectrum_dict})

        raw_captions = self._generate_caption_batch(
            raw_inputs_list, questions=prompt, max_new_tokens=self.generate_max_tokens
        )

        return [
            GeneratedCaption(
                sample_id=sample.sample_id,
                caption=caption.strip(),
                responder_type="astrobridge",
                model_id=self.repo_id,
                caption_prompt=prompt,
                survey=sample.survey,
            )
            for sample, caption in zip(samples, raw_captions)
        ]

    def _generate_caption_batch(
        self,
        raw_inputs_list: list[dict],
        questions: list[str] | str | None = None,
        max_new_tokens: int = 512,
    ) -> list[str]:
        with torch.no_grad():
            if not raw_inputs_list:
                raise ValueError("raw_inputs_list is empty")

            B = len(raw_inputs_list)

            modality_batch = {}
            for name, out_dim in self.out_dims.items():
                T_m = self.max_tokens[name]
                tokens = torch.zeros((B, T_m, out_dim), dtype=torch.float32, device=self.device)
                mask = torch.ones((B, T_m), dtype=torch.bool, device=self.device)

                for i, raw_inputs in enumerate(raw_inputs_list):
                    if name in raw_inputs:
                        raw_tokens = self.encoders[name].encode(raw_inputs[name]).to(torch.float32)
                        n = min(raw_tokens.shape[1], T_m)
                        tokens[i, :n] = raw_tokens[0, :n].to(self.device)
                        mask[i, :n] = False
                modality_batch[name] = {"tokens": tokens, "mask": mask}

            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                prefix = self.model.fusion_stack(modality_batch)  # (B, n_queries, d_llm)

                system = self.cfg.prompt.system_variants[0]

                if isinstance(questions, str):
                    questions = [questions] * B
                elif questions is None:
                    questions = [self.cfg.prompt.instruction_variants[0]] * B

                embed_fn = self.model.llm.get_input_embeddings()
                embeds_list = []

                for i in range(B):
                    pre_text, post_text = build_wrapper_text(self.cfg.prompt, system, questions[i])

                    pre_ids = self.tokenizer(
                        pre_text, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].to(self.device)
                    post_ids = self.tokenizer(
                        post_text, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].to(self.device)

                    pre_embeds = embed_fn(pre_ids)
                    post_embeds = embed_fn(post_ids)

                    pref = prefix[i:i+1]
                    seq_embeds = torch.cat([pre_embeds, pref, post_embeds], dim=1)
                    embeds_list.append(seq_embeds.squeeze(0))

                max_len = max(emb.shape[0] for emb in embeds_list)
                d_llm = embeds_list[0].shape[-1]

                inputs_embeds = torch.zeros((B, max_len, d_llm), dtype=embeds_list[0].dtype, device=self.device)
                attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

                for i, emb in enumerate(embeds_list):
                    seq_len = emb.shape[0]
                    inputs_embeds[i, -seq_len:] = emb
                    attention_mask[i, -seq_len:] = 1

                pad_token_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )

                eos_ids = [self.tokenizer.eos_token_id]
                if self.tokenizer.convert_tokens_to_ids("<|im_end|>") not in eos_ids:
                    eos_ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if self.tokenizer.convert_tokens_to_ids("<|endoftext|>") not in eos_ids:
                    eos_ids.append(self.tokenizer.convert_tokens_to_ids("<|endoftext|>"))

                gen = self.model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_ids,
                )
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

