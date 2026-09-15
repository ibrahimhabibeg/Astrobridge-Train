from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel

from .base import (
    BaseResponder,
    GeneratedCaption,
    resolve_caption_prompt,
)
from .sample import SpectrumSample
from captioner.utils.config import load_config
from captioner.encoders.registry import build_encoder
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.data.spectra_dataset import trimmed_spectrum_arrays
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.prompt import build_wrapper_text


class AstroBridgeResponder(BaseResponder):
    def __init__(self, config: dict, device: str):
        assert "astrobridge_id" in config and config["astrobridge_id"], (
            "Missing 'astrobridge_id' in config"
        )
        self.device = device
        self.repo_id = config["astrobridge_id"]
        self.caption_prompt = resolve_caption_prompt(
            config, "caption_generation/astrobridge.jinja2"
        )
        self.cfg = load_config("base", "data", "modalities", "model", "stage2")

        llm, self.tokenizer = build_llm(self.cfg)
        llm = PeftModel.from_pretrained(llm, self.repo_id)
        d_llm = get_llm_hidden_size(llm)

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
        middle_pt_path = hf_hub_download(repo_id=self.repo_id, filename="middle.pt")
        fusion_stack.load_state_dict(
            torch.load(middle_pt_path, map_location="cpu", weights_only=False)
        )

        self.model = Captioner(
            fusion_stack, llm, n_queries=int(self.cfg.qformer.n_queries)
        )
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
            "type": "AstroBridgeResponder",
            "repo_id": self.repo_id,
            "caption_prompt": self.caption_prompt,
            "max_tokens": self.generate_max_tokens,
        }

    def _generate_caption_batch(
        self,
        raw_inputs_list: List[Dict[str, Any]],
        questions: str,
        max_new_tokens: int = 512,
    ) -> List[str]:
        features_list = []
        for raw_inputs in raw_inputs_list:
            feat_dict = {}
            for name, enc in self.encoders.items():
                if name in raw_inputs and raw_inputs[name] is not None:
                    device_inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in raw_inputs[name].items()
                    }
                    feat_dict[name] = enc(device_inputs).to(self.device)
            features_list.append(feat_dict)

        bsz = len(features_list)
        fused_features = {}
        for name in self.cfg.modalities:
            mod_feats = [fl[name] for fl in features_list if name in fl]
            if len(mod_feats) == bsz:
                fused_features[name] = torch.cat(mod_feats, dim=0)

        pre, post = build_wrapper_text(self.tokenizer, questions)
        pre_ids = self.tokenizer.encode(
            pre, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        post_ids = self.tokenizer.encode(
            post, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        pre_ids = pre_ids.repeat(bsz, 1)
        post_ids = post_ids.repeat(bsz, 1)

        with torch.no_grad():
            output_ids = self.model.generate(
                fused_features,
                pre_ids,
                post_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def generate_captions(
        self,
        samples: List[SpectrumSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt

        raw_inputs_list = []
        for sample in samples:
            flux, wavelength, ivar, mask = trimmed_spectrum_arrays(
                {
                    "flux": sample.flux,
                    "lambda": sample.wavelength,
                    "ivar": sample.ivar
                    if sample.ivar is not None
                    else np.ones_like(sample.flux),
                    "mask": sample.mask
                    if sample.mask is not None
                    else np.zeros_like(sample.flux, dtype=bool),
                }
            )

            spectrum_dict = {
                "flux": torch.tensor(flux).float().unsqueeze(0),
                "wavelength": torch.tensor(wavelength).float().unsqueeze(0),
                "ivar": torch.tensor(ivar).float().unsqueeze(0),
                "mask": torch.tensor(mask).bool().unsqueeze(0),
                "survey": [sample.survey],
            }
            raw_inputs_list.append({"spectra": spectrum_dict})

        generated_texts = self._generate_caption_batch(
            raw_inputs_list,
            questions=prompt,
            max_new_tokens=self.generate_max_tokens,
        )

        captions = []
        for sample, text in zip(samples, generated_texts):
            clean_text = text.strip()
            captions.append(
                GeneratedCaption(
                    sample_id=sample.sample_id,
                    caption=clean_text,
                    responder_type="astrobridge",
                    model_id=self.repo_id,
                    caption_prompt=prompt,
                    survey=sample.survey,
                )
            )
        return captions
