from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import (
    BaseCaptionResponder,
    CaptionSample,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
)
from ..responders.utils import subsample_spectrum, format_spectrum_text


class HFTextCaptionResponder(BaseCaptionResponder):
    """Specialized HuggingFace text-only responder dedicated solely to spectrum caption generation."""

    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], "Missing 'hf_model_id' in config"
        self._model_id = config["hf_model_id"]
        self.caption_prompt = config.get("caption_prompt", DEFAULT_SPECTRUM_CAPTION_PROMPT)
        self._num_points = config.get("num_points", 100)
        self._max_tokens = config.get("max_tokens", 512)
        self._device = device

        print(f"Loading HF Text Tokenizer and Model ({self._model_id})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "HFTextCaptionResponder",
            "model_id": self._model_id,
            "caption_prompt": self.caption_prompt,
            "num_points": self._num_points,
            "max_tokens": self._max_tokens,
        }

    def generate_captions(
        self,
        samples: List[CaptionSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt

        messages_batch = []
        for sample in samples:
            w_str, f_str = subsample_spectrum(sample.wavelength, sample.flux, num_points=self._num_points)
            spectrum_text = format_spectrum_text(w_str, f_str, self._num_points)
            full_user_content = f"{prompt}\n\n{spectrum_text}"

            messages = [
                {"role": "user", "content": full_user_content},
            ]
            messages_batch.append(messages)

        texts = [
            self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]

        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs = self._tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_captions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return [
            GeneratedCaption(
                sample_id=sample.sample_id,
                caption=caption.strip(),
                responder_type="hf_text",
                model_id=self._model_id,
                caption_prompt=prompt,
                survey=sample.survey,
            )
            for sample, caption in zip(samples, raw_captions)
        ]

