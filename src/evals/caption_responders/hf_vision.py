from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import (
    BaseCaptionResponder,
    CaptionSample,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
)
from ..responders.utils import render_spectrum_plot


class HFVisionCaptionResponder(BaseCaptionResponder):
    """Specialized HuggingFace vision responder dedicated solely to spectrum caption generation."""

    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], "Missing 'hf_model_id' in config"
        self._model_id = config["hf_model_id"]
        self.caption_prompt = config.get("caption_prompt", DEFAULT_SPECTRUM_CAPTION_PROMPT)
        self._max_tokens = config.get("max_tokens", 512)
        self._device = device

        print(f"Loading HF Vision Processor and Model ({self._model_id})...")
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "HFVisionCaptionResponder",
            "model_id": self._model_id,
            "caption_prompt": self.caption_prompt,
            "max_tokens": self._max_tokens,
        }

    def generate_captions(
        self,
        samples: List[CaptionSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt

        messages_batch = []
        images = []
        for sample in samples:
            png_bytes = render_spectrum_plot(sample.wavelength, sample.flux, mask=sample.mask)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            messages_batch.append(messages)

        texts = [
            self._processor.apply_chat_template(msgs, add_generation_prompt=True)
            for msgs in messages_batch
        ]

        self._processor.tokenizer.padding_side = "left"
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        inputs = self._processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_captions = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [
            GeneratedCaption(
                sample_id=sample.sample_id,
                caption=caption.strip(),
                responder_type="hf_vision",
                model_id=self._model_id,
                caption_prompt=prompt,
                survey=sample.survey,
            )
            for sample, caption in zip(samples, raw_captions)
        ]

