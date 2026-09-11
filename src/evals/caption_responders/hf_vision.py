from __future__ import annotations

import io
import re
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


def clean_and_extract_caption(raw_text: str) -> str:
    """Extract clean, concise caption text, stripping thoughts or reasoning markers."""
    if not raw_text:
        return ""
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()

    # 1. Match explicit CAPTION: ...
    match = re.search(r"\bCAPTION:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # 2. Match **Final Caption:** or **Caption:**
    match = re.search(r"\*\*(?:Final\s+)?Caption:?\*\*\s*:?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # 3. Match Draft
    match = re.search(r"\*(?:Revised\s+)?Draft(?:\s*\d+)?:\*\s*(?:CAPTION:\s*)?(.*)", text, re.DOTALL | re.IGNORECASE)
    if match and len(match.group(1).strip()) > 20:
        text = match.group(1).strip()

    # 4. Match # Summary or In summary:
    match = re.search(r"(?:In summary|Summary|Conclusion):\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Stop before trailing scratchpad sections
    stop_patterns = [
        r"\n\s*(?:Note|Explanation|Justification|Alternative interpretation|\d+\.\s*Final Polish|\*Wait).*$",
    ]
    for pattern in stop_patterns:
        text = re.split(pattern, text, flags=re.DOTALL | re.IGNORECASE)[0].strip()

    # 5. If the text begins with a reasoning preamble, extract the final substantive paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1 and any(
        kw in paragraphs[0].lower()
        for kw in ["the user wants", "analyze the image", "initial observation", "1. analyze"]
    ):
        for p in reversed(paragraphs):
            if not p.startswith("*") and not p.startswith("#") and not p.startswith("**") and len(p.split()) >= 6:
                return p.strip()

    return text.strip()


class HFVisionCaptionResponder(BaseCaptionResponder):
    """Specialized HuggingFace vision responder dedicated solely to spectrum caption generation."""

    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], "Missing 'hf_model_id' in config"
        self._model_id = config["hf_model_id"]
        self.caption_prompt = config.get("caption_prompt", DEFAULT_SPECTRUM_CAPTION_PROMPT)
        self._max_tokens = config.get("max_tokens", 256)
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

        system_message = (
            "You are an expert astrophysicist. Output only a direct, concise scientific caption (2-4 sentences) "
            "characterizing the astronomical observation and its spectrum. "
            "Do NOT describe plot formatting, axes, labels, colors, or step-by-step reasoning."
        )

        messages_batch = []
        images = []
        for sample in samples:
            png_bytes = render_spectrum_plot(sample.wavelength, sample.flux, mask=sample.mask)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(image)

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            messages_batch.append(messages)

        texts = []
        for msgs in messages_batch:
            try:
                formatted = self._processor.apply_chat_template(
                    msgs, add_generation_prompt=True, enable_thinking=False
                )
            except (TypeError, Exception):
                formatted = self._processor.apply_chat_template(
                    msgs, add_generation_prompt=True
                )

            # Close any unclosed thinking tag
            if formatted.rstrip().endswith("<think>"):
                formatted = formatted.rstrip() + "\n</think>\n"

            # Prefill assistant response with CAPTION:
            if not formatted.rstrip().endswith("CAPTION:"):
                formatted = formatted.rstrip() + "\nCAPTION: "

            texts.append(formatted)

        self._processor.tokenizer.padding_side = "left"
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        inputs = self._processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                repetition_penalty=1.15,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_captions = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        if len(samples) != len(raw_captions):
            raise RuntimeError(
                f"Batch mismatch: received {len(samples)} samples but decoded {len(raw_captions)} captions."
            )

        return [
            GeneratedCaption(
                sample_id=sample.sample_id,
                caption=clean_and_extract_caption(caption),
                responder_type="hf_vision",
                model_id=self._model_id,
                caption_prompt=prompt,
                survey=sample.survey,
            )
            for sample, caption in zip(samples, raw_captions)
        ]


