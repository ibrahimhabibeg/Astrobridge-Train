"""Generic HuggingFace vision-language responder.

Works with any HF model that supports the image-text-to-text pipeline
(e.g. LLaVA, InternVL, Phi-Vision, Pixtral, Idefics, etc.).
Uses AutoModelForImageTextToText + AutoProcessor instead of a
model-specific class like Qwen3_5ForConditionalGeneration.
"""

import io
from typing import List, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from . import EvalSample, ModelResponse
from .fallback import identify_failed_indices, merge_fallback_responses
from .utils import render_spectrum_plot


class HFVisionResponder:
    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], "Missing 'hf_model_id' in config"
        self._model_id = config["hf_model_id"]
        self._max_tokens = config.get("max_tokens", 2048)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 128)
        self._device = device
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_config(self) -> dict:
        return {"type": "HFVisionResponder", "model_id": self._model_id}

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        prompt = task.build_prompt(image_mode=True)

        messages_batch = []
        images = []
        for sample in samples:
            png_bytes = render_spectrum_plot(sample.wavelength, sample.flux, mask=sample.mask)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(image)

            # Standard HF format: {"type": "image"} is a placeholder;
            # actual PIL images are passed separately to the processor.
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

        return self._generate_from_messages_batch(messages_batch, images, task)

    def _generate_from_messages_batch(self, messages_batch, images, task: Any) -> List[ModelResponse]:
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
        raw_responses = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        parsed_results = [task.default_parse(r) for r in raw_responses]

        responses = [
            ModelResponse(
                parsed=p if p is not None else [],
                raw_text=raw_responses[i],
                forced_fallback=False,
            )
            for i, p in enumerate(parsed_results)
        ]

        # Fallback pass
        fallback_tag = task.fallback_tag()
        failed = identify_failed_indices(responses)

        if failed and fallback_tag:
            fallback_texts = [texts[i] + raw_responses[i] + fallback_tag for i in failed]
            fallback_images = [images[i] for i in failed]

            fb_inputs = self._processor(
                text=fallback_texts, images=fallback_images, return_tensors="pt", padding=True
            )
            fb_inputs = {k: v.to(self._device) for k, v in fb_inputs.items()}

            with torch.no_grad():
                fb_output_ids = self._model.generate(
                    **fb_inputs, max_new_tokens=self._fallback_max_tokens, do_sample=False
                )

            fb_input_len = fb_inputs["input_ids"].shape[1]
            fb_generated = fb_output_ids[:, fb_input_len:]
            fb_raw = self._processor.batch_decode(fb_generated, skip_special_tokens=True)

            merge_fallback_responses(responses, failed, fb_raw, fallback_tag, task)

        return responses

