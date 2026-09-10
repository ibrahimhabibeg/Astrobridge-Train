"""Generic HuggingFace text-only responder.

Works with any HF causal language model (e.g. Llama, Mistral, Gemma, Phi, etc.).
Uses AutoModelForCausalLM + AutoTokenizer instead of a model-specific class.
The spectrum is passed as subsampled text, identical to BaseQwenTextResponder.
"""

import numpy as np
import torch
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import EvalSample, ModelResponse
from .fallback import identify_failed_indices, merge_fallback_responses
from .base_qwen_text import subsample_spectrum, format_spectrum_text


class HFTextResponder:
    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], "Missing 'hf_model_id' in config"
        self._model_id = config["hf_model_id"]
        self._max_tokens = config.get("max_tokens", 2048)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 128)
        self._num_points = config.get("num_points", 100)
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_config(self) -> dict:
        return {
            "type": "HFTextResponder",
            "model_id": self._model_id,
            "num_points": self._num_points,
        }

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        messages_batch = []

        for sample in samples:
            w_str, f_str = subsample_spectrum(sample.wavelength, sample.flux, num_points=self._num_points)
            spectrum_text = format_spectrum_text(w_str, f_str, self._num_points)
            prompt = task.build_prompt(image_mode=False, spectrum_text=spectrum_text)

            messages = [
                {"role": "user", "content": prompt},
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
        raw_responses = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

            fb_inputs = self._tokenizer(fallback_texts, return_tensors="pt", padding=True)
            fb_inputs = {k: v.to(self._device) for k, v in fb_inputs.items()}

            with torch.no_grad():
                fb_output_ids = self._model.generate(
                    **fb_inputs, max_new_tokens=self._fallback_max_tokens, do_sample=False
                )

            fb_input_len = fb_inputs["input_ids"].shape[1]
            fb_generated = fb_output_ids[:, fb_input_len:]
            fb_raw = self._tokenizer.batch_decode(fb_generated, skip_special_tokens=True)

            merge_fallback_responses(responses, failed, fb_raw, fallback_tag, task)

        return responses

