import numpy as np
import torch
from typing import List, Any
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from . import EvalSample, ModelResponse
from .fallback import identify_failed_indices, merge_fallback_responses


def subsample_spectrum(wavelength, flux, num_points=100):
    """Downsample a spectrum to evenly-spaced points and format as strings."""
    wavelength = np.array(wavelength).flatten()
    flux = np.array(flux).flatten()
    indices = np.linspace(0, len(wavelength) - 1, num_points, dtype=int)
    w_sub = wavelength[indices]
    f_sub = flux[indices]
    w_str = ", ".join([f"{w:.1f}" for w in w_sub])
    f_str = ", ".join([f"{f:.3f}" for f in f_sub])
    return w_str, f_str


def format_spectrum_text(w_str: str, f_str: str, num_points: int) -> str:
    """Format subsampled spectrum data as a text block for prompts."""
    return (
        f"Spectrum Data ({num_points} evenly spaced points):\n"
        f"Wavelength (Å): [{w_str}]\n"
        f"Flux: [{f_str}]"
    )


class BaseQwenTextResponder:
    def __init__(self, config: dict, device: str):
        assert "base_llm_id" in config and config["base_llm_id"], "Missing 'base_llm_id' in config"
        self._model_id = config["base_llm_id"]
        self._max_tokens = config.get("max_tokens", 2048)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 128)
        self._num_points = config.get("num_points", 100)
        self._device = device
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_config(self) -> dict:
        return {
            "type": "BaseQwenTextResponder",
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
                {"role": "assistant", "content": "Brief analysis:"},
            ]
            messages_batch.append(messages)

        texts = [
            self._processor.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
            for msgs in messages_batch
        ]

        self._processor.tokenizer.padding_side = "left"
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        inputs = self._processor.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_responses = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

            fb_inputs = self._processor.tokenizer(fallback_texts, return_tensors="pt", padding=True)
            fb_inputs = {k: v.to(self._device) for k, v in fb_inputs.items()}

            with torch.no_grad():
                fb_output_ids = self._model.generate(
                    **fb_inputs, max_new_tokens=self._fallback_max_tokens, do_sample=False
                )

            fb_input_len = fb_inputs["input_ids"].shape[1]
            fb_generated = fb_output_ids[:, fb_input_len:]
            fb_raw = self._processor.tokenizer.batch_decode(fb_generated, skip_special_tokens=True)

            merge_fallback_responses(responses, failed, fb_raw, fallback_tag, task)

        return responses
