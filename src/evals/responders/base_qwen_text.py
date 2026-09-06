import re
import torch
import numpy as np
from typing import List, Any
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from . import EvalSample, ModelResponse
from ..tasks.distance_classification import DistanceClassPromptSpec, DistanceClassificationTask
from ..tasks.emission_lines import EmissionLinePromptSpec

def subsample_spectrum(wavelength, flux, num_points=100):
    wavelength = np.array(wavelength).flatten()
    flux = np.array(flux).flatten()
    
    indices = np.linspace(0, len(wavelength) - 1, num_points, dtype=int)
    w_sub = wavelength[indices]
    f_sub = flux[indices]
    
    w_str = ", ".join([f"{w:.1f}" for w in w_sub])
    f_str = ", ".join([f"{f:.3f}" for f in f_sub])
    return w_str, f_str

class BaseQwenTextResponder:
    def __init__(self, config: dict, device: str):
        assert "base_llm_id" in config and config["base_llm_id"], "Missing 'base_llm_id' in config"
        assert "max_tokens" in config, "Missing 'max_tokens' in config"
        assert "fallback_max_tokens" in config, "Missing 'fallback_max_tokens' in config"
        self._model_id = config["base_llm_id"]
        self._max_tokens = config["max_tokens"]
        self._fallback_max_tokens = config["fallback_max_tokens"]
        self._device = device
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self._model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self._model.eval()

    def get_config(self) -> dict:
        return {
            "type": "BaseQwenTextResponder",
            "model_id": self._model_id,
            "num_points": 100
        }

    def _build_distance_prompt(self, spec: DistanceClassPromptSpec, w_str: str, f_str: str) -> str:
        return (
            "Classify the redshift (z) of the following astronomical spectrum.\n\n"
            f"Categories:\n{spec.options_multiline}\n\n"
            "Spectrum Data (100 evenly spaced points):\n"
            f"Wavelength (Å): [{w_str}]\n"
            f"Flux: [{f_str}]\n\n"
            "Provide exactly ONE sentence of analysis, then on a new line write 'FINAL ANSWER: <label>'."
        )

    def _build_emission_prompt(self, spec: EmissionLinePromptSpec, w_str: str, f_str: str) -> str:
        return (
            "Briefly analyze and describe the following astronomical spectrum data and then identify all visible emission lines present in it.\n\n"
            f"Allowed candidate lines:\n{spec.vocabulary_text}\n\n"
            "Spectrum Data (100 evenly spaced points):\n"
            f"Wavelength (Å): [{w_str}]\n"
            f"Flux: [{f_str}]\n\n"
            "You MUST conclude your response with the exact format:\n"
            "EMISSION LINES: line1, line2, ...\n"
            "If no emission lines from the list are present, write:\n"
            "EMISSION LINES: NONE"
        )

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        if not hasattr(task, "get_prompt_spec") and hasattr(task, "format_options"):
            task = DistanceClassificationTask(task)

        spec = task.get_prompt_spec()
        messages_batch = []
        
        for sample in samples:
            w_str, f_str = subsample_spectrum(sample.wavelength, sample.flux, num_points=100)
            if isinstance(spec, DistanceClassPromptSpec):
                prompt = self._build_distance_prompt(spec, w_str, f_str)
            elif isinstance(spec, EmissionLinePromptSpec):
                prompt = self._build_emission_prompt(spec, w_str, f_str)
            else:
                prompt = task.default_prompt(wavelength_str=w_str, flux_str=f_str)
            
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": "Brief analysis:",
                },
            ]
            messages_batch.append(messages)
            
        texts = [self._processor.apply_chat_template(msgs, tokenize=False, continue_final_message=True) for msgs in messages_batch]
        
        self._processor.tokenizer.padding_side = 'left'
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token
            
        inputs = self._processor.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False
            )
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_responses = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        parsed_results = [task.default_parse(r) for r in raw_responses]

        failed_indices = [i for i, p in enumerate(parsed_results) if p is None or p == "UNKNOWN"]
        fallback_tag = task.fallback_tag() if hasattr(task, "fallback_tag") else ""
        
        if failed_indices and fallback_tag:
            fallback_texts = []
            for i in failed_indices:
                fallback_prompt = texts[i] + raw_responses[i] + fallback_tag
                fallback_texts.append(fallback_prompt)
                
            fallback_inputs = self._processor.tokenizer(fallback_texts, return_tensors="pt", padding=True)
            fallback_inputs = {k: v.to(self._device) for k, v in fallback_inputs.items()}
            
            with torch.no_grad():
                fallback_output_ids = self._model.generate(
                    **fallback_inputs,
                    max_new_tokens=self._fallback_max_tokens,
                    do_sample=False
                )
                
            fb_input_len = fallback_inputs['input_ids'].shape[1]
            fb_generated_ids = fallback_output_ids[:, fb_input_len:]
            fb_raw_responses = self._processor.tokenizer.batch_decode(fb_generated_ids, skip_special_tokens=True)
            
            for idx, f_idx in enumerate(failed_indices):
                new_raw = raw_responses[f_idx] + fallback_tag + " " + fb_raw_responses[idx]
                raw_responses[f_idx] = new_raw
                parsed = task.default_parse(new_raw)
                if parsed is None or parsed == "UNKNOWN":
                    if getattr(task, "name", "") == "emission_lines":
                        parsed = []
                    else:
                        parsed = "UNKNOWN"
                parsed_results[f_idx] = parsed
        
        return [
            ModelResponse(
                parsed=parsed_results[i] if parsed_results[i] is not None else [], 
                raw_text=raw_responses[i],
                forced_fallback=(i in failed_indices)
            ) for i in range(len(raw_responses))
        ]
