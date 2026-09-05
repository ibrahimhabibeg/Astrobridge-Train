import re
import torch
import numpy as np
from typing import List
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from . import EvalSample, BucketScheme, ModelResponse

def subsample_spectrum(wavelength, flux, num_points=100):
    wavelength = np.array(wavelength).flatten()
    flux = np.array(flux).flatten()
    
    indices = np.linspace(0, len(wavelength) - 1, num_points, dtype=int)
    w_sub = wavelength[indices]
    f_sub = flux[indices]
    
    w_str = ", ".join([f"{w:.1f}" for w in w_sub])
    f_str = ", ".join([f"{f:.3f}" for f in f_sub])
    return w_str, f_str

def build_mcq_prompt(scheme: BucketScheme, w_str: str, f_str: str) -> str:
    prompt = (
        "Classify the redshift (z) of the following astronomical spectrum.\n\n"
        f"Categories:\n{scheme.format_options_multiline()}\n\n"
        "Spectrum Data (100 evenly spaced points):\n"
        f"Wavelength (Å): [{w_str}]\n"
        f"Flux: [{f_str}]\n\n"
        "Provide exactly ONE sentence of analysis, then on a new line write 'FINAL ANSWER: <label>'."
    )
    return prompt

def extract_final_answer(response: str, labels: List[str]) -> str:
    match = re.search(r"FINAL ANSWER:\s*([" + "".join(labels) + r"])", response, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "UNKNOWN"

class BaseQwenTextResponder:
    def __init__(self, config: dict, device: str):
        assert "base_llm_id" in config and config["base_llm_id"], "Missing 'base_llm_id' in config"
        self._model_id = config["base_llm_id"]
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

    def respond_batch(self, samples: List[EvalSample], scheme: BucketScheme) -> List[ModelResponse]:        
        messages_batch = []
        
        for sample in samples:
            w_str, f_str = subsample_spectrum(sample.wavelength, sample.flux, num_points=100)
            prompt = build_mcq_prompt(scheme, w_str, f_str)
            
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
                max_new_tokens=1024,
                do_sample=False
            )
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_responses = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels = [extract_final_answer(r, scheme.labels) for r in raw_responses]
        failed_indices = [i for i, label in enumerate(labels) if label == "UNKNOWN"]
        
        if failed_indices:
            fallback_texts = []
            for i in failed_indices:
                # Append the forced answer prompt to the cut-off reasoning
                fallback_prompt = texts[i] + raw_responses[i] + "\n\nFINAL ANSWER:"
                fallback_texts.append(fallback_prompt)
                
            fallback_inputs = self._processor.tokenizer(fallback_texts, return_tensors="pt", padding=True)
            fallback_inputs = {k: v.to(self._device) for k, v in fallback_inputs.items()}
            
            with torch.no_grad():
                fallback_output_ids = self._model.generate(
                    **fallback_inputs,
                    max_new_tokens=10, # Just enough tokens for the letter
                    do_sample=False
                )
                
            fb_input_len = fallback_inputs['input_ids'].shape[1]
            fb_generated_ids = fallback_output_ids[:, fb_input_len:]
            fb_raw_responses = self._processor.tokenizer.batch_decode(fb_generated_ids, skip_special_tokens=True)
            
            for idx, f_idx in enumerate(failed_indices):
                new_raw = raw_responses[f_idx] + "\n\nFINAL ANSWER:" + fb_raw_responses[idx]
                raw_responses[f_idx] = new_raw
                labels[f_idx] = extract_final_answer(new_raw, scheme.labels)
        
        return [
            ModelResponse(
                label=labels[i], 
                raw_text=raw_responses[i],
                forced_fallback=(i in failed_indices)
            ) 
            for i in range(len(raw_responses))
        ]
