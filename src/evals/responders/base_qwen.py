import io
import re
from typing import List, Any
from . import EvalSample, ModelResponse
from .utils import render_spectrum_plot
from ..tasks.distance_classification import DistanceClassPromptSpec, DistanceClassificationTask
from ..tasks.emission_lines import EmissionLinePromptSpec
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import torch
from PIL import Image

class BaseQwenResponder:
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
            "type": "BaseQwenResponder",
            "model_id": self._model_id
        }

    def _build_distance_prompt(self, spec: DistanceClassPromptSpec) -> str:
        return (
            "Classify the redshift (z) of the astronomical spectrum shown in the image.\n\n"
            f"Categories:\n{spec.options_multiline}\n\n"
            "Provide exactly ONE sentence of analysis, then on a new line write 'FINAL ANSWER: <label>'."
        )

    def _build_emission_prompt(self, spec: EmissionLinePromptSpec) -> str:
        return (
            "Identify all visible emission lines present in the astronomical spectrum shown in the image.\n\n"
            f"Allowed candidate lines:\n{spec.vocabulary_text}\n\n"
            "Provide exactly ONE sentence of analysis, then on a new line write 'EMISSION LINES: line1, line2, ...' or 'EMISSION LINES: NONE'."
        )

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        if not hasattr(task, "get_prompt_spec") and hasattr(task, "format_options"):
            task = DistanceClassificationTask(task)

        spec = task.get_prompt_spec()
        if isinstance(spec, DistanceClassPromptSpec):
            prompt = self._build_distance_prompt(spec)
        elif isinstance(spec, EmissionLinePromptSpec):
            prompt = self._build_emission_prompt(spec)
        else:
            prompt = task.default_prompt()

        messages_batch = []
        for sample in samples:
            png_bytes = render_spectrum_plot(
                sample.wavelength, sample.flux,
                mask=sample.mask, survey=sample.survey,
            )
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Brief analysis:",
                },
            ]
            messages_batch.append(messages)
            
        return self._generate_from_messages_batch(messages_batch, task)

    def _generate_from_messages_batch(self, messages_batch, task: Any) -> List[ModelResponse]:        
        texts = [self._processor.apply_chat_template(msgs, continue_final_message=True) for msgs in messages_batch]
                
        images = []
        for msgs in messages_batch:
            for msg in msgs:
                for content in msg.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        images.append(content["image"])
        
        self._processor.tokenizer.padding_side = 'left'
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token
            
        inputs = self._processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False
            )
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_responses = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        spec = task.get_prompt_spec()
        parsed_results = [task.default_parse(r) for r in raw_responses]

        if isinstance(spec, DistanceClassPromptSpec):
            failed_indices = [i for i, p in enumerate(parsed_results) if p == "UNKNOWN"]
            fallback_tag = "\n\nFINAL ANSWER:"
            max_fb_tokens = 5
        elif isinstance(spec, EmissionLinePromptSpec):
            failed_indices = [i for i, p in enumerate(parsed_results) if not p and "EMISSION LINES" not in raw_responses[i].upper()]
            fallback_tag = "\n\nEMISSION LINES:"
            max_fb_tokens = 64
        else:
            failed_indices = []
            fallback_tag = ""
            max_fb_tokens = 10
        
        if failed_indices:
            fallback_texts = []
            fallback_images = []
            for i in failed_indices:
                fallback_prompt = texts[i] + raw_responses[i] + fallback_tag
                fallback_texts.append(fallback_prompt)
                fallback_images.append(images[i])
                
            fallback_inputs = self._processor(text=fallback_texts, images=fallback_images, return_tensors="pt", padding=True)
            fallback_inputs = {k: v.to(self._device) for k, v in fallback_inputs.items()}
            
            with torch.no_grad():
                fallback_output_ids = self._model.generate(
                    **fallback_inputs,
                    max_new_tokens=max_fb_tokens,
                    do_sample=False
                )
                
            fb_input_len = fallback_inputs['input_ids'].shape[1]
            fb_generated_ids = fallback_output_ids[:, fb_input_len:]
            fb_raw_responses = self._processor.batch_decode(fb_generated_ids, skip_special_tokens=True)
            
            for idx, f_idx in enumerate(failed_indices):
                new_raw = raw_responses[f_idx] + fallback_tag + " " + fb_raw_responses[idx]
                raw_responses[f_idx] = new_raw
                parsed_results[f_idx] = task.default_parse(new_raw)
        
        return [
            ModelResponse(
                parsed=parsed_results[i], 
                raw_text=raw_responses[i],
                forced_fallback=(i in failed_indices)
            ) for i in range(len(raw_responses))
        ]
