import io
import re
from typing import List
from . import EvalSample, BucketScheme, ModelResponse
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import torch
from PIL import Image
import matplotlib.pyplot as plt

def render_spectrum_plot(wavelength, flux, mask=None, survey=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if mask is not None:
        valid = ~mask
        ax.plot(wavelength[valid], flux[valid], color='blue', lw=1, label='Valid')
        ax.plot(wavelength[mask], flux[mask], color='red', lw=1, alpha=0.5, label='Masked')
    else:
        ax.plot(wavelength, flux, color='blue', lw=1)
        
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title('Spectrum')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def build_mcq_prompt(scheme: BucketScheme) -> str:
    prompt = (
        "Look at this astronomical spectrum and estimate the object's redshift.\n"
        "Choose exactly one option:\n"
    )
    prompt += scheme.format_options_multiline() + "\n\n"
    prompt += (
        "Reply with ONLY a single line in this format:\n"
        "FINAL ANSWER: X\n"
        f"where X is one of {''.join(scheme.labels)}. Do not explain your reasoning."
    )
    return prompt

def extract_final_answer(response: str, labels_str: str) -> str:
    match = re.search(r"([" + labels_str + r"])", response)
    if match: return match.group(1)
    return "UNKNOWN"

class BaseQwenResponder:
    def __init__(self, model_id: str, device: str):
        self._model_id = model_id
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, 
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

    def respond_batch(self, samples: List[EvalSample], scheme: BucketScheme) -> List[ModelResponse]:        
        prompt = build_mcq_prompt(scheme)
        messages_batch = []
        
        for sample in samples:
            png_bytes = render_spectrum_plot(
                sample.wavelength, sample.flux,
                mask=sample.mask, survey=sample.survey,
            )
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    
            labels_str = "".join(scheme.labels)
            
            messages = [
                {
                    "role": "system",
                    "content": f"You are an astronomer. When shown a spectrum, reply with ONLY 'FINAL ANSWER: X' where X is one of {labels_str}. No explanation."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "FINAL ANSWER:",
                },
            ]
            messages_batch.append(messages)
            
        raw_responses = self._generate_from_messages_batch(messages_batch)
        
        labels_str = "".join(scheme.labels)
        return [
            ModelResponse(
                label=extract_final_answer(r, labels_str), 
                raw_text=r
            ) for r in raw_responses
        ]

    def _generate_from_messages_batch(self, messages_batch) -> List[str]:        
        texts = [self._processor.apply_chat_template(msgs, continue_final_message=True, enable_thinking=False) for msgs in messages_batch]
                
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
            output_ids = self._model.generate(**inputs, max_new_tokens=16)
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)
