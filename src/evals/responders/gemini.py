import io
import re
from typing import List
from . import EvalSample, BucketScheme, ModelResponse
from .utils import render_spectrum_plot
from google import genai
from google.genai import types
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def build_mcq_prompt(scheme: BucketScheme) -> str:
    prompt = (
        "Classify the redshift (z) of the astronomical spectrum shown in the image.\n\n"
        f"Categories:\n{scheme.format_options_multiline()}\n\n"
        "Provide your classification in the exact format: 'FINAL ANSWER: <label>'."
    )
    return prompt

def extract_final_answer(response: str, labels: List[str]) -> str:
    match = re.search(r"FINAL ANSWER:\s*([" + "".join(labels) + r"])", response, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "UNKNOWN"

class GeminiResponder:
    def __init__(self, config: dict, device: str):
        assert "gemini_model" in config and config["gemini_model"], "Missing 'gemini_model' in config"
        assert "gemini_num_workers" in config, "Missing 'gemini_num_workers' in config"
        
        # We ignore device for Gemini API, but accept it for compatibility
        self._model_name = config["gemini_model"]
        self._num_workers = config["gemini_num_workers"]
        self._max_output_tokens = config.get("gemini_max_output_tokens", 1024)
        self._temperature = config.get("gemini_temperature", 0.0)
        self.client = genai.Client()

    def get_config(self) -> dict:
        return {
            "type": "GeminiResponder",
            "model_name": self._model_name,
            "num_workers": self._num_workers
        }

    def respond_batch(self, samples: List[EvalSample], scheme: BucketScheme) -> List[ModelResponse]:        
        prompt = build_mcq_prompt(scheme)
        
        # Prepare inputs
        images = []
        for sample in samples:
            png_bytes = render_spectrum_plot(
                sample.wavelength, sample.flux,
                mask=sample.mask, survey=sample.survey,
            )
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(image)
            
        # Process concurrently
        responses = [None] * len(samples)
        
        def process_sample(idx: int, img: Image.Image):
            try:
                chat = self.client.chats.create(model=self._model_name)
                
                # Build generation config
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=self._max_output_tokens,
                    temperature=self._temperature
                )
                
                response = chat.send_message([img, prompt], config=gen_config)
                raw_text = response.text if response.text else ""
            except Exception as e:
                print(f"Gemini API error on index {idx}: {e}")
                raw_text = ""
            
            label = extract_final_answer(raw_text, scheme.labels)
            forced_fallback = False
            
            # Conditional Fallback: If it stopped prematurely or failed to format the answer
            if label == "UNKNOWN" and raw_text:
                try:
                    forced_fallback = True
                    fallback_prompt = "You did not provide the final answer. Please provide it now in the exact format: 'FINAL ANSWER: <label>'."                    
                    fallback_config = types.GenerateContentConfig(
                        max_output_tokens=15,
                        temperature=self._temperature
                    )
                    fallback_response = chat.send_message(fallback_prompt, config=fallback_config)
                    fallback_text = fallback_response.text if fallback_response.text else ""
                    raw_text += "\n\n[FALLBACK]: " + fallback_text
                    label = extract_final_answer(fallback_text, scheme.labels)
                except Exception as e:
                    print(f"Gemini API fallback error on index {idx}: {e}")
            
            return ModelResponse(label=label, raw_text=raw_text, forced_fallback=forced_fallback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample, idx, img): idx 
                for idx, img in enumerate(images)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(images), desc="Gemini API", leave=False):
                idx = future_to_idx[future]
                responses[idx] = future.result()
                
        return responses
