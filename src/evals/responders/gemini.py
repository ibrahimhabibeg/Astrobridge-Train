import io
import re
from typing import List, Any
from . import EvalSample, ModelResponse
from .utils import render_spectrum_plot
from ..tasks.distance_classification import DistanceClassPromptSpec, DistanceClassificationTask
from ..tasks.emission_lines import EmissionLinePromptSpec
from google import genai
from google.genai import types
from PIL import Image
import concurrent.futures
from tqdm import tqdm

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

    def _build_distance_prompt(self, spec: DistanceClassPromptSpec) -> str:
        return (
            "Classify the redshift (z) of the astronomical spectrum shown in the image.\n\n"
            f"Categories:\n{spec.options_multiline}\n\n"
            "Provide your classification in the exact format: 'FINAL ANSWER: <label>'."
        )

    def _build_emission_prompt(self, spec: EmissionLinePromptSpec) -> str:
        return (
            "Identify all visible emission lines present in the astronomical spectrum shown in the image.\n\n"
            f"Allowed candidate lines:\n{spec.vocabulary_text}\n\n"
            "Provide your extracted lines on the final line in the exact format: 'EMISSION LINES: line1, line2, ...'.\n"
            "If no emission lines from the candidate list are visible, write: 'EMISSION LINES: NONE'."
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
                
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=self._max_output_tokens,
                    temperature=self._temperature
                )
                
                response = chat.send_message([img, prompt], config=gen_config)
                raw_text = response.text if response.text else ""
            except Exception as e:
                print(f"Gemini API error on index {idx}: {e}")
                raw_text = ""
            
            parsed = task.default_parse(raw_text)
            forced_fallback = False
            
            # Conditional Fallback: If it stopped prematurely or failed to format the answer
            should_fallback = False
            fallback_prompt = ""
            fallback_tokens = 30

            if isinstance(spec, DistanceClassPromptSpec):
                if parsed == "UNKNOWN" and raw_text:
                    should_fallback = True
                    fallback_prompt = "You did not provide the final answer. Please provide it now in the exact format: 'FINAL ANSWER: <label>'."
                    fallback_tokens = 15
            elif isinstance(spec, EmissionLinePromptSpec):
                if not parsed and raw_text and "EMISSION LINES" not in raw_text.upper():
                    should_fallback = True
                    fallback_prompt = "You did not provide the final answer. Please provide it now in the exact format: 'EMISSION LINES: line1, line2, ...' or 'EMISSION LINES: NONE'."
                    fallback_tokens = 128

            if should_fallback:
                try:
                    forced_fallback = True
                    fallback_config = types.GenerateContentConfig(
                        max_output_tokens=fallback_tokens,
                        temperature=self._temperature
                    )
                    fallback_response = chat.send_message(fallback_prompt, config=fallback_config)
                    fallback_text = fallback_response.text if fallback_response.text else ""
                    raw_text += "\n\n[FALLBACK]: " + fallback_text
                    parsed = task.default_parse(fallback_text)
                except Exception as e:
                    print(f"Gemini API fallback error on index {idx}: {e}")
            
            return ModelResponse(parsed=parsed, raw_text=raw_text, forced_fallback=forced_fallback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample, idx, img): idx 
                for idx, img in enumerate(images)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(images), desc="Gemini API", leave=False):
                idx = future_to_idx[future]
                responses[idx] = future.result()
                
        return responses
