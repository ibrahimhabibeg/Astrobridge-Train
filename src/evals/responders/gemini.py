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
        assert "max_tokens" in config, "Missing 'max_tokens' in config"
        assert "fallback_max_tokens" in config, "Missing 'fallback_max_tokens' in config"
        
        # We ignore device for Gemini API, but accept it for compatibility
        self._model_name = config["gemini_model"]
        self._num_workers = config["gemini_num_workers"]
        self._max_tokens = config["max_tokens"]
        self._fallback_max_tokens = config["fallback_max_tokens"]
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
            "Analyze and describe the astronomical spectrum shown in the image and then identify all visible emission lines present in it.\n\n"
            f"Allowed candidate lines:\n{spec.vocabulary_text}\n\n"
            "You MUST conclude your response with the exact format:\n"
            "EMISSION LINES: line1, line2, ...\n"
            "If no emission lines from the list are present, write:\n"
            "EMISSION LINES: NONE"
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
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                    temperature=self._temperature
                )
                
                contents = [types.Content(role="user", parts=[types.Part.from_image(img), types.Part.from_text(text=prompt)])]
                response = self.client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=gen_config
                )
                raw_text = response.text if response.text else ""
            except Exception as e:
                print(f"Gemini API error on index {idx}: {e}")
                raw_text = ""
            
            parsed = task.default_parse(raw_text)
            forced_fallback = False
            
            fallback_tag = task.fallback_tag() if hasattr(task, "fallback_tag") else ""
            
            if (parsed is None or parsed == "UNKNOWN") and fallback_tag:
                try:
                    forced_fallback = True
                    fallback_config = types.GenerateContentConfig(
                        max_output_tokens=self._fallback_max_tokens,
                        temperature=self._temperature
                    )
                    fallback_contents = [
                        types.Content(role="user", parts=[types.Part.from_image(img), types.Part.from_text(text=prompt)]),
                        types.Content(role="model", parts=[types.Part.from_text(text=raw_text + fallback_tag)])
                    ]
                    fallback_response = self.client.models.generate_content(
                        model=self._model_name,
                        contents=fallback_contents,
                        config=fallback_config
                    )
                    fallback_text = fallback_response.text if fallback_response.text else ""
                    raw_text += fallback_tag + " " + fallback_text
                    parsed = task.default_parse(raw_text)
                    if parsed is None or parsed == "UNKNOWN":
                        if getattr(task, "name", "") == "emission_lines":
                            parsed = []
                        else:
                            parsed = "UNKNOWN"
                except Exception as e:
                    print(f"Gemini API fallback error on index {idx}: {e}")
            
            return ModelResponse(parsed=parsed if parsed is not None else [], raw_text=raw_text, forced_fallback=forced_fallback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample, idx, img): idx 
                for idx, img in enumerate(images)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(images), desc="Gemini API", leave=False):
                idx = future_to_idx[future]
                responses[idx] = future.result()
                
        return responses
