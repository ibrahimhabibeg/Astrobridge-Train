import io
from typing import List, Any

from google import genai
from google.genai import types
from PIL import Image
import concurrent.futures
from tqdm import tqdm

from . import EvalSample, ModelResponse
from .utils import render_spectrum_plot


class GeminiResponder:
    def __init__(self, config: dict, device: str):
        assert "gemini_model" in config and config["gemini_model"], "Missing 'gemini_model' in config"
        self._model_name = config["gemini_model"]
        self._num_workers = config.get("gemini_num_workers", 4)
        self._max_tokens = config.get("max_tokens", 2048)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 128)
        self._temperature = config.get("gemini_temperature", 0.0)
        self.client = genai.Client()

    def get_config(self) -> dict:
        return {
            "type": "GeminiResponder",
            "model_name": self._model_name,
            "num_workers": self._num_workers,
        }

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        prompt = task.build_prompt(image_mode=True)

        images = []
        for sample in samples:
            png_bytes = render_spectrum_plot(sample.wavelength, sample.flux, mask=sample.mask)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(image)

        responses = [None] * len(samples)

        def process_sample(idx: int, img: Image.Image):
            try:
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                chat = self.client.chats.create(model=self._model_name, config=gen_config)
                response = chat.send_message([img, prompt])
                raw_text = response.text if response.text else ""
            except Exception as e:
                print(f"Gemini API error on index {idx}: {e}")
                raw_text = ""
                chat = None

            parsed = task.default_parse(raw_text)
            forced_fallback = False
            fallback_tag = task.fallback_tag()

            if (parsed is None or parsed == "UNKNOWN") and fallback_tag and chat:
                try:
                    forced_fallback = True
                    fallback_config = types.GenerateContentConfig(
                        max_output_tokens=self._fallback_max_tokens,
                        temperature=self._temperature,
                    )
                    fallback_prompt = (
                        f"Please continue your previous response and output only the "
                        f"final answer starting with the indicator '{fallback_tag.strip()}'"
                    )
                    fallback_response = chat.send_message(fallback_prompt, config=fallback_config)
                    fallback_text = fallback_response.text if fallback_response.text else ""
                    raw_text += "\n" + fallback_text
                    parsed = task.default_parse(raw_text)
                    if parsed is None or parsed == "UNKNOWN":
                        parsed = [] if getattr(task, "name", "") == "emission_lines" else "UNKNOWN"
                except Exception as e:
                    print(f"Gemini API fallback error on index {idx}: {e}")

            return ModelResponse(
                parsed=parsed if parsed is not None else [],
                raw_text=raw_text,
                forced_fallback=forced_fallback,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample, idx, img): idx
                for idx, img in enumerate(images)
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(images),
                desc="Gemini API",
                leave=False,
            ):
                idx = future_to_idx[future]
                responses[idx] = future.result()

        return responses
