from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .base import FrontierModel, FrontierResponse


class MockFrontierModel(FrontierModel):
    """Mock frontier model for deterministic offline testing and CI."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.default_answer = self.config.get("mock_answer", "B")
        self.simulate_fallback = self.config.get("simulate_fallback", False)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "MockFrontierModel",
            "mock_answer": self.default_answer,
            "simulate_fallback": self.simulate_fallback,
        }

    def predict(
        self,
        prompt: str,
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> FrontierResponse:
        forced_fallback = False

        # Detect task type from prompt context to give smart mock answers
        if "EMISSION LINES:" in prompt:
            # Emission lines task
            if self.simulate_fallback:
                raw_text = "I observe some spectral peaks."
                parsed = parse_fn(raw_text)
                if parsed is None and fallback_tag:
                    raw_text += f"{fallback_tag}Hα, Hβ"
                    parsed = parse_fn(raw_text)
                    forced_fallback = True
            else:
                raw_text = "Analysis shows strong lines.\nEMISSION LINES: Hα, Hβ"
                parsed = parse_fn(raw_text)
        elif "Galaxy" in prompt or "Quasar" in prompt:
            # Source classification task
            raw_text = f"Based on the broad emission lines, this object is a Quasar.\nFINAL ANSWER: Quasar"
            parsed = parse_fn(raw_text)
        elif "Starburst" in prompt or "AGN" in prompt:
            # Subclass classification task
            raw_text = f"Based on the spectral features, this is AGN.\nFINAL ANSWER: AGN"
            parsed = parse_fn(raw_text)
        else:
            # Distance classification task
            ans = self.default_answer
            if self.simulate_fallback:
                raw_text = "The spectrum shows redshifted lines."
                parsed = parse_fn(raw_text)
                if parsed is None and fallback_tag:
                    raw_text += f"{fallback_tag}{ans}"
                    parsed = parse_fn(raw_text)
                    forced_fallback = True
            else:
                raw_text = f"Based on the redshift indicators, category {ans} applies.\nFINAL ANSWER: {ans}"
                parsed = parse_fn(raw_text)

        return FrontierResponse(
            raw_text=raw_text,
            parsed=parsed if parsed is not None else "UNKNOWN",
            forced_fallback=forced_fallback,
            metadata={"mock": True},
        )

    def predict_batch(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[FrontierResponse]:
        return [
            self.predict(p, parse_fn, fallback_tag=fallback_tag, system_prompt=system_prompt)
            for p in prompts
        ]

