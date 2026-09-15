from __future__ import annotations

from typing import Any, Dict

from .sample import SpectrumSample, CaptionSample
from .mock import MockCaptionResponder
from .base import (
    BaseResponder,
    BaseCaptionResponder,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
    resolve_caption_prompt,
)


def get_responder(config: Dict[str, Any], device: str) -> BaseResponder:
    from .astrobridge import AstroBridgeResponder
    from .hf_vision import HFVisionResponder
    from .hf_text import HFTextResponder

    REGISTRY = {
        "astrobridge": AstroBridgeResponder,
        "hf_vision": HFVisionResponder,
        "hf_text": HFTextResponder,
        "mock": MockCaptionResponder,
    }

    responder_type = config.get("responder_type")
    if responder_type not in REGISTRY:
        raise ValueError(
            f"Unknown responder '{responder_type}'. Available: {list(REGISTRY.keys())}"
        )

    return REGISTRY[responder_type](config, device)


get_caption_responder = get_responder

__all__ = [
    "SpectrumSample",
    "CaptionSample",
    "BaseResponder",
    "BaseCaptionResponder",
    "GeneratedCaption",
    "MockCaptionResponder",
    "DEFAULT_SPECTRUM_CAPTION_PROMPT",
    "resolve_caption_prompt",
    "get_responder",
    "get_caption_responder",
]
