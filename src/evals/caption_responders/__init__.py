from __future__ import annotations

from typing import Any, Dict

from .base import (
    BaseCaptionResponder,
    CaptionSample,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
)


def get_caption_responder(config: Dict[str, Any], device: str) -> BaseCaptionResponder:
    """Instantiate a specialized caption responder from configuration."""
    from .astrobridge import AstroBridgeCaptionResponder
    from .hf_vision import HFVisionCaptionResponder
    from .hf_text import HFTextCaptionResponder
    from .mock import MockCaptionResponder

    REGISTRY = {
        "astrobridge": AstroBridgeCaptionResponder,
        "hf_vision": HFVisionCaptionResponder,
        "hf_text": HFTextCaptionResponder,
        "mock": MockCaptionResponder,
    }

    responder_type = config.get("responder_type")
    if responder_type not in REGISTRY:
        raise ValueError(
            f"Unknown caption responder '{responder_type}'. Available: {list(REGISTRY.keys())}"
        )

    return REGISTRY[responder_type](config, device)


__all__ = [
    "BaseCaptionResponder",
    "CaptionSample",
    "GeneratedCaption",
    "DEFAULT_SPECTRUM_CAPTION_PROMPT",
    "get_caption_responder",
]

