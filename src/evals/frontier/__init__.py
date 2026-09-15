from __future__ import annotations

from typing import Any, Dict

from .base import FrontierModel, FrontierResponse
from .gemini import GeminiFrontierModel
from .mock import MockFrontierModel

FRONTIER_REGISTRY = {
    "gemini": GeminiFrontierModel,
    "mock": MockFrontierModel,
}


def get_frontier_model(config: Dict[str, Any]) -> FrontierModel:
    model_type = config.get("frontier_type", "gemini")
    if model_type not in FRONTIER_REGISTRY:
        raise ValueError(
            f"Unknown frontier model type '{model_type}'. Available: {list(FRONTIER_REGISTRY.keys())}"
        )
    return FRONTIER_REGISTRY[model_type](config)


MockFrontierJudge = MockFrontierModel
get_frontier_judge = get_frontier_model

__all__ = [
    "FrontierModel",
    "FrontierResponse",
    "GeminiFrontierModel",
    "MockFrontierModel",
    "MockFrontierJudge",
    "get_frontier_model",
    "get_frontier_judge",
]
