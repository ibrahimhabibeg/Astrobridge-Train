from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass
class FrontierResponse:
    """Represents a response returned by a frontier evaluation model."""
    raw_text: str
    parsed: Any
    forced_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrontierModel(Protocol):
    """Protocol for frontier models querying property prediction from captions."""

    def predict(
        self,
        prompt: str,
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> FrontierResponse:
        """Query the model with a single prompt, applying parsing and optional fallback."""
        ...

    def predict_batch(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[FrontierResponse]:
        """Query the model with a batch of prompts concurrently."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return serializable metadata/configuration of this frontier model."""
        ...

