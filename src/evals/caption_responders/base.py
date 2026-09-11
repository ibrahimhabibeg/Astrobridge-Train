from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

DEFAULT_SPECTRUM_CAPTION_PROMPT = (
    "Describe the given spectrum in detail, noting its continuum shape, "
    "prominent emission or absorption features, and spectral characteristics."
)


@dataclass
class CaptionSample:
    """Sample representation for spectrum captioning."""
    sample_id: str
    wavelength: Any  # numpy array
    flux: Any        # numpy array
    mask: Any        # numpy array
    survey: str
    ivar: Optional[Any] = None  # numpy array (optional)


@dataclass
class GeneratedCaption:
    """Stores the generated caption and metadata for a sample."""
    sample_id: str
    caption: str
    responder_type: str
    model_id: str
    caption_prompt: str
    survey: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCaptionResponder(Protocol):
    """Protocol for dedicated caption responders."""

    def generate_captions(
        self,
        samples: List[CaptionSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        """Generate captions for a batch of spectrum samples."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for this responder."""
        ...

