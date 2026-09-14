from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..prompts import render_prompt

DEFAULT_SPECTRUM_CAPTION_PROMPT = render_prompt("caption_generation/default.jinja2")


def resolve_caption_prompt(
    config: Dict[str, Any],
    default_template: str = "caption_generation/default.jinja2",
) -> str:
    """Resolve caption prompt from config: supports explicit text, template name, or default template."""
    if "caption_prompt" in config and config["caption_prompt"]:
        prompt_val = str(config["caption_prompt"]).strip()
        if prompt_val.endswith(".jinja2"):
            return render_prompt(prompt_val)
        return prompt_val
    if "caption_prompt_template" in config and config["caption_prompt_template"]:
        return render_prompt(str(config["caption_prompt_template"]).strip())
    return render_prompt(default_template)


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

