from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .sample import SpectrumSample
from ..prompts import render_prompt

DEFAULT_SPECTRUM_CAPTION_PROMPT = render_prompt("caption_generation/default.jinja2")


def resolve_caption_prompt(
    config: Dict[str, Any],
    default_template: str = "caption_generation/default.jinja2",
) -> str:
    if config.get("caption_prompt"):
        prompt_val = str(config["caption_prompt"]).strip()
        return render_prompt(prompt_val) if prompt_val.endswith(".jinja2") else prompt_val
    if config.get("caption_prompt_template"):
        return render_prompt(str(config["caption_prompt_template"]).strip())
    return render_prompt(default_template)


@dataclass
class GeneratedCaption:
    sample_id: str
    caption: str
    responder_type: str
    model_id: str
    caption_prompt: str
    survey: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseResponder(Protocol):
    def generate_captions(
        self,
        samples: List[SpectrumSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...


BaseCaptionResponder = BaseResponder

