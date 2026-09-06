from dataclasses import dataclass
from typing import Protocol, List, Any, Optional
from ..tasks import EvalTask

@dataclass
class EvalSample:
    wavelength: object # numpy array
    flux: object       # numpy array
    mask: object       # numpy array
    survey: str
    ivar: object = None # numpy array (optional)

@dataclass
class ModelResponse:
    raw_text: str = ""
    parsed: Any = None
    label: str = ""
    forced_fallback: bool = False

    def __post_init__(self):
        if self.parsed is not None and not self.label and isinstance(self.parsed, str):
            self.label = self.parsed
        elif self.label and self.parsed is None:
            self.parsed = self.label

class Responder(Protocol):
    def respond_batch(self, samples: List[EvalSample], task: EvalTask) -> List[ModelResponse]:
        """
        Given raw spectra samples and an evaluation task, return predictions and raw text.
        """
        ...
        
    def get_config(self) -> dict:
        """
        Return the configuration/metadata of this responder.
        """
        ...

def get_responder(config: dict, device: str) -> Responder:
    from .astrobridge import AstroBridgeResponder
    from .base_qwen import BaseQwenResponder
    from .base_qwen_text import BaseQwenTextResponder
    from .gemini import GeminiResponder

    RESPONDER_REGISTRY = {
        "astrobridge": AstroBridgeResponder,
        "base_qwen": BaseQwenResponder,
        "base_qwen_text": BaseQwenTextResponder,
        "gemini": GeminiResponder,
    }

    responder_id = config.get("responder_type")
    if responder_id not in RESPONDER_REGISTRY:
        raise ValueError(f"Unknown responder '{responder_id}'. Available: {list(RESPONDER_REGISTRY.keys())}")
    
    return RESPONDER_REGISTRY[responder_id](config, device)
