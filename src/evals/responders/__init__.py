from dataclasses import dataclass
from typing import Protocol, List
from ..buckets import BucketScheme

@dataclass
class EvalSample:
    wavelength: object # numpy array
    flux: object       # numpy array
    mask: object       # numpy array
    survey: str
    ivar: object = None # numpy array (optional)

@dataclass
class ModelResponse:
    label: str
    raw_text: str
    forced_fallback: bool = False

class Responder(Protocol):
    def respond_batch(self, samples: List[EvalSample], scheme: BucketScheme) -> List[ModelResponse]:
        """
        Given raw spectra samples and a bucket scheme, return predicted labels and raw text.
        """
        ...
        
    def get_config(self) -> dict:
        """
        Return the configuration/metadata of this responder.
        """
        ...

def get_responder(config: dict, device: str) -> Responder:
    # Import inside the factory to avoid circular imports during initialization
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
