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

def get_responder(responder_id: str, model_id: str, device: str) -> Responder:
    from .astrobridge import AstroBridgeResponder
    from .base_qwen import BaseQwenResponder

    RESPONDER_REGISTRY = {
        "astrobridge": AstroBridgeResponder,
        "base_qwen": BaseQwenResponder,
    }

    if responder_id not in RESPONDER_REGISTRY:
        raise ValueError(f"Unknown responder '{responder_id}'. Available: {list(RESPONDER_REGISTRY.keys())}")
    
    return RESPONDER_REGISTRY[responder_id](model_id, device)
