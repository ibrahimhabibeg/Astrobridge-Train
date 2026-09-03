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
