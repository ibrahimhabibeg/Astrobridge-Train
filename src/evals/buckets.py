from dataclasses import dataclass
from typing import List

@dataclass
class BucketScheme:
    name: str
    labels: List[str]
    thresholds: List[float]
    descriptions: List[str]
    
    def __post_init__(self):
        if len(self.thresholds) != len(self.labels) - 1:
            raise ValueError("Number of thresholds must be one less than number of labels")
        if len(self.descriptions) != len(self.labels):
            raise ValueError("Number of descriptions must match number of labels")
            
    def get_config(self) -> dict:
        return {
            "name": self.name,
            "labels": self.labels,
            "thresholds": self.thresholds,
            "descriptions": self.descriptions
        }
            
    def classify(self, z: float) -> str:
        for i, threshold in enumerate(self.thresholds):
            if z < threshold:
                return self.labels[i]
        return self.labels[-1]
        
    def format_options(self) -> str:
        options = []
        for label, desc in zip(self.labels, self.descriptions):
            options.append(f"{label}: {desc}")
        return ", ".join(options)
        
    def format_options_multiline(self) -> str:
        options = []
        for label, desc in zip(self.labels, self.descriptions):
            options.append(f"{label}: {desc}")
        return "\n".join(options)

def five_bucket_scheme() -> BucketScheme:
    return BucketScheme(
        name="5-group (A-E)",
        labels=["A", "B", "C", "D", "E"],
        thresholds=[0.1, 0.5, 1.0, 2.0],
        descriptions=[
            "very close (z < 0.1)",
            "close (0.1 <= z < 0.5)",
            "intermediate (0.5 <= z < 1.0)",
            "far (1.0 <= z < 2.0)",
            "very far (z >= 2.0)"
        ]
    )

def three_bucket_scheme() -> BucketScheme:
    return BucketScheme(
        name="3-group (A-C)",
        labels=["A", "B", "C"],
        thresholds=[0.1, 0.5],
        descriptions=[
            "very close (z < 0.1)",
            "close (0.1 <= z < 0.5)",
            "intermediate or far (z >= 0.5)"
        ]
    )
