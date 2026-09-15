from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol


class Task(Protocol):
    name: str
    benchmark_name: str

    def build_frontier_prompt(self, caption: str, item: Optional[Dict[str, Any]] = None) -> str:
        ...

    def fallback_tag(self) -> str:
        ...

    def default_parse(self, raw_text: str, **kwargs: Any) -> Any:
        ...

    def get_parse_fn(self, item: Optional[Dict[str, Any]] = None) -> Callable[[str], Any]:
        ...

    def extract_ground_truth(self, item: Any) -> Any:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...
