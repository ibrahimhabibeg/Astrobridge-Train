from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Set


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


def parse_single_choice(
    raw_text: str, allowed_letters: Optional[Set[str]] = None
) -> Optional[str]:
    if not raw_text:
        return None

    match = re.search(
        r"FINAL ANSWER:\s*(?:option\s*|choice\s*)?[\*\(\[]*([A-Za-z])[\*\)\]\.\:]*(?:\s|$)",
        raw_text,
        re.IGNORECASE,
    )
    if match:
        ch = match.group(1).upper()
        if not allowed_letters or ch in allowed_letters:
            return ch

    match = re.search(
        r"(?:answer|category|choice|option)\s*(?:is\s*)?[:\s]*[\*\(\[]*([A-Za-z])[\*\)\]\.\:]*(?:\s|$)",
        raw_text,
        re.IGNORECASE,
    )
    if match:
        ch = match.group(1).upper()
        if not allowed_letters or ch in allowed_letters:
            return ch

    return None


def parse_multi_choice(
    raw_text: str, allowed_letters: Optional[Set[str]] = None
) -> List[str]:
    if not raw_text:
        return []

    match = re.search(r"FINAL ANSWER:\s*(.*)", raw_text, re.IGNORECASE)
    target = match.group(1).strip() if match else raw_text.strip()

    if re.search(r"\bNONE\b", target, re.IGNORECASE):
        other_letters = [
            ch
            for ch in re.findall(r"\b[A-Za-z]\b", target)
            if not allowed_letters or ch.upper() in allowed_letters
        ]
        if not other_letters:
            return []

    letters = re.findall(r"\b[A-Za-z]\b", target)
    seen = set()
    extracted = []
    for l in letters:
        upper_l = l.upper()
        if (not allowed_letters or upper_l in allowed_letters) and upper_l not in seen:
            seen.add(upper_l)
            extracted.append(upper_l)
    return extracted
