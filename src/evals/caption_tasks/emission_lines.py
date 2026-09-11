from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import pandas as pd

from ..data import load_emission_line_ground_truth
from ..tasks.emission_lines import (
    CANONICAL_LINES,
    CSV_TO_CANONICAL,
    CLEAN_TO_CANONICAL,
    clean_key,
)
from .base import CaptionEvalTask


class CaptionEmissionLineTask(CaptionEvalTask):
    """Evaluates the model's ability to report visible emission lines in spectrum captions."""
    name: str = "caption_emission_lines"

    def __init__(self, ground_truth_df: Optional[pd.DataFrame] = None, **kwargs):
        self.canonical_lines = list(CANONICAL_LINES)
        self._vocabulary_text = ", ".join(self.canonical_lines)

        if ground_truth_df is None:
            ground_truth_df = load_emission_line_ground_truth()

        self.ground_truth_by_id: Dict[str, Dict[str, float]] = {}
        for _, row in ground_truth_df.iterrows():
            eid = str(row["wiki_entity_id"])
            raw_line = str(row["LINE_NAME"])
            snr = float(row["SNR"])

            if raw_line in CSV_TO_CANONICAL:
                canonical = CSV_TO_CANONICAL[raw_line]
                if canonical not in self.canonical_lines:
                    continue
                if eid not in self.ground_truth_by_id:
                    self.ground_truth_by_id[eid] = {}
                if (
                    canonical not in self.ground_truth_by_id[eid]
                    or snr > self.ground_truth_by_id[eid][canonical]
                ):
                    self.ground_truth_by_id[eid][canonical] = snr

    def build_frontier_prompt(self, caption: str) -> str:
        return (
            "You are an expert astrophysicist. You will be provided with a scientific description of an astronomical spectrum.\n"
            "Based ONLY on the description of the spectrum, identify all visible emission lines that are reported, detected, or evidenced in it.\n\n"
            f"Allowed candidate lines:\n{self._vocabulary_text}\n\n"
            f"Spectrum Description:\n\"\"\"\n{caption.strip()}\n\"\"\"\n\n"
            "You MUST conclude your response with the exact format:\n"
            "EMISSION LINES: line1, line2, ...\n"
            "If no emission lines from the list are present or mentioned, write:\n"
            "EMISSION LINES: NONE"
        )

    def fallback_tag(self) -> str:
        return "\n\nEMISSION LINES: "

    def default_parse(self, raw_text: str) -> Optional[List[str]]:
        if not raw_text or not raw_text.strip():
            return None

        matches = re.findall(r"EMISSION LINES:\s*(.*)", raw_text, re.IGNORECASE)
        if matches:
            target_str = matches[-1].strip()
        else:
            return None

        if re.search(r"\bNONE\b", target_str, re.IGNORECASE) and not re.search(
            r"[A-Za-z0-9]", target_str.replace("NONE", "").replace("none", "")
        ):
            return []

        raw_tokens = re.split(r"[,;\n]+", target_str)
        extracted = []
        for raw_tok in raw_tokens:
            tok = re.sub(r"^\s*[-*•]\s+", "", raw_tok).strip()
            ck = clean_key(tok)
            if not ck or ck == "none":
                continue
            if ck in CLEAN_TO_CANONICAL:
                can_line = CLEAN_TO_CANONICAL[ck]
                if can_line not in extracted:
                    extracted.append(can_line)

        if not extracted:
            for clean_k, can_name in CLEAN_TO_CANONICAL.items():
                if len(clean_k) >= 3 and clean_k in clean_key(target_str):
                    if can_name not in extracted:
                        extracted.append(can_name)

        return extracted

    def extract_ground_truth(self, item: Any) -> Dict[str, float]:
        if isinstance(item, str):
            eid = item
        elif isinstance(item, (dict, pd.Series)):
            eid = str(item.get("wiki_entity_id", ""))
        else:
            raise ValueError(f"Cannot extract ground truth wiki_entity_id from item of type {type(item)}")
        return self.ground_truth_by_id.get(eid, {})

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "num_canonical_lines": len(self.canonical_lines),
            "canonical_lines": self.canonical_lines,
        }

