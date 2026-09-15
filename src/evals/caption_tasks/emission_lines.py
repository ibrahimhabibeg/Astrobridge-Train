from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from ..prompts import render_prompt
from .base import CaptionEvalTask

# 14 standard diagnostic lines in the benchmark
CANONICAL_LINES: List[str] = [
    "HALPHA",
    "HBETA",
    "HGAMMA",
    "OIII_5007",
    "OIII_4959",
    "OII_3726",
    "OII_3729",
    "NII_6584",
    "NII_6548",
    "SII_6716",
    "SII_6731",
    "MGII_2796",
    "MGII_2803",
    "CIV_1549",
]

LINE_DISPLAY_NAMES: Dict[str, str] = {
    "HALPHA": "Hα (6563 Å)",
    "HBETA": "Hβ (4861 Å)",
    "HGAMMA": "Hγ (4340 Å)",
    "OIII_5007": "[O III] 5007 Å",
    "OIII_4959": "[O III] 4959 Å",
    "OII_3726": "[O II] 3726 Å",
    "OII_3729": "[O II] 3729 Å",
    "NII_6584": "[N II] 6584 Å",
    "NII_6548": "[N II] 6548 Å",
    "SII_6716": "[S II] 6716 Å",
    "SII_6731": "[S II] 6731 Å",
    "MGII_2796": "Mg II 2796 Å",
    "MGII_2803": "Mg II 2803 Å",
    "CIV_1549": "C IV 1549 Å",
}


def clean_key(s: str) -> str:
    """Normalize line string for robust alias matching."""
    s = s.strip().lower()
    s = (
        s.replace("α", "alpha")
        .replace("β", "beta")
        .replace("γ", "gamma")
        .replace("δ", "delta")
    )
    return re.sub(r"[^a-z0-9]", "", s)


# Direct exact aliases mapping clean_key to canonical line name
_DIRECT_ALIASES: Dict[str, str] = {
    # Hα
    "halpha": "HALPHA",
    "ha": "HALPHA",
    "6563": "HALPHA",
    "balmeralpha": "HALPHA",
    "halpha6563": "HALPHA",
    # Hβ
    "hbeta": "HBETA",
    "hb": "HBETA",
    "4861": "HBETA",
    "balmerbeta": "HBETA",
    "hbeta4861": "HBETA",
    # Hγ
    "hgamma": "HGAMMA",
    "hg": "HGAMMA",
    "4340": "HGAMMA",
    "balmergamma": "HGAMMA",
    "hgamma4340": "HGAMMA",
    # [O III]
    "oiii5007": "OIII_5007",
    "o35007": "OIII_5007",
    "5007": "OIII_5007",
    "oiii4959": "OIII_4959",
    "o34959": "OIII_4959",
    "4959": "OIII_4959",
    # [O II]
    "oii3726": "OII_3726",
    "o23726": "OII_3726",
    "3726": "OII_3726",
    "oii3729": "OII_3729",
    "o23729": "OII_3729",
    "3729": "OII_3729",
    # [N II]
    "nii6584": "NII_6584",
    "n26584": "NII_6584",
    "6584": "NII_6584",
    "nii6583": "NII_6584",
    "6583": "NII_6584",
    "nii6548": "NII_6548",
    "n26548": "NII_6548",
    "6548": "NII_6548",
    # [S II]
    "sii6716": "SII_6716",
    "s26716": "SII_6716",
    "6716": "SII_6716",
    "sii6731": "SII_6731",
    "s26731": "SII_6731",
    "6731": "SII_6731",
    # Mg II
    "mgii2796": "MGII_2796",
    "mg22796": "MGII_2796",
    "2796": "MGII_2796",
    "mgii2803": "MGII_2803",
    "mg22803": "MGII_2803",
    "2803": "MGII_2803",
    # C IV
    "civ1549": "CIV_1549",
    "c41549": "CIV_1549",
    "1549": "CIV_1549",
    "civ": "CIV_1549",
    "c4": "CIV_1549",
}

for line in CANONICAL_LINES:
    _DIRECT_ALIASES[clean_key(line)] = line
    _DIRECT_ALIASES[clean_key(LINE_DISPLAY_NAMES[line])] = line


def _resolve_ambiguous_line(
    token_clean: str, candidate_set: Optional[Set[str]]
) -> List[str]:
    """Resolve generic line mentions (e.g. [O III], [O II], [S II], Mg II) using candidate pool."""
    resolved: List[str] = []

    if token_clean in ("oiii", "o3"):
        if candidate_set:
            if "OIII_5007" in candidate_set:
                resolved.append("OIII_5007")
            if "OIII_4959" in candidate_set:
                resolved.append("OIII_4959")
        if not resolved:
            resolved.append("OIII_5007")

    elif token_clean in ("oii", "o2", "3727", "oii3727"):
        if candidate_set:
            if "OII_3726" in candidate_set:
                resolved.append("OII_3726")
            if "OII_3729" in candidate_set:
                resolved.append("OII_3729")
        if not resolved:
            resolved.extend(["OII_3726", "OII_3729"])

    elif token_clean in ("nii", "n2"):
        if candidate_set:
            if "NII_6584" in candidate_set:
                resolved.append("NII_6584")
            if "NII_6548" in candidate_set:
                resolved.append("NII_6548")
        if not resolved:
            resolved.append("NII_6584")

    elif token_clean in ("sii", "s2", "6720", "sii6720"):
        if candidate_set:
            if "SII_6716" in candidate_set:
                resolved.append("SII_6716")
            if "SII_6731" in candidate_set:
                resolved.append("SII_6731")
        if not resolved:
            resolved.extend(["SII_6716", "SII_6731"])

    elif token_clean in ("mgii", "mg2", "2800", "mgii2800"):
        if candidate_set:
            if "MGII_2796" in candidate_set:
                resolved.append("MGII_2796")
            if "MGII_2803" in candidate_set:
                resolved.append("MGII_2803")
        if not resolved:
            resolved.extend(["MGII_2796", "MGII_2803"])

    return resolved


class CaptionEmissionLineTask(CaptionEvalTask):
    """Evaluates the model's ability to report visible emission lines from a candidate query list."""
    name: str = "caption_emission_lines"
    benchmark_name: str = "emission_lines"

    def __init__(self, **kwargs):
        self.canonical_lines = list(CANONICAL_LINES)
        self.line_display_names = dict(LINE_DISPLAY_NAMES)

    def build_frontier_prompt(
        self, caption: str, item: Optional[Dict[str, Any]] = None
    ) -> str:
        """Builds dynamic prompt containing sample-specific candidate query lines."""
        if item is not None and "candidate_query_lines" in item:
            raw_candidates = item["candidate_query_lines"]
            candidate_keys = [str(k) for k in raw_candidates]
        else:
            candidate_keys = list(self.canonical_lines)

        items_text = "\n".join(
            f"- {self.line_display_names.get(k, k)} [{k}]" for k in candidate_keys
        )
        return render_prompt(
            "caption_eval/emission_lines.jinja2",
            candidate_lines=items_text,
            caption=caption.strip(),
        )

    def fallback_tag(self) -> str:
        return "\n\nEMISSION LINES: "

    def default_parse(
        self, raw_text: str, candidate_lines: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        if not raw_text or not raw_text.strip():
            return None

        cand_set = set(candidate_lines) if candidate_lines else None

        matches = re.findall(r"EMISSION LINES:\s*(.*)", raw_text, re.IGNORECASE)
        if matches:
            target_str = matches[-1].strip()
        else:
            target_str = raw_text.strip()

        if re.search(r"\bNONE\b", target_str, re.IGNORECASE) and not re.search(
            r"[A-Za-z0-9]", target_str.replace("NONE", "").replace("none", "")
        ):
            return []

        raw_tokens = re.split(r"[,;\n]+", target_str)
        extracted: List[str] = []

        for raw_tok in raw_tokens:
            tok = re.sub(r"^\s*[-*•]\s+", "", raw_tok).strip()
            ck = clean_key(tok)
            if not ck or ck == "none":
                continue

            if ck in _DIRECT_ALIASES:
                can_line = _DIRECT_ALIASES[ck]
                if can_line not in extracted:
                    extracted.append(can_line)
            else:
                ambig = _resolve_ambiguous_line(ck, cand_set)
                for line in ambig:
                    if line not in extracted:
                        extracted.append(line)

        # Fallback substring scan if token splitting found nothing
        if not extracted:
            for ck, can_name in _DIRECT_ALIASES.items():
                if len(ck) >= 4 and ck in clean_key(target_str):
                    if can_name not in extracted:
                        extracted.append(can_name)

        if cand_set:
            extracted = [l for l in extracted if l in cand_set]

        return extracted

    def extract_ground_truth(self, item: Any) -> Dict[str, float]:
        """Extracts ground truth dictionary of {detected_line: snr}."""
        if item is None:
            return {}

        if isinstance(item, (dict, pd.Series)):
            gt_obj = item["ground_truth"] if "ground_truth" in item else item
            if isinstance(gt_obj, dict):
                det_lines = gt_obj.get("detected_lines")
                if det_lines is not None:
                    details = gt_obj.get("line_details", {})
                    gt_dict: Dict[str, float] = {}
                    for line in det_lines:
                        line_str = str(line)
                        snr = 1.0
                        if isinstance(details, dict) and line_str in details:
                            d = details[line_str]
                            if isinstance(d, dict) and "snr" in d and d["snr"] is not None:
                                snr = float(d["snr"])
                        gt_dict[line_str] = snr
                    return gt_dict

                # If already dict of {line: snr}
                return {str(k): float(v) for k, v in gt_obj.items() if isinstance(v, (int, float))}

            if isinstance(gt_obj, (list, set)):
                return {str(l): 1.0 for l in gt_obj}

        if isinstance(item, (list, set)):
            return {str(l): 1.0 for l in item}

        return {}

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "num_canonical_lines": len(self.canonical_lines),
            "canonical_lines": self.canonical_lines,
        }
