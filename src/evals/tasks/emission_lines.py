import re
from typing import List, Dict, Any, Optional
import pandas as pd

from ..data import load_emission_line_ground_truth

CANONICAL_LINES: List[str] = [
    # "Lyα",
    # "O I 1304",
    "[O II] 3727",
    "Hγ",
    # "[O III] 4363",
    "Hβ",
    "[O III] 5007",
    "[N II] 6583",
    "Hα",
    "[S II] 6720",
    # "[O II] 7325",
]

# Mapping from CSV LINE_NAME to canonical line name
CSV_TO_CANONICAL: Dict[str, str] = {
    # [O II] 3727
    "OII_3726": "[O II] 3727",
    "OII_3729": "[O II] 3727",
    # Hγ
    "HGAMMA": "Hγ",
    "HGAMMA_BROAD": "Hγ",
    # Hβ
    "HBETA": "Hβ",
    "HBETA_BROAD": "Hβ",
    # [O III] 5007
    "OIII_4959": "[O III] 5007",
    "OIII_5007": "[O III] 5007",
    # [N II] 6583
    "NII_6548": "[N II] 6583",
    "NII_6584": "[N II] 6583",
    # Hα
    "HALPHA": "Hα",
    "HALPHA_BROAD": "Hα",
    # [S II] 6720
    "SII_6716": "[S II] 6720",
    "SII_6731": "[S II] 6720",
}

# Inactive line mappings (kept for reference, filtered out below)
_INACTIVE_CSV_MAPPINGS: Dict[str, str] = {
    "LYALPHA": "Lyα",
    "OI_1304": "O I 1304",
    "OIII_4363": "[O III] 4363",
    "OII_7320": "[O II] 7325",
    "OII_7330": "[O II] 7325",
}


def clean_key(s: str) -> str:
    """Normalize line name string for robust alias matching."""
    s = s.strip().lower()
    s = (
        s.replace("α", "alpha")
        .replace("β", "beta")
        .replace("γ", "gamma")
        .replace("δ", "delta")
    )
    for ch in "[](){}*-_,;:. \t\n/\\":
        s = s.replace(ch, "")
    return s


def _build_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}

    for line in CANONICAL_LINES:
        alias_map[clean_key(line)] = line

    for csv_name, canonical in CSV_TO_CANONICAL.items():
        alias_map[clean_key(csv_name)] = canonical

    manual_aliases: Dict[str, str] = {
        # Hα
        "halpha": "Hα",
        "ha": "Hα",
        "6563": "Hα",
        # Hβ
        "hbeta": "Hβ",
        "hb": "Hβ",
        "4861": "Hβ",
        # Hγ
        "hgamma": "Hγ",
        "hg": "Hγ",
        "4340": "Hγ",
        # [O III] 5007
        "oiii": "[O III] 5007",
        "oiii5007": "[O III] 5007",
        "oiii4959": "[O III] 5007",
        "oiii49595007": "[O III] 5007",
        "5007": "[O III] 5007",
        "4959": "[O III] 5007",
        # [O II] 3727
        "oii": "[O II] 3727",
        "oii3727": "[O II] 3727",
        "oii3726": "[O II] 3727",
        "oii3729": "[O II] 3727",
        "3727": "[O II] 3727",
        "3726": "[O II] 3727",
        "3729": "[O II] 3727",
        # [N II] 6583
        "nii": "[N II] 6583",
        "nii6583": "[N II] 6583",
        "nii6584": "[N II] 6583",
        "nii6548": "[N II] 6583",
        "6584": "[N II] 6583",
        "6548": "[N II] 6583",
        # [S II] 6720
        "sii": "[S II] 6720",
        "sii6720": "[S II] 6720",
        "sii6716": "[S II] 6720",
        "sii6731": "[S II] 6720",
        "6716": "[S II] 6720",
        "6731": "[S II] 6720",
    }

    for k, v in manual_aliases.items():
        if v in CANONICAL_LINES:
            alias_map[clean_key(k)] = v

    return alias_map


CLEAN_TO_CANONICAL = _build_alias_map()


class EmissionLineTask:
    name: str = "emission_lines"

    def __init__(self, ground_truth_df: Optional[pd.DataFrame] = None, **kwargs):
        self.canonical_lines = list(CANONICAL_LINES)
        self._vocabulary_text = ", ".join(self.canonical_lines)

        # Pre-load and group ground truth by wiki_entity_id
        if ground_truth_df is None:
            ground_truth_df = load_emission_line_ground_truth()

        self.ground_truth_by_id: Dict[str, Dict[str, float]] = {}
        for _, row in ground_truth_df.iterrows():
            eid = str(row["wiki_entity_id"])
            raw_line = str(row["LINE_NAME"])
            snr = float(row["SNR"])

            if raw_line in CSV_TO_CANONICAL:
                canonical = CSV_TO_CANONICAL[raw_line]
                if eid not in self.ground_truth_by_id:
                    self.ground_truth_by_id[eid] = {}
                # If multiple lines map to same canonical (e.g. doublets/broad), take max SNR
                if (
                    canonical not in self.ground_truth_by_id[eid]
                    or snr > self.ground_truth_by_id[eid][canonical]
                ):
                    self.ground_truth_by_id[eid][canonical] = snr

    def build_prompt(
        self, *, image_mode: bool, spectrum_text: Optional[str] = None
    ) -> str:
        if image_mode:
            intro = "Briefly analyze and describe the given spectrum shown in the image and then identify all visible emission lines present in it."
        elif spectrum_text is not None:
            intro = "Briefly analyze and describe the following spectrum data and then identify all visible emission lines present in it."
        else:
            intro = "Briefly analyze and describe the given spectrum and then identify all visible emission lines present in it."

        parts = [
            intro,
            f"\n\nAllowed candidate lines:\n{self._vocabulary_text}\n",
        ]

        if spectrum_text is not None:
            parts.append(f"\n{spectrum_text}\n")

        parts.append(
            "\nYou MUST conclude your response with the exact format:\n"
            "EMISSION LINES: line1, line2, ...\n"
            "If no emission lines from the list are present, write:\n"
            "EMISSION LINES: NONE"
        )

        return "".join(parts)

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
            # Strip leading bullet points (e.g. "- Halpha" or "* Hbeta")
            tok = re.sub(r"^\s*[-*•]\s+", "", raw_tok).strip()
            ck = clean_key(tok)
            if not ck or ck == "none":
                continue
            if ck in CLEAN_TO_CANONICAL:
                can_line = CLEAN_TO_CANONICAL[ck]
                if can_line not in extracted:
                    extracted.append(can_line)

        # Search the target_str directly for known canonical lines and key aliases
        if not extracted:
            for clean_k, can_name in CLEAN_TO_CANONICAL.items():
                if len(clean_k) >= 3 and clean_k in clean_key(target_str):
                    if can_name not in extracted:
                        extracted.append(can_name)

        return extracted

    def extract_ground_truth(self, item: Any) -> Dict[str, float]:
        """Returns a dict of {canonical_line_name: max_snr} for the observation."""
        if isinstance(item, str):
            eid = item
        elif isinstance(item, (dict, pd.Series)):
            eid = str(item.get("wiki_entity_id", ""))
        else:
            raise ValueError(
                f"Cannot extract ground truth wiki_entity_id from item of type {type(item)}"
            )
        return self.ground_truth_by_id.get(eid, {})

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "num_canonical_lines": len(self.canonical_lines),
            "canonical_lines": self.canonical_lines,
        }
