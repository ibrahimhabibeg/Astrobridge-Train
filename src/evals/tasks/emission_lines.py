import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
import pandas as pd

from ..data import load_emission_line_ground_truth

# 27 Canonical emission lines ordered by approximate rest wavelength
CANONICAL_LINES: List[str] = [
    "Lyα",
    "N V 1240",
    "O I 1304",
    "Si IV 1396",
    "C IV 1549",
    "He II 1640",
    "Al III 1857",
    "Si III] 1892",
    "C III] 1908",
    "Mg II 2800",
    "[Ne V] 3426",
    "[O II] 3727",
    "[Ne III] 3869",
    "Hδ",
    "Hγ",
    "[O III] 4363",
    "He II 4686",
    "Hβ",
    "[O III] 5007",
    "He I 5876",
    "[O I] 6300",
    "[N II] 6583",
    "Hα",
    "[S II] 6720",
    "[Ar III] 7135",
    "[O II] 7325",
    "[S III] 9069",
]

# Mapping from CSV LINE_NAME to canonical line name
CSV_TO_CANONICAL: Dict[str, str] = {
    # 1. Lyα
    "LYALPHA": "Lyα",
    # 2. N V 1240
    "NV_1240": "N V 1240",
    # 3. O I 1304
    "OI_1304": "O I 1304",
    # 4. Si IV 1396
    "SILIV_1396": "Si IV 1396",
    # 5. C IV 1549
    "CIV_1549": "C IV 1549",
    # 6. He II 1640
    "HEII_1640": "He II 1640",
    # 7. Al III 1857
    "ALIII_1857": "Al III 1857",
    # 8. Si III] 1892
    "SILIII_1892": "Si III] 1892",
    # 9. C III] 1908
    "CIII_1908": "C III] 1908",
    # 10. Mg II 2800
    "MGII_2796": "Mg II 2800",
    "MGII_2803": "Mg II 2800",
    # 11. [Ne V] 3426
    "NEV_3346": "[Ne V] 3426",
    "NEV_3426": "[Ne V] 3426",
    # 12. [O II] 3727
    "OII_3726": "[O II] 3727",
    "OII_3729": "[O II] 3727",
    # 13. [Ne III] 3869
    "NEIII_3869": "[Ne III] 3869",
    # 14. Hδ
    "HDELTA": "Hδ",
    "HDELTA_BROAD": "Hδ",
    # 15. Hγ
    "HGAMMA": "Hγ",
    "HGAMMA_BROAD": "Hγ",
    # 16. [O III] 4363
    "OIII_4363": "[O III] 4363",
    # 17. He II 4686
    "HEII_4686": "He II 4686",
    # 18. Hβ
    "HBETA": "Hβ",
    "HBETA_BROAD": "Hβ",
    # 19. [O III] 5007
    "OIII_4959": "[O III] 5007",
    "OIII_5007": "[O III] 5007",
    # 20. He I 5876
    "HEI_5876": "He I 5876",
    # 21. [O I] 6300
    "OI_6300": "[O I] 6300",
    # 22. [N II] 6583
    "NII_6548": "[N II] 6583",
    "NII_6584": "[N II] 6583",
    # 23. Hα
    "HALPHA": "Hα",
    "HALPHA_BROAD": "Hα",
    # 24. [S II] 6720
    "SII_6716": "[S II] 6720",
    "SII_6731": "[S II] 6720",
    # 25. [Ar III] 7135
    "ARIII_7135": "[Ar III] 7135",
    # 26. [O II] 7325
    "OII_7320": "[O II] 7325",
    "OII_7330": "[O II] 7325",
    # 27. [S III] 9069
    "SIII_6312": "[S III] 9069",
    "SIII_9069": "[S III] 9069",
    "SIII_9532": "[S III] 9069",
}

def clean_key(s: str) -> str:
    """Normalize line name string for robust alias matching."""
    s = s.strip().lower()
    s = s.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma").replace("δ", "delta")
    for ch in "[](){}*-_,;:. \t\n/\\":
        s = s.replace(ch, "")
    return s

def _build_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}

    # Map canonical names and their clean keys
    for line in CANONICAL_LINES:
        alias_map[clean_key(line)] = line

    # Map CSV names
    for csv_name, canonical in CSV_TO_CANONICAL.items():
        alias_map[clean_key(csv_name)] = canonical

    # Add common astronomical aliases
    manual_aliases: Dict[str, str] = {
        # Lyα
        "lymanalpha": "Lyα",
        "lyalpha": "Lyα",
        "lymana": "Lyα",
        "lya": "Lyα",
        "lya1216": "Lyα",
        "lyalpha1216": "Lyα",
        "1216": "Lyα",
        # Hα
        "halpha": "Hα",
        "ha": "Hα",
        "halpha6563": "Hα",
        "ha6563": "Hα",
        "6563": "Hα",
        # Hβ
        "hbeta": "Hβ",
        "hb": "Hβ",
        "hbeta4861": "Hβ",
        "hb4861": "Hβ",
        "4861": "Hβ",
        # Hγ
        "hgamma": "Hγ",
        "hg": "Hγ",
        "hgamma4340": "Hγ",
        "hg4340": "Hγ",
        "4340": "Hγ",
        # Hδ
        "hdelta": "Hδ",
        "hd": "Hδ",
        "hdelta4102": "Hδ",
        "hd4102": "Hδ",
        "4102": "Hδ",
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
        # Mg II 2800
        "mgii": "Mg II 2800",
        "mgii2800": "Mg II 2800",
        "mgii2796": "Mg II 2800",
        "mgii2803": "Mg II 2800",
        "2800": "Mg II 2800",
        "2796": "Mg II 2800",
        "2803": "Mg II 2800",
        # C IV 1549
        "civ": "C IV 1549",
        "civ1549": "C IV 1549",
        "1549": "C IV 1549",
        # C III] 1908
        "ciii": "C III] 1908",
        "ciii1908": "C III] 1908",
        "1908": "C III] 1908",
        # N V 1240
        "nv": "N V 1240",
        "nv1240": "N V 1240",
        "1240": "N V 1240",
        # O I 1304
        "oi1304": "O I 1304",
        "1304": "O I 1304",
        # Si IV 1396
        "siiv": "Si IV 1396",
        "siiv1396": "Si IV 1396",
        "siliv": "Si IV 1396",
        "siliv1396": "Si IV 1396",
        "1396": "Si IV 1396",
        # He II 1640
        "heii1640": "He II 1640",
        "1640": "He II 1640",
        # Al III 1857
        "aliii": "Al III 1857",
        "aliii1857": "Al III 1857",
        "1857": "Al III 1857",
        # Si III] 1892
        "siiii": "Si III] 1892",
        "siiii1892": "Si III] 1892",
        "siliii": "Si III] 1892",
        "siliii1892": "Si III] 1892",
        "1892": "Si III] 1892",
        # [Ne V] 3426
        "nev": "[Ne V] 3426",
        "nev3426": "[Ne V] 3426",
        "nev3346": "[Ne V] 3426",
        "3426": "[Ne V] 3426",
        "3346": "[Ne V] 3426",
        # [Ne III] 3869
        "neiii": "[Ne III] 3869",
        "neiii3869": "[Ne III] 3869",
        "3869": "[Ne III] 3869",
        # [O III] 4363
        "oiii4363": "[O III] 4363",
        "4363": "[O III] 4363",
        # He II 4686
        "heii4686": "He II 4686",
        "4686": "He II 4686",
        # He I 5876
        "hei5876": "He I 5876",
        "5876": "He I 5876",
        # [O I] 6300
        "oi": "[O I] 6300",
        "oi6300": "[O I] 6300",
        "6300": "[O I] 6300",
        # [Ar III] 7135
        "ariii": "[Ar III] 7135",
        "ariii7135": "[Ar III] 7135",
        "7135": "[Ar III] 7135",
        # [O II] 7325
        "oii7325": "[O II] 7325",
        "oii7320": "[O II] 7325",
        "oii7330": "[O II] 7325",
        "7320": "[O II] 7325",
        "7330": "[O II] 7325",
        "7325": "[O II] 7325",
        # [S III] 9069
        "siii": "[S III] 9069",
        "siii9069": "[S III] 9069",
        "siii9532": "[S III] 9069",
        "siii6312": "[S III] 9069",
        "9069": "[S III] 9069",
        "9532": "[S III] 9069",
    }

    for k, v in manual_aliases.items():
        alias_map[clean_key(k)] = v

    return alias_map

CLEAN_TO_CANONICAL = _build_alias_map()

@dataclass
class EmissionLinePromptSpec:
    canonical_lines: List[str]
    output_format_tag: str
    vocabulary_text: str

class EmissionLineTask:
    name: str = "emission_lines"

    def __init__(self, ground_truth_df: Optional[pd.DataFrame] = None, **kwargs):
        self.canonical_lines = list(CANONICAL_LINES)
        vocab_str = ", ".join(self.canonical_lines)
        self.spec = EmissionLinePromptSpec(
            canonical_lines=self.canonical_lines,
            output_format_tag="EMISSION LINES",
            vocabulary_text=vocab_str,
        )

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
                if canonical not in self.ground_truth_by_id[eid] or snr > self.ground_truth_by_id[eid][canonical]:
                    self.ground_truth_by_id[eid][canonical] = snr

    def get_prompt_spec(self) -> EmissionLinePromptSpec:
        return self.spec

    def default_prompt(self, **kwargs) -> str:
        return (
            "Identify all visible emission lines present in the spectrum.\n\n"
            f"Allowed candidate lines:\n{self.spec.vocabulary_text}\n\n"
            "Think step-by-step, but you MUST conclude on the final line with the exact format:\n"
            "EMISSION LINES: line1, line2, line3\n"
            "If no emission lines are visible, conclude with:\n"
            "EMISSION LINES: NONE"
        )

    def default_parse(self, raw_text: str) -> List[str]:
        if not raw_text or not raw_text.strip():
            return []

        # 1. Look for explicit tag
        match = re.search(r"EMISSION LINES:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
        if match:
            target_str = match.group(1).strip()
        else:
            # Fallback: look for "LINES:" or check the full text
            lines_match = re.search(r"\bLINES:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
            if lines_match:
                target_str = lines_match.group(1).strip()
            else:
                target_str = raw_text

        # If it says NONE
        if re.search(r"\bNONE\b", target_str, re.IGNORECASE) and not re.search(r"[A-Za-z0-9]", target_str.replace("NONE", "").replace("none", "")):
            return []

        # Split items by comma, semicolon, or newline
        raw_tokens = re.split(r"[,;\n]+", target_str)
        extracted = []
        for raw_tok in raw_tokens:
            # Strip leading bullet points or numbers (e.g. "- Halpha" or "1. Hbeta")
            tok = re.sub(r"^\s*[-*•\d\.]+\s*", "", raw_tok).strip()
            ck = clean_key(tok)
            if not ck or ck == "none":
                continue
            if ck in CLEAN_TO_CANONICAL:
                can_line = CLEAN_TO_CANONICAL[ck]
                if can_line not in extracted:
                    extracted.append(can_line)

        # Fallback if delimiter splitting didn't find anything:
        # Search the target_str directly for known canonical lines and key aliases
        if not extracted:
            for clean_k, can_name in CLEAN_TO_CANONICAL.items():
                if len(clean_k) >= 3 and clean_k in clean_key(target_str):
                    if can_name not in extracted:
                        extracted.append(can_name)

        return extracted

    def extract_ground_truth(self, item: Any) -> Dict[str, float]:
        """
        Returns a dict of {canonical_line_name: max_snr} for the observation.
        """
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
