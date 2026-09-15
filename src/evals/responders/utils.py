from __future__ import annotations

import io
import re
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def render_spectrum_plot(wavelength: np.ndarray, flux: np.ndarray, mask: np.ndarray | None = None) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))
    if mask is not None:
        valid = ~mask
        ax.plot(wavelength[valid], flux[valid], color="blue", lw=1, label="Valid")
        ax.plot(wavelength[mask], flux[mask], color="red", lw=1, alpha=0.5, label="Masked")
    else:
        ax.plot(wavelength, flux, color="blue", lw=1)

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title("Spectrum")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def subsample_spectrum(wavelength: np.ndarray, flux: np.ndarray, num_points: int = 100) -> Tuple[str, str]:
    w_arr = np.array(wavelength).flatten()
    f_arr = np.array(flux).flatten()
    indices = np.linspace(0, len(w_arr) - 1, num_points, dtype=int)
    w_sub = w_arr[indices]
    f_sub = f_arr[indices]
    w_str = ", ".join([f"{w:.1f}" for w in w_sub])
    f_str = ", ".join([f"{f:.3f}" for f in f_sub])
    return w_str, f_str


def format_spectrum_text(w_str: str, f_str: str, num_points: int) -> str:
    return (
        f"Spectrum Data ({num_points} evenly spaced points):\n"
        f"Wavelength (Å): [{w_str}]\n"
        f"Flux: [{f_str}]"
    )


def clean_and_extract_caption(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()

    match = re.search(r"\bCAPTION:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    match = re.search(r"\*\*(?:Final\s+)?Caption:?\*\*\s*:?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    match = re.search(r"\*(?:Revised\s+)?Draft(?:\s*\d+)?:\*\s*(?:CAPTION:\s*)?(.*)", text, re.DOTALL | re.IGNORECASE)
    if match and len(match.group(1).strip()) > 20:
        text = match.group(1).strip()

    match = re.search(r"(?:In summary|Summary|Conclusion):\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    stop_patterns = [
        r"\n\s*(?:Note|Explanation|Justification|Alternative interpretation|\d+\.\s*Final Polish|\*Wait).*$",
    ]
    for pattern in stop_patterns:
        text = re.split(pattern, text, flags=re.DOTALL | re.IGNORECASE)[0].strip()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1 and any(
        kw in paragraphs[0].lower()
        for kw in ["the user wants", "analyze the image", "analyze the spectrum", "initial observation", "1. analyze"]
    ):
        for p in reversed(paragraphs):
            if not p.startswith("*") and not p.startswith("#") and not p.startswith("**") and len(p.split()) >= 6:
                return p.strip()

    return text.strip()
