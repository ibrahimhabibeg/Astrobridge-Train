import io
import matplotlib.pyplot as plt


def render_spectrum_plot(wavelength, flux, mask=None):
    """Render a 1D spectrum to PNG bytes for vision-language models."""
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

import numpy as np

def subsample_spectrum(wavelength, flux, num_points=100):
    """Downsample a spectrum to evenly-spaced points and format as strings."""
    wavelength = np.array(wavelength).flatten()
    flux = np.array(flux).flatten()
    indices = np.linspace(0, len(wavelength) - 1, num_points, dtype=int)
    w_sub = wavelength[indices]
    f_sub = flux[indices]
    w_str = ", ".join([f"{w:.1f}" for w in w_sub])
    f_str = ", ".join([f"{f:.3f}" for f in f_sub])
    return w_str, f_str

def format_spectrum_text(w_str: str, f_str: str, num_points: int) -> str:
    """Format subsampled spectrum data as a text block for prompts."""
    return (
        f"Spectrum Data ({num_points} evenly spaced points):\n"
        f"Wavelength (Å): [{w_str}]\n"
        f"Flux: [{f_str}]"
    )
