import io
import matplotlib.pyplot as plt

def render_spectrum_plot(wavelength, flux, mask=None, survey=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if mask is not None:
        valid = ~mask
        ax.plot(wavelength[valid], flux[valid], color='blue', lw=1, label='Valid')
        ax.plot(wavelength[mask], flux[mask], color='red', lw=1, alpha=0.5, label='Masked')
    else:
        ax.plot(wavelength, flux, color='blue', lw=1)
        
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title('Spectrum')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
