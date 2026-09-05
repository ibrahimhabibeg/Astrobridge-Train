import sys
import dotenv
import io
import numpy as np
from PIL import Image

from google import genai
from google.genai import types

dotenv.load_dotenv()
sys.path.insert(0, "src")

from evals.data import load_test_spectra
from evals.responders.utils import render_spectrum_plot

# 1. Load the exact first sample that is failing
df_test = load_test_spectra()
row = df_test.iloc[0] # gmw_00000301

spec_data = row["spectrum"]
flux = np.array(spec_data["flux"])
wavelength = np.array(spec_data["lambda"])
mask = np.array(spec_data["mask"]).astype(bool) if "mask" in spec_data else np.zeros_like(flux, dtype=bool)

# 2. Render image
png_bytes = render_spectrum_plot(wavelength, flux, mask=mask, survey="sdss")
img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

# 3. Setup Gemini exactly as in gemini.py
client = genai.Client()
model_name = "gemini-3.5-flash-lite"
prompt = "Provide your classification in the exact format: 'FINAL ANSWER: <label>'."

gen_config = types.GenerateContentConfig(
    max_output_tokens=4096, 
    temperature=0.0
)

print("Sending request to Gemini...")
chat = client.chats.create(model=model_name)
response = chat.send_message([img, prompt], config=gen_config)

# 4. Print the crucial debug info!
print("\n--- DEBUG INFO ---")
print("FINISH REASON:", response.candidates[0].finish_reason)
print("TOKEN USAGE:", response.usage_metadata)
print("RAW TEXT:\n", response.text)