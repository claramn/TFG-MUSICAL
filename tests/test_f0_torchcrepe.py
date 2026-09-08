import sys
sys.path.append("src")

import torch
import torchcrepe
from dataset import NSynth

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# cargar una muestra
ds = NSynth("training")
waveform, sr, key, metadata = ds[0]

# convertir a mono si hiciera falta
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio = waveform.to(device)

# hop razonable para pitch tracking
hop_length = int(sr / 200.0)

# rango típico musical
fmin = 32.70
fmax = 1975.53

pitch, periodicity = torchcrepe.predict(
    audio,
    sr,
    hop_length,
    fmin,
    fmax,
    model="tiny",
    return_periodicity=True,
    batch_size=512,
    device=device,
)

print("key:", key)
print("sample rate:", sr)
print("audio shape:", audio.shape)
print("pitch shape:", pitch.shape)
print("periodicity shape:", periodicity.shape)
"""
valid = pitch[pitch > 0]
if valid.numel() > 0:
    print("pitch medio (Hz):", valid.mean().item())
else:
    print("no se detectó pitch válido")
"""
mask = (pitch > 0) & (periodicity > 0.7)
valid = pitch[mask]

if valid.numel() > 0:
    print("pitch medio filtrado (Hz):", valid.mean().item())
    print("pitch mediano filtrado (Hz):", valid.median().item())
else:
    print("no se detectó pitch válido")