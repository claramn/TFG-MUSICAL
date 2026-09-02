import torch
import sys
sys.path.append("src")
from torch.utils.data import DataLoader
from src.dataset import NSynth  # Asegúrate de que tu clase ya carga los .pt en el __getitem__

# Creamos el dataset y un loader básico
ds = NSynth('training')
loader = DataLoader(ds, batch_size=16, shuffle=True)

# Pillamos el primer batch de 16 audios de golpe
waveform, sr, key, metadata, features = next(iter(loader))

print("=== TEST DE DATALOADER ===")
print(f"Batch Waveform: {waveform.shape}")
print(f"Batch f0: {features['f0'].shape}")
print(f"Batch Loudness: {features['loudness_db'].shape}")