import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import sys
sys.path.append("src")

from dataset import NSynth
from models2 import ConditionalVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando: {device}")

#cargamos dataset y sacamos UN batch
print("1. Cargando un batch del dataset...")
ds = NSynth('training')
loader = DataLoader(ds, batch_size=4, shuffle=True)
waveform, sr, key, metadata, features = next(iter(loader))

#pasamos el Audio Raw a Mel-Spectrogram
print("2. Transformando audio a Mel-spectrogram...")
mel_transform = T.MelSpectrogram(
    sample_rate=16000, 
    n_fft=1024, 
    hop_length=160, 
    n_mels=80
)
mel = mel_transform(waveform)

# OJO: NSynth dura 4s (400 frames). Para este test, lo recortamos a 128 frames 
# para que coincida con el input_size=(80,128) por defecto del modelo.
mel = mel[:, :, :, :128].to(device)

#Preparamos las etiquetas falsas (simulando que ya las hemos procesado)
# Más adelante sacaremos esto del 'metadata' y de 'features'
B = waveform.shape[0]
import torch.nn.functional as F
inst_oh = F.one_hot(torch.randint(0, 11, (B,)), num_classes=11).float().to(device)
pitch_n = torch.rand(B, 1).to(device)
vel_n = torch.rand(B, 1).to(device)
bright = torch.rand(B, 1).to(device)
sustain = torch.rand(B, 1).to(device)

#cargamos el Ferrari y le damos al contacto
print("3. Pasando todo por la Conditional VAE...")
model = ConditionalVAE(channels = [1,32,64,128,256]).to(device)

mel_hat, ddsp_params, kl = model(mel, inst_oh, pitch_n, vel_n, bright, sustain)

print("\nTodo conectado perfecto.")
print(f"Salida Mel_hat    : {mel_hat.shape}")
print(f"Salida f0_scale   : {ddsp_params['f0_scale'].shape}")