from src.utils.dataset import *
from src.utils.audio_utils import *
from torch.utils.data import Dataset
import os

class NSynth(Dataset):
    def __init__(self, partition, transform=None):
        self._partition = partition
        self._transform = transform

        json_data = load_json(partition)
        self._metadata = process_metadata(json_data)
        self._keys = list(self._metadata.keys())

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        key = self._keys[index]
        metadata = self._metadata[key]

        # Load the raw .wav file
        waveform, sample_rate = load_raw_waveform(self._partition, key)

        # Apply transformation if any
        if self._transform:
            waveform = self._transform(waveform)

        # Return (waveform, sample_rate, key, metadata)
        return waveform, sample_rate, key, metadata

class LatentNSynth(Dataset):
    # TODO revisar
    ''' Using the VAE to get the latent representation of the audio, and then using that as input for the diffusion model'''
    def __init__(self, nsynth, vae_model_path, stft_transform, device):
        super().__init__()
        self.nsynth = nsynth
        self.stft_transform = stft_transform
        self.device = device

        # Cargar el modelo VAE
        self.vae = torch.load(vae_model_path, map_location=device)
        self.vae.eval()

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, index):
        waveform, sample_rate, key, metadata = self.nsynth[index]
        # Convertir a mono si es estéreo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Normalizar el audio
        waveform = waveform / torch.max(torch.abs(waveform))
        # Calcular el espectrograma
        spectrogram = self.stft_transform(waveform)
        # Obtener magnitud y fase
        # mag, sin, cos = compute_magnitude_and_phase_sin_cos(spectrogram) ## SINCOS
        
        # Usar el vae para obtener la imagen latente
        with torch.no_grad():
            latent = self.vae.encode(spectrogram.to(self.device)) # TODO no estoy seguro de que sea así
        return latent, sample_rate, key, metadata # TODO tampoco estoy seguro de si tengo que devolver el sample_rate, key y metadata, o si solo me interesa el latent para entrenar el modelo de difusión

class LatentDataset(Dataset):
    def __init__(self, directorio, samplear=True):
        """
        directorio: carpeta con los .pt generados
        samplear:   True  → devuelve z = mu + eps*std  (para entrenamiento)
                    False → devuelve mu directamente    (para inferencia/eval)
        """
        self.directorio = directorio
        self.samplear = samplear
        self.archivos = sorted([
            f for f in os.listdir(directorio) if f.endswith(".pt")
        ])

        if not self.archivos:
            raise FileNotFoundError(f"No se encontraron latentes en: {directorio}")

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        datos = torch.load(
            os.path.join(self.directorio, self.archivos[idx]),
            weights_only=True
        )
        mu      = datos["mu"]
        log_var = datos["log_var"]

        if self.samplear:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z
