
import torch
import torch.nn as nn

from src.diffusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.utils.models import adjust_shape, compute_magnitude_and_phase, compute_magnitude_and_phase_sin_cos
import torchaudio.transforms as T


class AudioPipeline:
    def __init__(self, stft_transform):
        self.stft_transform = stft_transform

    def __call__(self, x):
        wave, _, _, _, _, _ = x
        wave = wave.to(device)  # <-- mover wave a GPU antes de la STFT
        stft_spec = self.stft_transform(wave)
        del wave
        log_mag, sin, cos = compute_magnitude_and_phase_sin_cos(stft_spec)
        return torch.cat([log_mag, sin, cos], dim=1)  # ya está en device

def setup_audio_model(timesteps=1000, channels=32, norm_groups=6, emb_dim=12, n_fft=1500, hop_length=250, win_length=1500):
    ''' Devuelve el modelo de audio listo para entrenar'''
    
    config = {
        'timesteps': timesteps,
        'channels': channels,
        'norm_groups': norm_groups,
        'emb_dim': emb_dim,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length
    }
    
    
    sample_rate = 16000
    # n_fft = 1500 # DISMINUIR TAMAÑO PARA OPTIMIZAR
    # hop_length = 250
    # win_length = n_fft
    
    sample_rate = 16000
    # n_fft = 1024//2
    # hop_length = 256
    # win_length = 1024//2
    
    # Data pipeline
    stft_transform = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length, 
        hop_length=hop_length,
        power=None, 
        onesided=True,
        center=False
    ).to(device)
    
    
    transform_audio = AudioPipeline(stft_transform)
    
    # Layers
    down_layers = [  
        DummyLayer(channels,    channels*2,  norm_groups, emb_dim, skip=True, stride=1).to(device),  # skip: c canales
        DummyLayer(channels*2,  channels*4,  norm_groups, emb_dim, skip=True, stride=1).to(device),  # skip: c*2 canales
    ]

    bottleneck = DummyLayer(channels*4, channels*4, norm_groups, emb_dim).to(device)

    up_layers = [ # TODO Podria hacer esto con stride = -2? o coger y hacer que si el stride es negativo, aumente manualmente 
        DummyLayer(channels*4 + channels*2, channels*2, norm_groups, emb_dim, stride=1).to(device),
        DummyLayer(channels*2 + channels, channels, norm_groups, emb_dim, stride=1).to(device),
    ]

    # Embedder
    embedder = Embedder(num_timesteps=timesteps, embed_dim=emb_dim).to(device)

    # scheduler
    scheduler = Scheduler(num_timesteps=timesteps).to(device)

    # model
    model = DiffusionModel(
        layer_channels=(channels, channels), # Por ahora los canales de entrada y salida son los mismos
        norm_groups=norm_groups,
        up_layers=up_layers,
        down_layers=down_layers,
        bottleneck=bottleneck,
        embedder=embedder, 
        input_channels=3, ##SINCOS
        output_channels=3, ##SINCOS
        config=config
    ).to(device)
    
    transform_audio = AudioPipeline(stft_transform)
    
    return model, scheduler, transform_audio
