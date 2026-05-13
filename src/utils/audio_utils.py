import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Audio, display
import plotly.express as px
import pandas as pd
import plotly.io as pio
from src.utils.models import adjust_shape, compute_magnitude_and_phase
import torchaudio.transforms

pio.renderers.default = "notebook"


def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.title("Original Waveform")
    plt.plot(waveform[0].numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

# requiere pio.renderers.default = "notebook"
def plot_interactive_waveform(waveform, sample_rate=1, name=""):
    df = pd.DataFrame({"amp": waveform})
    df["secs"] = df.index / sample_rate
    wave = px.line(df, y="amp", title=name)
    wave.show()    

def plot_mel_spectrogram(mel_spec_db):
    plt.figure(figsize=(10, 4))
    plt.title("Mel Spectrogram")
    plt.imshow(mel_spec_db.squeeze(0).numpy(), origin="lower", aspect="auto")
    plt.colorbar(label="dB")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency Bins (Mel)")
    plt.show()

def plot_spectrogram(waveform, sample_rate, n_fft, hop_length):

    ft = np.abs(librosa.stft(waveform, n_fft=n_fft,  hop_length=512))
    librosa.display.specshow(ft, sr=sample_rate, x_axis='time', y_axis='linear')
    plt.colorbar()

    ft_dB = librosa.amplitude_to_db(ft, ref=np.max)
    librosa.display.specshow(ft_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')

    mel_sp = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=2048, hop_length=1024)
    mel_sp = librosa.power_to_db(ft_dB, ref=np.max)
    librosa.display.specshow(mel_sp, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')

    # Compute and plot the STFT spectrogram
    D = librosa.stft(waveform.numpy()[0], n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    plt.title("Spectrogram")
    librosa.display.specshow(D_db, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(label='dB')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def listen(waveform, sample_rate=16000):
    """
    Utility to play audio in a Jupyter notebook.
    waveform: torch.Tensor or np.ndarray, shape [1, time] or [time].
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().cpu().numpy()
    else:
        waveform = waveform.squeeze()
    display(Audio(waveform, rate=sample_rate))  

def waveform_to_spectrogram(waveform, n_fft=1500, hop_length=250, win_length=1500, uselibrosa=False):
    """
    waveform: tensor (B, T) o (1, T)
    returns: tensor (B, 2, 1500, frames)
    """
    if uselibrosa:
        w = waveform.cpu().numpy()[0]  # (T,)
        stft = librosa.stft(w, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, center=False)  # (1500, frames) complejo
        mag   = torch.tensor(np.abs(stft)).unsqueeze(0).unsqueeze(0)    # (1, 1, 1500, frames)
        phase = torch.tensor(np.angle(stft)).unsqueeze(0).unsqueeze(0)  # (1, 1, 1500, frames)
        return torch.cat([mag, phase], dim=1)  # (1, 2, 1500, frames)
    else:
        stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            power=None, onesided=False, center=False
        ).to(waveform.device)
        stft = stft_transform(waveform)
        mag, phase = compute_magnitude_and_phase(stft, normalize=False)
        return torch.cat([mag, phase], dim=1)  # (B, 2, 1500, frames)


def spectrogram_to_waveform(spectrogram, n_fft=1500, hop_length=250, win_length=1500, uselibrosa=False):
    """
    spectrogram: tensor (B, 2, 1500, frames)
    returns: numpy (B, T)
    """
    mag   = spectrogram[:, 0, :, :]  # (B, 1500, frames)
    phase = spectrogram[:, 1, :, :]  # (B, 1500, frames)

    if uselibrosa:
        # librosa trabaja con numpy y sin batch
        mag_np   = mag[0].cpu().numpy()    # (1500, frames)
        phase_np = phase[0].cpu().numpy()  # (1500, frames)
        complex_stft = mag_np * np.exp(1j * phase_np)
        waveform = librosa.istft(complex_stft, hop_length=hop_length,
                                 win_length=win_length, center=False)  # (T,)
        return waveform[:,]  # (T,1) para consistencia
    else:
        istft_transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            onesided=False, center=False
        ).to(spectrogram.device)
        complex_stft = mag * torch.exp(1j * phase)
        waveform = istft_transform(complex_stft)  # (B, T)
        return waveform.detach().cpu().numpy()