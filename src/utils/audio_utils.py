import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio

def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.title("Original Waveform")
    plt.plot(waveform[0].numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

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
    return Audio(waveform, rate=sample_rate)