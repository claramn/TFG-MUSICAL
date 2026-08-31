import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

from src.dataset import NSynth




SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 80
MAX_FRAMES = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "data/nsynth_mels"



mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
).to(DEVICE)


def waveform_to_mel(waveform):
    """
    waveform: [1, T]

    devuelve:
        [1, N_MELS, MAX_FRAMES]
    """

    # [1, T] -> [1, 1, T]
    waveform = waveform.unsqueeze(0).to(DEVICE)

    mel = mel_transform(waveform)

    # Igual que en tu función original
    mel = torch.log1p(mel)

    # Recortar / rellenar a MAX_FRAMES
    if mel.shape[-1] > MAX_FRAMES:
        mel = mel[..., :MAX_FRAMES]

    elif mel.shape[-1] < MAX_FRAMES:
        mel = F.pad(
            mel,
            (0, MAX_FRAMES - mel.shape[-1])
        )

    # [1, 1, N_MELS, MAX_FRAMES]
    # -> [1, N_MELS, MAX_FRAMES]
    return mel.squeeze(0)




def precompute_partition(partition):

    output_dir = os.path.join(
        OUTPUT_DIR,
        partition
    )

    os.makedirs(output_dir, exist_ok=True)

    dataset = NSynth(
        partition,
        require_features=False
    )

    print()
    print(f"Partition: {partition}")
    print(f"Samples:   {len(dataset)}")
    print(f"Output:    {output_dir}")
    print(f"Device:    {DEVICE}")
    print()

    for i in tqdm(
        range(len(dataset)),
        desc=f"Precomputing {partition}"
    ):

        # Obtenemos exactamente el mismo waveform
        # que utilizas actualmente durante training
        waveform, sr, key, metadata, features, condition = dataset[i]

        output_path = os.path.join(
            output_dir,
            f"{key}.pt"
        )

        # Si ya existe, no lo recalculamos
        if os.path.exists(output_path):
            continue

        mel = waveform_to_mel(waveform)

        # Guardamos en CPU para no ocupar VRAM
        torch.save(
            mel.cpu(),
            output_path
        )


if __name__ == "__main__":

    precompute_partition("training")
    precompute_partition("validation")

    print("\nPrecomputación terminada.")