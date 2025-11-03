from src.utils.dataset import *
from src.utils.audio_utils import *
from torch.utils.data import Dataset

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
