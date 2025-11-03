import json
from config import DATA_PATH
import torchaudio
import torch

NUM_INSTRUMENTS = 11

# List of instrument names
INSTRUMENT_ID_2_STR = [
    "bass",         # 0
    "brass",        # 1
    "flute",        # 2
    "guitar",       # 3
    "keyboard",     # 4
    "mallet",       # 5
    "organ",        # 6
    "reed",         # 7
    "string",       # 8
    "synth_lead",   # 9
    "vocal"         # 10
]

# ------------------------------------------------------------------------------
# JSON Loading and Processing
# ------------------------------------------------------------------------------

def load_json(partition):
    """
    Reads `examples.json` from folder:
    e.g. data/training/examples.json
    """
    json_path = DATA_PATH / partition / "examples.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_metadata(json_data):
    model_metadata = {}
    for key, metadata in json_data.items():
        instrument_family = metadata["instrument_family"]
        one_hot = [int(instrument_family == i) for i in range(NUM_INSTRUMENTS)]
        model_metadata[key] = {"one_hot_instrument": torch.tensor(one_hot, dtype=torch.float)}

    
    return model_metadata

def load_raw_waveform(partition, key):
    """
    Loads the raw .wav file for a given key from e.g. data/training/audio/<key>.wav.
    Returns (waveform, sample_rate).
    """
    wav_path = DATA_PATH / partition / "audio" / f"{key}.wav"
    waveform, sr = torchaudio.load(wav_path)
    return waveform, sr

