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
INSTRUMENT_STR_2_ID = {name: idx for idx, name in enumerate(INSTRUMENT_ID_2_STR)}

# ------------------------------------------------------------------------------
# JSON Loading and Processing
# ------------------------------------------------------------------------------

def resolve_instrument_id(instrument):
    """
    Resolve an instrument selector to a numeric NSynth instrument_family id.

    Accepted inputs:
      - None: no filtering
      - int: direct instrument id in [0, NUM_INSTRUMENTS-1]
      - str: instrument family name (e.g. "guitar", "string")
    """
    if instrument is None:
        return None

    if isinstance(instrument, int):
        if 0 <= instrument < NUM_INSTRUMENTS:
            return instrument
        raise ValueError(
            f"instrument id must be in [0, {NUM_INSTRUMENTS - 1}], got {instrument}"
        )

    if isinstance(instrument, str):
        name = instrument.strip().lower()
        aliases = {
            "strings": "string",
            "guitars": "guitar",
        }
        name = aliases.get(name, name)
        if name in INSTRUMENT_STR_2_ID:
            return INSTRUMENT_STR_2_ID[name]
        valid = ", ".join(INSTRUMENT_ID_2_STR)
        raise ValueError(f"Unknown instrument '{instrument}'. Valid names: {valid}")

    raise TypeError("instrument must be None, int, or str")

def load_json(partition):
    """
    Reads `examples.json` from folder:
    e.g. data/training/examples.json
    """
    json_path = DATA_PATH / partition / "examples.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_metadata(json_data,target_instrument=None):
    target_id = resolve_instrument_id(target_instrument)
    model_metadata = {}
    for key, metadata in json_data.items():
        instrument_family = int(metadata["instrument_family"])
        
        # Si hemos definido un target_id y no coincide, saltamos esta muestra
        if target_id is not None and instrument_family != target_id:
            continue
            
        one_hot = [int(instrument_family == i) for i in range(NUM_INSTRUMENTS)]
        model_metadata[key] = {
            "one_hot_instrument": torch.tensor(one_hot, dtype=torch.float),
            "instrument_family_id": instrument_family,
            "instrument_family_str": INSTRUMENT_ID_2_STR[instrument_family],
        }

    print(f"Filtrado completado: {len(model_metadata)} muestras de '{target_instrument}' cargadas.")
    return model_metadata

def load_raw_waveform(partition, key):
    """
    Loads the raw .wav file for a given key from e.g. data/training/audio/<key>.wav.
    Returns (waveform, sample_rate).
    """
    wav_path = DATA_PATH / partition / "audio" / f"{key}.wav"
    waveform, sr = torchaudio.load(wav_path)
    return waveform, sr

