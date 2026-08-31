import json
from config import DATA_PATH
import torchaudio
import torch

NUM_INSTRUMENTS = 11

""" 
cambios respecto al og:
  - process_metadata guarda ahora pitch, velocity y qualities además del one_hot
  - build_condition lee esos campos para construir la bandeja del ConditionalVAE
  - Todo lo demás (load_json, load_raw_waveform, INSTRUMENT_*) queda =
"""
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
PITCH_MAX    = 127.0
VELOCITY_MAX = 127.0
 
# Índices dentro del vector qualities (longitud 10)
# bright=0  dark=1  distortion=2  fast_decay=3  long_release=4
# multiphonic=5  nonlinear_env=6  percussive=7  reverb=8  tempo-synced=9
_IDX_BRIGHT       = 0
_IDX_DARK         = 1
_IDX_LONG_RELEASE = 4



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

def process_metadata(json_data: dict, target_instrument=None) -> dict:
    """
    Procesa el JSON de NSynth y devuelve un dict por sample.
 
    Campos añadidos respecto al original:
      'pitch'      : int    MIDI note 0-127
      'velocity'   : int    0-127
      'qualities'  : list   10 ints (bright, dark, distortion, …)
 
    Campos originales conservados:
      'one_hot_instrument'     : Tensor (11,)
      'instrument_family_id'   : int
      'instrument_family_str'  : str
    """
    target_id      = resolve_instrument_id(target_instrument)
    model_metadata = {}
 
    for key, meta in json_data.items():
        family = int(meta["instrument_family"])
 
        if target_id is not None and family != target_id:
            continue
 
        one_hot = [int(family == i) for i in range(NUM_INSTRUMENTS)]
 
        # qualities puede venir como lista de ints (0/1) o estar ausente
        qualities = list(meta.get("qualities", [0] * 10))
        if len(qualities) < 10:
            qualities += [0] * (10 - len(qualities))
 
        model_metadata[key] = {
            # originales
            "one_hot_instrument":   torch.tensor(one_hot, dtype=torch.float),
            "instrument_family_id": family,
            "instrument_family_str": INSTRUMENT_ID_2_STR[family],
            #nuevos 
            "pitch":    int(meta.get("pitch",    60)),
            "velocity": int(meta.get("velocity", 64)),
            "qualities": qualities,          # lista Python, no tensor
        }
 
    if target_instrument is None:
        print(f"Carga completada: {len(model_metadata)} muestras de todos los instrumentos.")
    else:
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

#lee los campos q process_metadata guarda
def build_condition(metadata: dict) -> dict:
    """
    Construye la bandeja de condición para ConditionalVAE.forward()
    a partir del dict que devuelve process_metadata.
 
    Salida — todos los tensores en float32:
      'instrument_onehot'  (11,)   one-hot de familia
      'pitch_norm'          (1,)   MIDI/127  ∈ [0,1]
      'velocity_norm'       (1,)   vel/127   ∈ [0,1]
      'brightness'          (1,)   derivado de qualities[bright/dark]
      'sustain'             (1,)   derivado de qualities[long_release]
 
    brightness:
      bright=1, dark=0  →  1.0
      bright=0, dark=1  →  0.0
      ninguno/ambos     →  0.5 (neutral)
 
    sustain:
      long_release=1  →  1.0
      long_release=0  →  0.0
    """
    qualities = metadata["qualities"]
 
    bright       = float(qualities[_IDX_BRIGHT])
    dark         = float(qualities[_IDX_DARK])
    long_release = float(qualities[_IDX_LONG_RELEASE])
 
    brightness_val = 0.5 + 0.5 * (bright - dark)   # ∈ {0.0, 0.5, 1.0}
    sustain_val    = long_release                    # ∈ {0.0, 1.0}
 
    return {
        "instrument_onehot": metadata["one_hot_instrument"],          # (11,)  ya es Tensor
        "pitch_norm":        torch.tensor([metadata["pitch"]    / PITCH_MAX],    dtype=torch.float32),
        "velocity_norm":     torch.tensor([metadata["velocity"] / VELOCITY_MAX], dtype=torch.float32),
        "brightness":        torch.tensor([brightness_val], dtype=torch.float32),
        "sustain":           torch.tensor([sustain_val],    dtype=torch.float32),
    }
 