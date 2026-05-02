import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset import *
from src.utils.audio_utils import *
from torch.utils.data import Dataset
import os
import torch
from src.utils.dataset import (
    load_json,
    process_metadata,
    load_raw_waveform,
    build_condition,
)
 
 
"""
CAMBIOS CLAVE PARA LA ETIQUETACION
etiquetas reales para el conditional vae 
__getitem__ devuelve:
waveform: tensor(1,t)   audio crudo
sample_rate: int
key: str    identificador del sample
metadata: dict  metadata og del json
features: dict  f0 y loudness dsd .pt
condition: dict     bandeja completa lista para el modelo:
    instrument_onehot (11,) float32 familia d instrumento
    pitch_norm      (1,) float32    MIDI/127 ->[0,1]
    velocity_norm   (1,) float32    velocity/127 -> [0,1]
    brightness      (1,) float32     derivado de qualities[0] (bright) y qualities[1] (dark)
    sustain         (1,) float32    derivado d qualities[4] (long realease)
    
Familias de instrumento (instrument_family 0-10):
  0  bass       1  brass      2  flute      3  guitar
  4  keyboard   5  mallet     6  organ      7  reed
  8  string     9  synth_lead 10 vocal
 
Qualities (índices del vector de 10):
  0 bright  1 dark  2 distortion  3 fast_decay  4 long_release
  5 multiphonic  6 nonlinear_env  7 percussive  8 reverb  9 tempo-synced
    
"""

#CONSTANTES NSYNTH
N_FAMILIES    = 11
PITCH_MAX     = 127.0
VELOCITY_MAX  = 127.0
 
#Indices dentro del vector qualities (longitud 10)
IDX_BRIGHT       = 0
IDX_DARK         = 1
IDX_LONG_RELEASE = 4
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'nsynth_features', 'train')
#helpers d condicion
def instrument_to_onehot(family_int: int) -> torch.Tensor:
    """int 0-10  →  FloatTensor (11,)"""
    oh = torch.zeros(N_FAMILIES, dtype=torch.float32)
    oh[int(family_int)] = 1.0
    return oh
 
 
def qualities_to_brightness(qualities: list) -> float:
    """
    Escala continua [0, 1] de brillo derivada de las etiquetas booleanas.
 
    Lógica:
      - 'bright' activo  → +1
      - 'dark'   activo  → -1
      - ninguno / ambos  →  0  (neutral)
    El resultado se normaliza a [0, 1] para que sea compatible con el modelo.
 
    Ejemplos:
      bright=1, dark=0  →  1.0
      bright=0, dark=1  →  0.0
      bright=0, dark=0  →  0.5  (neutral)
      bright=1, dark=1  →  0.5  (señal contradictoria → neutral)
    """
    bright = float(qualities[IDX_BRIGHT])
    dark   = float(qualities[IDX_DARK])
    return float(0.5 + 0.5 * (bright - dark))  # ∈ {0.0, 0.5, 1.0}
 
 
def qualities_to_sustain(qualities: list) -> float:
    """
    Escala continua [0, 1] de sustain derivada de 'long_release'.
    long_release=1  →  1.0  (nota larga, mucho sustain)
    long_release=0  →  0.0
    """
    return float(qualities[IDX_LONG_RELEASE])
 
 
 
def load_features(key: str, features_dir: str = FEATURES_DIR) -> dict:
    """
    Carga el .pt con f0 y loudness_db extraídos por extract_features.py.
    Si el archivo no existe, devuelve tensores vacíos para no bloquear el DataLoader.
    """
    feature_path = os.path.join(features_dir, f'{key}.pt')
    if os.path.exists(feature_path):
        return torch.load(feature_path, weights_only=True)
    # Fallback: tensores de longitud 1 → el training loop debe detectarlo y saltarlo
    return {
        'f0':           torch.zeros(1),
        'loudness_db':  torch.zeros(1),
    }
 
class NSynth(Dataset):
    def __init__(self, partition: str, transform=None,
                 target_instrument=None,
                 features_dir: str = FEATURES_DIR,
                 require_features: bool = False):
 
        self._partition        = partition
        self._transform        = transform
        self._features_dir     = features_dir
        self._require_features = require_features
 
        json_data    = load_json(partition)
        all_metadata = process_metadata(json_data, target_instrument)
 
        if require_features:
            self._metadata = {
                k: v for k, v in all_metadata.items()
                if os.path.exists(os.path.join(features_dir, f'{k}.pt'))
            }
            skipped = len(all_metadata) - len(self._metadata)
            if skipped:
                print(f'[NSynth] require_features=True: '
                      f'{skipped} samples sin .pt omitidos '
                      f'({len(self._metadata)} disponibles)')
        else:
            self._metadata = all_metadata
 
        self._keys = list(self._metadata.keys())
 
    def __len__(self) -> int:
        return len(self._metadata)
 
    def __getitem__(self, index: int):
        key      = self._keys[index]
        metadata = self._metadata[key]
 
        waveform, sr = load_raw_waveform(self._partition, key)
 
        if self._transform:
            waveform = self._transform(waveform)
 
        features  = load_features(key, self._features_dir)
        condition = build_condition(metadata)
 
        return waveform, sr, key, metadata, features, condition
 
 
def nsynth_collate_fn(batch):
    waveforms = torch.stack([b[0] for b in batch])
    srs       = [b[1] for b in batch]
    keys      = [b[2] for b in batch]
    metadatas = [b[3] for b in batch]
 
    f0_list = [b[4]['f0']          for b in batch]
    ld_list = [b[4]['loudness_db'] for b in batch]
 
    if all(t.shape == f0_list[0].shape for t in f0_list):
        features = {
            'f0':          torch.stack(f0_list),
            'loudness_db': torch.stack(ld_list),
        }
    else:
        features = {'f0': f0_list, 'loudness_db': ld_list}
 
    condition = {
        'instrument_onehot': torch.stack([b[5]['instrument_onehot'] for b in batch]),
        'pitch_norm':        torch.stack([b[5]['pitch_norm']        for b in batch]),
        'velocity_norm':     torch.stack([b[5]['velocity_norm']     for b in batch]),
        'brightness':        torch.stack([b[5]['brightness']        for b in batch]),
        'sustain':           torch.stack([b[5]['sustain']           for b in batch]),
    }
 
    return waveforms, srs, keys, metadatas, features, condition
 
 
if __name__ == '__main__':
    from torch.utils.data import DataLoader
 
    ds = NSynth('training', require_features=False)
    print(f'Dataset size: {len(ds)}\n')
 
    waveform, sr, key, metadata, features, condition = ds[0]
 
    print(f'key             : {key}')
    print(f'waveform        : {waveform.shape}   sr={sr}')
    print(f'pitch (MIDI)    : {metadata["pitch"]}  →  pitch_norm={condition["pitch_norm"].item():.4f}')
    print(f'velocity        : {metadata["velocity"]}  →  velocity_norm={condition["velocity_norm"].item():.4f}')
    print(f'instrumento     : {metadata["instrument_family_str"]}')
    print(f'qualities       : {metadata["qualities"]}')
    print(f'brightness      : {condition["brightness"].item():.1f}')
    print(f'sustain         : {condition["sustain"].item():.1f}')
    print(f'f0 shape        : {features["f0"].shape}')
    print(f'loudness shape  : {features["loudness_db"].shape}')
 
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=nsynth_collate_fn)
    wvs, _, _, _, feats, conds = next(iter(loader))
    print(f'\nDataLoader batch:')
    print(f'  waveforms          : {wvs.shape}')
    print(f'  instrument_onehot  : {conds["instrument_onehot"].shape}')
    print(f'  pitch_norm         : {conds["pitch_norm"].shape}')
    print('collate_fn OK ✓')