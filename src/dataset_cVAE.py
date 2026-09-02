import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset import *
from src.utils.audio_utils import *
from torch.utils.data import Dataset
import os

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
        self._mel_dir = os.path.join(
            PROJECT_ROOT,
            'data',
            'nsynth_mels',
            partition
        )
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
 
    """
    def __getitem__(self, index: int):
        key      = self._keys[index]
        metadata = self._metadata[key]
 
        waveform, sr = load_raw_waveform(self._partition, key)
 
        if self._transform:
            waveform = self._transform(waveform)
        mel_path = os.path.join(
            self._mel_dir,
            f"{key}.pt"
        )

        mel = torch.load(
            mel_path,
            map_location="cpu",
            weights_only=True
        )

        features  = load_features(key, self._features_dir)
        condition = build_condition(metadata)
 
        return waveform, mel, sr, key, metadata, features, condition
 
        
        """
        
    def __getitem__(self, index: int):
        key = self._keys[index]
        metadata = self._metadata[key]

        mel_path = os.path.join(
            self._mel_dir,
            f"{key}.pt"
        )

        mel = torch.load(
            mel_path,
            map_location="cpu",
            weights_only=True
        )

        features = load_features(
            key,
            self._features_dir
        )

        condition = build_condition(metadata)

        return mel, key, metadata, features, condition
        
        """
def nsynth_collate_fn(batch):

    waveforms = torch.stack([b[0] for b in batch])

    mels = torch.stack([b[1] for b in batch])

    srs = [b[2] for b in batch]
    keys = [b[3] for b in batch]
    metadatas = [b[4] for b in batch]

    f0_list = [b[5]['f0'] for b in batch]
    ld_list = [b[5]['loudness_db'] for b in batch]

    if all(t.shape == f0_list[0].shape for t in f0_list):

        features = {
            'f0': torch.stack(f0_list),
            'loudness_db': torch.stack(ld_list),
        }

    else:

        features = {
            'f0': f0_list,
            'loudness_db': ld_list
        }

    condition = {
        'instrument_onehot': torch.stack(
            [b[6]['instrument_onehot'] for b in batch]
        ),
        'pitch_norm': torch.stack(
            [b[6]['pitch_norm'] for b in batch]
        ),
        'velocity_norm': torch.stack(
            [b[6]['velocity_norm'] for b in batch]
        ),
        'brightness': torch.stack(
            [b[6]['brightness'] for b in batch]
        ),
        'sustain': torch.stack(
            [b[6]['sustain'] for b in batch]
        ),
    }
         

    return (
        waveforms,
        mels,
        srs,
        keys,
        metadatas,
        features,
        condition
    )
        """
def nsynth_collate_fn(batch):

    # b[0] = mel
    mels = torch.stack([b[0] for b in batch])

    # b[1] = key
    keys = [b[1] for b in batch]

    # b[2] = metadata
    metadatas = [b[2] for b in batch]

    # b[3] = features
    f0_list = [b[3]['f0'] for b in batch]
    ld_list = [b[3]['loudness_db'] for b in batch]

    if all(t.shape == f0_list[0].shape for t in f0_list):
        features = {
            'f0': torch.stack(f0_list),
            'loudness_db': torch.stack(ld_list),
        }
    else:
        features = {
            'f0': f0_list,
            'loudness_db': ld_list,
        }

    # b[4] = condition
    conditions = {
        'instrument_onehot': torch.stack(
            [b[4]['instrument_onehot'] for b in batch]
        ),
        'pitch_norm': torch.stack(
            [b[4]['pitch_norm'] for b in batch]
        ),
        'velocity_norm': torch.stack(
            [b[4]['velocity_norm'] for b in batch]
        ),
        'brightness': torch.stack(
            [b[4]['brightness'] for b in batch]
        ),
        'sustain': torch.stack(
            [b[4]['sustain'] for b in batch]
        ),
    }

    return (
        mels,
        keys,
        metadatas,
        features,
        conditions,
    )

class LatentBatchDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir):
        self.files = sorted(
            [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')]
        )
        # Índice: para cada muestra global, sabemos en qué archivo y qué posición está
        self.index = []
        for file_idx, f in enumerate(self.files):
            n = torch.load(f, map_location='cpu', weights_only=True).shape[0]
            self.index.extend([(file_idx, i) for i in range(n)])
        self._cache_file_idx = None
        self._cache_tensor = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, pos = self.index[idx]
        if file_idx != self._cache_file_idx:
            self._cache_tensor = torch.load(self.files[file_idx], map_location='cpu')
            self._cache_file_idx = file_idx
        return self._cache_tensor[pos]
    
import os
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info

class ShardedLatentDataset(IterableDataset):
    '''
    Dataset de latentes troceados en varios .pt (shards).

    - Cada shard se carga del disco una sola vez por época (no N veces por muestra).
    - Los shards se reparten entre workers del DataLoader (sin duplicar trabajo).
    - Shuffle real: se mezcla el orden de los shards + un buffer que combina
      varios shards cargados a la vez, para que muestras de distintos shards
      se intercalen (en vez de servir un shard entero seguido).
    '''

    def __init__(self, save_dir, shuffle=True, shard_buffer=2, seed=0):
        self.files = sorted(
            os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')
        )
        self.shuffle = shuffle
        self.shard_buffer = max(1, shard_buffer)  # nº de shards mezclados a la vez
        self.epoch = 0
        self.seed = seed

        # tamaño total, solo para poder usar len() si hace falta (p.ej. logging)
        self._len = sum(
            torch.load(f, map_location='cpu', mmap=True, weights_only=True).shape[0]
            for f in self.files
        )

    def __len__(self):
        return self._len

    def set_epoch(self, epoch):
        # llamar al principio de cada época para que el shuffle cambie entre épocas
        self.epoch = epoch

    def _shard_order(self):
        files = list(self.files)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(files)
        return files

    def __iter__(self):
        worker_info = get_worker_info()
        files = self._shard_order()

        # reparte shards entre workers (cada worker procesa un subconjunto disjunto)
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        rng = random.Random(self.seed + self.epoch + (worker_info.id if worker_info else 0))

        buffer = []  # buffer de muestras mezclando varios shards
        pending_shards = list(files)

        def load_shard(path):
            t = torch.load(path, map_location='cpu', weights_only=True)
            idxs = list(range(t.shape[0]))
            if self.shuffle:
                rng.shuffle(idxs)
            return t, idxs

        loaded = []  # lista de (tensor, idxs_restantes) cargados actualmente

        while pending_shards or loaded or buffer:
            # rellena el pool de shards cargados hasta shard_buffer
            while pending_shards and len(loaded) < self.shard_buffer:
                loaded.append(load_shard(pending_shards.pop()))

            if not loaded:
                break

            # elige un shard al azar del pool cargado y saca una muestra
            shard_i = rng.randrange(len(loaded)) if self.shuffle else 0
            tensor, idxs = loaded[shard_i]
            pos = idxs.pop()
            yield tensor[pos]

            if not idxs:
                loaded.pop(shard_i)  # shard agotado, se libera de memoria
