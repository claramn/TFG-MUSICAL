from src.utils.dataset import *
from src.utils.audio_utils import *
from torch.utils.data import Dataset
import os

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