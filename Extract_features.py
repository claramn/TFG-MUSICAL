"""
extract_features.py
Extrae f0 (CREPE) y loudness (RMS en dB) de NSynth y los guarda como .pt

Fixes respecto al esqueleto original:
  - win_len fijo en ms (50 ms) en vez de hop*2 (que era solo 20 ms y
    provocaba RMS casi constante en 1)
  - Interpolación lineal real de huecos con numpy (no rellenar con F_MIN)
  - Umbral adaptativo de periodicity (busca el más alto que deje ≥30% válidos)
  - Alineación explícita de frames RMS a frames CREPE
  - Nota MIDI del metadata como fallback si CREPE falla completamente
"""

import os
import math
import numpy as np
import torch
import torchaudio
import torchcrepe
import sys
import matplotlib.pyplot as plt
sys.path.append('src')
from src.dataset_og import NSynth

# ── Configuración ──────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

F_MIN       = 32.70    # C1 Hz — límite inferior de CREPE
F_MAX       = 2093.0   # C7 Hz — límite superior
HOP_LENGTH  = 160      # 10 ms a 16 kHz (100 fps) — alinea CREPE y RMS
WIN_MS      = 50       # ventana RMS en milisegundos (independiente del hop)
MODEL       = 'tiny'   # 'full' para más precisión, 'tiny' para velocidad
BATCH_SIZE  = 512

OUTPUT_DIR  = 'data/nsynth_features/train'


# ── Helpers ────────────────────────────────────────────────────────────────────

def choose_threshold(periodicity: torch.Tensor,
                     target_frac: float = 0.30) -> float:
    """
    Devuelve el umbral más alto que deja al menos target_frac de frames válidos.
    Esto evita usar un umbral fijo que silencia instrumentos con baja periodicidad.
    """
    for thr in [0.85, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
        if (periodicity >= thr).float().mean().item() >= target_frac:
            return thr
    return 0.10  # último recurso


def extract_f0(audio: torch.Tensor, sr: int,
               f0_from_midi: float | None = None) -> torch.Tensor:
    """
    Extrae f0 con CREPE y devuelve un tensor (T_frames,) en Hz sin huecos.

    - Usa umbral adaptativo de periodicity
    - Interpola linealmente los huecos (no rellena con F_MIN)
    - Si CREPE falla del todo, usa la nota MIDI del metadata como fallback
    """
    pitch_raw, periodicity = torchcrepe.predict(
        audio.to(device),
        sr,
        HOP_LENGTH,
        F_MIN,
        F_MAX,
        model=MODEL,
        return_periodicity=True,
        batch_size=BATCH_SIZE,
        device=device,
    )

    pitch_raw   = torch.nan_to_num(pitch_raw.squeeze(0).cpu(),   nan=0.0)
    periodicity = torch.nan_to_num(periodicity.squeeze(0).cpu(), nan=0.0)

    thresh = choose_threshold(periodicity)
    valid  = periodicity >= thresh
    n_valid = valid.sum().item()
    n_total = len(valid)

    pitch_np = pitch_raw.numpy().astype(np.float64)

    if n_valid == 0:
        # CREPE no detectó nada: usar nota MIDI si existe
        if f0_from_midi is not None:
            pitch_np[:] = f0_from_midi
        else:
            # Sin información: mantener F_MIN (al menos no es 1 Hz)
            pitch_np[:] = F_MIN
    else:
        # Marcar inválidos y rellenar con interpolación lineal
        pitch_np[~valid.numpy()] = np.nan
        idx = np.arange(n_total)
        not_nan = ~np.isnan(pitch_np)
        pitch_np[np.isnan(pitch_np)] = np.interp(
            idx[np.isnan(pitch_np)],
            idx[not_nan],
            pitch_np[not_nan]
        )

    pitch_np = np.clip(pitch_np, F_MIN, F_MAX)
    return torch.tensor(pitch_np, dtype=torch.float32)   # (T_frames,)


def extract_loudness_db(audio: torch.Tensor, sr: int,
                        n_frames: int) -> torch.Tensor:
    """
    Calcula el RMS por frame con ventana Hann de WIN_MS ms,
    lo convierte a dB y lo recorta / rellena a n_frames para alinear con CREPE.

    win_len se define en ms, NO como hop*2, para que no dependa del hop.
    """
    win_len = int(sr * WIN_MS / 1000)
    win_len += win_len % 2          # asegurar par

    hann_win = torch.hann_window(win_len)
    audio_1d = audio.squeeze(0).cpu()

    # Padding reflect para conservar frames en los bordes
    pad = win_len // 2
    audio_padded = torch.nn.functional.pad(
        audio_1d.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect'
    ).squeeze()

    frames     = audio_padded.unfold(0, win_len, HOP_LENGTH)   # (N, win_len)
    frames_win = frames * hann_win
    rms        = torch.sqrt((frames_win ** 2).mean(dim=-1) + 1e-8)  # (N,)

    # Alinear a n_frames (= frames de CREPE)
    if rms.shape[0] > n_frames:
        rms = rms[:n_frames]
    elif rms.shape[0] < n_frames:
        rms = torch.nn.functional.pad(rms, (0, n_frames - rms.shape[0]))

    # dB — las redes aprenden mucho mejor en escala logarítmica
    loudness_db = 20.0 * torch.log10(rms + 1e-5)
    return loudness_db   # (T_frames,)


def process_and_save(waveform: torch.Tensor, sr: int,
                     metadata: dict, save_path: str) -> None:
    """
    Pipeline completo para un audio: extrae f0 + loudness y guarda el .pt
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Nota MIDI → Hz como fallback para CREPE
    midi = metadata.get('pitch', None)
    f0_from_midi = 440.0 * 2 ** ((midi - 69) / 12.0) if midi is not None else None

    f0          = extract_f0(waveform, sr, f0_from_midi)        # (T,)
    loudness_db = extract_loudness_db(waveform, sr, len(f0))    # (T,)

    torch.save({'f0': f0, 'loudness_db': loudness_db}, save_path)


# ── Bucle principal ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split',  default='training',
                        help='split de NSynth: training | validation | test')
    parser.add_argument('--limit',  type=int, default=5,
                        help='cuántos audios procesar (0 = todos)')
    parser.add_argument('--outdir', default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds    = NSynth(args.split)
    total = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    print(f'device  : {device}')
    print(f'split   : {args.split}')
    print(f'audios  : {total}')
    print(f'outdir  : {args.outdir}')
    print(f'modelo  : {MODEL}  hop={HOP_LENGTH} win={WIN_MS}ms')
    print()

    errors = []
    for idx in range(total):
        sample = ds[idx]
        waveform, sr, key, metadata = sample[:4]
        save_path = os.path.join(args.outdir, f'{key}.pt')

        if os.path.exists(save_path):
            # Checkpointing: no reprocesar si ya existe
            if idx % 500 == 0:
                print(f'[{idx}/{total}] skip (ya existe): {key}')
            continue

        try:
            process_and_save(waveform, sr, metadata, save_path)
            if idx % 100 == 0:
                print(f'[{idx}/{total}] ok: {key}')
        except Exception as e:
            print(f'[{idx}/{total}] ERROR en {key}: {e}')
            errors.append((key, str(e)))

    print(f'\nFinalizado. Errores: {len(errors)}')
    if errors:
        for k, err in errors:
            print(f'  {k}: {err}')
            
            
"""
#esto es pa ver q guarda .pt
archivo_pt = 'data/nsynth_features/train/guitar_acoustic_001-082-050.pt'

# 1. Cargar el archivo
features = torch.load(archivo_pt)

# 2. Ver qué tiene dentro
print("Claves guardadas:", features.keys())

f0 = features['f0']
loudness = features['loudness_db']

print(f"Shape de f0: {f0.shape}")
print(f"Shape de loudness: {loudness.shape}")

# 3. Dibujarlo para asegurarnos de que tiene sentido
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(f0.numpy(), color='royalblue')
axes[0].set_title('Pitch extraído (f0)')
axes[0].set_ylabel('Hz')

axes[1].plot(loudness.numpy(), color='tomato')
axes[1].set_title('Volumen extraído (Loudness en dB)')
axes[1].set_ylabel('dB')
axes[1].set_xlabel('Frames')

plt.tight_layout()
plt.show()
"""
"""
extract_features.py
Extrae f0 (CREPE) y loudness (RMS en dB) de NSynth y los guarda como .pt

Fixes respecto al esqueleto original:
  - win_len fijo en ms (50 ms) en vez de hop*2 (que era solo 20 ms y
    provocaba RMS casi constante en 1)
  - Interpolación lineal real de huecos con numpy (no rellenar con F_MIN)
  - Umbral adaptativo de periodicity (busca el más alto que deje ≥30% válidos)
  - Alineación explícita de frames RMS a frames CREPE
  - Nota MIDI del metadata como fallback si CREPE falla completamente
"""

import os
import math
import numpy as np
import torch
import torchaudio
import torchcrepe
import sys
import matplotlib.pyplot as plt
sys.path.append('src')
from src.dataset_og import NSynth

#  Configuración
device = 'cuda' if torch.cuda.is_available() else 'cpu'

F_MIN       = 32.70    # C1 Hz — límite inferior de CREPE
F_MAX       = 2093.0   # C7 Hz — límite superior
HOP_LENGTH  = 160      # 10 ms a 16 kHz (100 fps) — alinea CREPE y RMS
WIN_MS      = 50       # ventana RMS en milisegundos (independiente del hop)
MODEL       = 'tiny'   # 'full' para más precisión, 'tiny' para velocidad
BATCH_SIZE  = 512

OUTPUT_DIR  = 'data/nsynth_features/train'


#  Helpers 

def choose_threshold(periodicity: torch.Tensor,
                     target_frac: float = 0.30) -> float:
    """
    Devuelve el umbral más alto que deja al menos target_frac de frames válidos.
    Esto evita usar un umbral fijo que silencia instrumentos con baja periodicidad.
    """
    for thr in [0.85, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
        if (periodicity >= thr).float().mean().item() >= target_frac:
            return thr
    return 0.10  # último recurso


def extract_f0(audio: torch.Tensor, sr: int,
               f0_from_midi: float | None = None) -> torch.Tensor:
    """
    Extrae f0 con CREPE y devuelve un tensor (T_frames,) en Hz sin huecos.

    - Usa umbral adaptativo de periodicity
    - Interpola linealmente los huecos (no rellena con F_MIN)
    - Si CREPE falla del todo, usa la nota MIDI del metadata como fallback
    """
    pitch_raw, periodicity = torchcrepe.predict(
        audio.to(device),
        sr,
        HOP_LENGTH,
        F_MIN,
        F_MAX,
        model=MODEL,
        return_periodicity=True,
        batch_size=BATCH_SIZE,
        device=device,
    )

    pitch_raw   = torch.nan_to_num(pitch_raw.squeeze(0).cpu(),   nan=0.0)
    periodicity = torch.nan_to_num(periodicity.squeeze(0).cpu(), nan=0.0)

    thresh = choose_threshold(periodicity)
    valid  = periodicity >= thresh
    n_valid = valid.sum().item()
    n_total = len(valid)

    pitch_np = pitch_raw.numpy().astype(np.float64)

    if n_valid == 0:
        # CREPE no detectó nada: usar nota MIDI si existe
        if f0_from_midi is not None:
            pitch_np[:] = f0_from_midi
        else:
            # Sin información: mantener F_MIN (al menos no es 1 Hz)
            pitch_np[:] = F_MIN
    else:
        # Marcar inválidos y rellenar con interpolación lineal
        pitch_np[~valid.numpy()] = np.nan
        idx = np.arange(n_total)
        not_nan = ~np.isnan(pitch_np)
        pitch_np[np.isnan(pitch_np)] = np.interp(
            idx[np.isnan(pitch_np)],
            idx[not_nan],
            pitch_np[not_nan]
        )

    pitch_np = np.clip(pitch_np, F_MIN, F_MAX)
    return torch.tensor(pitch_np, dtype=torch.float32)   # (T_frames,)


def extract_loudness_db(audio: torch.Tensor, sr: int,
                        n_frames: int) -> torch.Tensor:
    """
    Calcula el RMS por frame con ventana Hann de WIN_MS ms,
    lo convierte a dB y lo recorta / rellena a n_frames para alinear con CREPE.

    win_len se define en ms, NO como hop*2, para que no dependa del hop.
    """
    win_len = int(sr * WIN_MS / 1000)
    win_len += win_len % 2          # asegurar par

    hann_win = torch.hann_window(win_len)
    audio_1d = audio.squeeze(0).cpu()

    # Padding reflect para conservar frames en los bordes
    pad = win_len // 2
    audio_padded = torch.nn.functional.pad(
        audio_1d.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect'
    ).squeeze()

    frames     = audio_padded.unfold(0, win_len, HOP_LENGTH)   # (N, win_len)
    frames_win = frames * hann_win
    rms        = torch.sqrt((frames_win ** 2).mean(dim=-1) + 1e-8)  # (N,)

    # Alinear a n_frames (= frames de CREPE)
    if rms.shape[0] > n_frames:
        rms = rms[:n_frames]
    elif rms.shape[0] < n_frames:
        rms = torch.nn.functional.pad(rms, (0, n_frames - rms.shape[0]))

    # dB — las redes aprenden mucho mejor en escala logarítmica
    loudness_db = 20.0 * torch.log10(rms + 1e-5)
    return loudness_db   # (T_frames,)


def process_and_save(waveform: torch.Tensor, sr: int,
                     metadata: dict, save_path: str) -> None:
    """
    Pipeline completo para un audio: extrae f0 + loudness y guarda el .pt
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Nota MIDI → Hz como fallback para CREPE
    midi = metadata.get('pitch', None)
    f0_from_midi = 440.0 * 2 ** ((midi - 69) / 12.0) if midi is not None else None

    f0          = extract_f0(waveform, sr, f0_from_midi)        # (T,)
    loudness_db = extract_loudness_db(waveform, sr, len(f0))    # (T,)

    torch.save({'f0': f0, 'loudness_db': loudness_db}, save_path)


# Bucle principal

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split',  default='training',
                        help='split de NSynth: training | validation | test')
    parser.add_argument('--limit',  type=int, default=5,
                        help='cuántos audios procesar (0 = todos)')
    parser.add_argument('--outdir', default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds    = NSynth(args.split)
    total = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    print(f'device  : {device}')
    print(f'split   : {args.split}')
    print(f'audios  : {total}')
    print(f'outdir  : {args.outdir}')
    print(f'modelo  : {MODEL}  hop={HOP_LENGTH} win={WIN_MS}ms')
    print()

    errors = []
    for idx in range(total):
        sample = ds[idx]
        waveform, sr, key, metadata = sample[:4]
        save_path = os.path.join(args.outdir, f'{key}.pt')

        if os.path.exists(save_path):
            # Checkpointing: no reprocesar si ya existe
            if idx % 500 == 0:
                print(f'[{idx}/{total}] skip (ya existe): {key}')
            continue

        try:
            process_and_save(waveform, sr, metadata, save_path)
            if idx % 100 == 0:
                print(f'[{idx}/{total}] ok: {key}')
        except Exception as e:
            print(f'[{idx}/{total}] ERROR en {key}: {e}')
            errors.append((key, str(e)))

    print(f'\nFinalizado. Errores: {len(errors)}')
    if errors:
        for k, err in errors:
            print(f'  {k}: {err}')
            
            
"""
#esto es pa ver q guarda .pt
archivo_pt = 'data/nsynth_features/train/guitar_acoustic_001-082-050.pt'

# 1. Cargar el archivo
features = torch.load(archivo_pt)

# 2. Ver qué tiene dentro
print("Claves guardadas:", features.keys())

f0 = features['f0']
loudness = features['loudness_db']

print(f"Shape de f0: {f0.shape}")
print(f"Shape de loudness: {loudness.shape}")

# 3. Dibujarlo para asegurarnos de que tiene sentido
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(f0.numpy(), color='royalblue')
axes[0].set_title('Pitch extraído (f0)')
axes[0].set_ylabel('Hz')

axes[1].plot(loudness.numpy(), color='tomato')
axes[1].set_title('Volumen extraído (Loudness en dB)')
axes[1].set_ylabel('dB')
axes[1].set_xlabel('Frames')

plt.tight_layout()
plt.show()
"""
