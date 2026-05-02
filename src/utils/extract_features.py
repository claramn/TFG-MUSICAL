"""
extract_features_v2.py

Precomputa features para NSynth y las guarda como .pt por sample.

Guarda por cada audio:
    {
        "f0": Tensor(T_frames),
        "loudness_db": Tensor(T_frames)
    }

Ejemplos:
    python extract_features_v2.py --split training --limit 100
    python extract_features_v2.py --split training --limit 0
    python extract_features_v2.py --split validation --limit 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchcrepe


# ---------------------------------------------------------------------
# Paths robustos
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset_og import NSynth  # devuelve waveform, sr, key, metadata


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

F_MIN = 32.70      # C1
F_MAX = 2093.0     # C7
HOP_LENGTH = 160   # 10 ms a 16 kHz
WIN_MS = 50        # ventana RMS
MODEL = "tiny"     # "tiny" rápido, "full" más preciso
BATCH_SIZE = 512


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def choose_threshold(periodicity: torch.Tensor, target_frac: float = 0.30) -> float:
    """
    Elige el umbral de periodicity más alto que deja suficientes frames válidos.
    Evita que instrumentos difíciles se queden sin f0.
    """
    for thr in [0.85, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
        if (periodicity >= thr).float().mean().item() >= target_frac:
            return thr
    return 0.10


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def extract_f0(
    waveform: torch.Tensor,
    sr: int,
    device: str,
    f0_fallback: float | None = None,
) -> torch.Tensor:
    """
    Extrae f0 con torchcrepe.
    Devuelve Tensor(T_frames) en Hz.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    with torch.no_grad():
        pitch_raw, periodicity = torchcrepe.predict(
            waveform,
            sr,
            HOP_LENGTH,
            F_MIN,
            F_MAX,
            model=MODEL,
            return_periodicity=True,
            batch_size=BATCH_SIZE,
            device=device,
        )

    pitch_raw = torch.nan_to_num(pitch_raw.squeeze(0).detach().cpu(), nan=0.0)
    periodicity = torch.nan_to_num(periodicity.squeeze(0).detach().cpu(), nan=0.0)

    valid = periodicity >= choose_threshold(periodicity)
    pitch_np = pitch_raw.numpy().astype(np.float64)

    if valid.sum().item() == 0:
        if f0_fallback is not None:
            pitch_np[:] = f0_fallback
        else:
            pitch_np[:] = F_MIN
    else:
        valid_np = valid.numpy()
        pitch_np[~valid_np] = np.nan

        idx = np.arange(len(pitch_np))
        not_nan = ~np.isnan(pitch_np)

        pitch_np[np.isnan(pitch_np)] = np.interp(
            idx[np.isnan(pitch_np)],
            idx[not_nan],
            pitch_np[not_nan],
        )

    pitch_np = np.clip(pitch_np, F_MIN, F_MAX)
    return torch.tensor(pitch_np, dtype=torch.float32)


def extract_loudness_db(
    waveform: torch.Tensor,
    sr: int,
    n_frames: int,
) -> torch.Tensor:
    """
    Calcula loudness por frame usando RMS en dB.
    Devuelve Tensor(T_frames).
    """
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = waveform.detach().cpu()

    win_len = int(sr * WIN_MS / 1000)
    if win_len % 2 == 1:
        win_len += 1

    pad = win_len // 2
    hann = torch.hann_window(win_len)

    padded = torch.nn.functional.pad(
        waveform.view(1, 1, -1),
        (pad, pad),
        mode="reflect",
    ).view(-1)

    frames = padded.unfold(0, win_len, HOP_LENGTH)
    frames = frames * hann

    rms = torch.sqrt((frames ** 2).mean(dim=-1) + 1e-8)

    if rms.shape[0] > n_frames:
        rms = rms[:n_frames]
    elif rms.shape[0] < n_frames:
        rms = torch.nn.functional.pad(rms, (0, n_frames - rms.shape[0]))

    loudness_db = 20.0 * torch.log10(rms + 1e-5)
    return loudness_db.float()


def process_one_sample(waveform, sr, metadata, device):
    midi = metadata.get("pitch", None)
    fallback = midi_to_hz(int(midi)) if midi is not None else None

    f0 = extract_f0(waveform, sr, device=device, f0_fallback=fallback)
    loudness_db = extract_loudness_db(waveform, sr, n_frames=len(f0))

    return {
        "f0": f0,
        "loudness_db": loudness_db,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="training", choices=["training", "validation", "test"])
    parser.add_argument("--limit", type=int, default=20, help="0 = todos")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--instrument", default=None, help="opcional: guitar, bass, string...")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.outdir is None:
        split_name = "train" if args.split == "training" else args.split
        outdir = PROJECT_ROOT / "data" / "nsynth_features" / split_name
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    ds = NSynth(args.split)
    total = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    print("device :", device)
    print("split  :", args.split)
    print("samples:", total)
    print("outdir :", outdir)
    print("model  :", MODEL)
    print()

    errors = []

    for idx in range(total):
        waveform, sr, key, metadata = ds[idx]

        if args.instrument is not None:
            instr = metadata.get("instrument_family_str", "")
            if str(instr).lower() != args.instrument.lower():
                continue

        save_path = outdir / f"{key}.pt"

        if save_path.exists() and not args.overwrite:
            if idx % 500 == 0:
                print(f"[{idx}/{total}] skip: {key}")
            continue

        try:
            feats = process_one_sample(waveform, sr, metadata, device)
            torch.save(feats, save_path)

            if idx % 100 == 0:
                print(
                    f"[{idx}/{total}] ok: {key} "
                    f"f0={tuple(feats['f0'].shape)} "
                    f"loudness={tuple(feats['loudness_db'].shape)}"
                )

        except Exception as exc:
            print(f"[{idx}/{total}] ERROR {key}: {exc}")
            errors.append((key, str(exc)))

    print()
    print("Finalizado.")
    print("Errores:", len(errors))

    if errors:
        print("Primeros errores:")
        for key, err in errors[:10]:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
