from src.paths import *
import os
import torch

from src.model_cVAE_2D import ConditionalVAE

def load_cVAE(device, path=PATHS['cVAE_2D']['model']):
    model = ConditionalVAE(
        input_size=CVAE_INPUT_SIZE,
        latent_dim=CVAE_LATENT_DIM,
        channels=CVAE_CHANNELS,
        condition_dim=CVAE_CONDITION_DIM,
        n_frames=CVAE_MAX_FRAMES,
        n_harmonics=CVAE_N_HARMONICS,
        ddsp_hidden=CVAE_DDSP_HIDDEN,
    ).to(device)
    
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model file {path} not found. Returning uninitialized model.")
    model.eval()
    return model

from src.latent_models import latent_VAE

def load_VAE(device, path=PATHS['VAE_2D']['model']):
    model = latent_VAE(
        input_size=VAE_INPUT_SIZE,
        latent_dim=VAE_LATENT_DIM,
        channels=VAE_CHANNELS,
        strides=VAE_STRIDES,
    ).to(device)
    
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model file {path} not found. Returning uninitialized model.")
    model.eval()
    return model

from src.latent_models import latent_AutoEncoder

def load_AE(device, path=PATHS['autoencoder_2D']['model']):
    model = latent_AutoEncoder(
        input_size=AE_INPUT_SIZE,
        latent_dim=AE_LATENT_DIM,
        channels=AE_CHANNELS,
        variational=False
    ).to(device)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model file {path} not found. Returning uninitialized model.")
    model.eval()
    return model
    