from src.diffusion_setups.setup_audio import setup_audio_model
from src.diffusion_setups.setup_minst import setup_minst_model
from src.diffusion_setups.setup_latent import setup_latent_model
from src.paths import *
import torch
import json

def load_audio_model():
    config = {}
    with open(PATHS['diffusion']['config'], 'r') as f:
        config = json.load(f) 
    model, scheduler, _ = setup_audio_model(
        timesteps=config['timesteps'],
        channels=config['channels'],
        norm_groups=config['norm_groups'],
        emb_dim=config['emb_dim'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        win_length=config['win_length']
    )
    model = torch.load(PATHS['diffusion']['model'])
    scheduler = torch.load(PATHS['diffusion']['scheduler'])
    model.eval()
    scheduler.eval()
    return model, scheduler

def load_minst_model():
    config = {}
    with open(PATHS['minst_diffusion']['config'], 'r') as f:
        config = json.load(f)  
    model, scheduler, _ = setup_minst_model(
        timesteps=config['timesteps'],
        channels=config['channels'],
        norm_groups=config['norm_groups'],
        emb_dim=config['emb_dim']
    )
    # model.load_state_dict(torch.load(paths['minst']['model']))
    model = torch.load(PATHS['minst_diffusion']['model'])
    scheduler = torch.load(PATHS['minst_diffusion']['scheduler'])
    model.eval()
    scheduler.eval()
    return model, scheduler

def load_latent_model(source='vae_diffusion'):
    ''' Source can be either 'vae_diffusion' or 'cvae_diffusion' '''
    config = {}
    with open(PATHS[source]['config'], 'r') as f:
        config = json.load(f) 
        print(config)   
    model, scheduler, _ = setup_latent_model(
        timesteps=config['timesteps'],
        emb_dim=config['emb_dim'],
        norm_groups=config['norm_groups'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        deep=config['deep'],
        increase_dims=config['increase_dims']
    )
    model = torch.load(PATHS[source]['model'])
    scheduler = torch.load(PATHS[source]['scheduler'])
    model.eval()
    scheduler.eval()
    return model, scheduler