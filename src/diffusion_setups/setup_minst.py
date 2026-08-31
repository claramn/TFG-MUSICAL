
import torch
import torch.nn as nn

from src.diffusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MINST_transform(x):
    img, _ = x
    return img

def setup_minst_model(timesteps=1000, channels=32, norm_groups=6, emb_dim=128):
    ## Image Size
    # input_height = 28
    # input_width = 28
    # input_size = (input_height, input_width)

    config = {
        'timesteps': timesteps,
        'channels': channels,
        'norm_groups': norm_groups,
        'emb_dim': emb_dim
    }

    down_layers = [  
        DummyLayer(channels,    channels*2,  norm_groups, emb_dim, skip=True, stride=1).to(device),  # skip: c canales
        DummyLayer(channels*2,  channels*4,  norm_groups, emb_dim, skip=False, stride=1).to(device),  # skip: c*2 canales
        DummyLayer(channels*4,  channels*8,  norm_groups, emb_dim, skip=False, stride=1).to(device),  # skip: c*4 canales
        # DummyLayer(c*8,  c*16, norm_groups, emb_dim, skip=True, stride=2).to(device),  # skip: c*8 canales
    ]

    bottleneck = DummyLayer(channels*8, channels*8, norm_groups, emb_dim).to(device)

    up_layers = [
        # DummyLayer(c*16 + c*8,  c*8, norm_groups, emb_dim, stride=-2).to(device),
        DummyLayer(channels*8,  channels*4, norm_groups, emb_dim, stride=-1).to(device),
        DummyLayer(channels*4,  channels*2, norm_groups, emb_dim, stride=-1).to(device),
        DummyLayer(channels*2 + channels,    channels,   norm_groups, emb_dim, stride=-1).to(device),
    ]

    # EMBEDDER
    embedder = Embedder(num_timesteps=timesteps, embed_dim=emb_dim).to(device)

    # SCHEDULER
    scheduler = Scheduler(num_timesteps=timesteps).to(device)

    # MODEL
    model = DiffusionModel(
        layer_channels=(channels, channels),
        norm_groups=norm_groups,
        up_layers=up_layers,
        down_layers=down_layers,
        bottleneck=bottleneck,
        embedder=embedder, 
        input_channels=1,
        output_channels=1,
        config=config
    ).to(device)
    
    return model, scheduler, MINST_transform
