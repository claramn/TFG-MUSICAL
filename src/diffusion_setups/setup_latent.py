
import torch
import torch.nn as nn

from src.diffusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latent_pipeline(x):
    if isinstance(x, (list, tuple)):
        x = x[0]        # TensorDataset wraps in tuple → [B, 200]
    return x             # [B, 200] ← ya es la forma correcta para el MLP

def setup_latent_model(timesteps=1000, emb_dim=128, hidden_dims=32, latent_dim=200, deep=4, increase_dims=True):
    ''' Devuelve el modelo de audio listo para entrenar'''
    
    config = {
        # Hyperparámetros para referencia futura
        'timesteps': timesteps,
        'emb_dim': emb_dim,
        'hidden_dims': hidden_dims,
        'latent_dim': latent_dim,
        'deep': deep,
        'increase_dims': increase_dims
    }
       
    if isinstance(hidden_dims, int):
        if increase_dims:
            hidden_dims = [hidden_dims * (2 ** i) for i in range(deep)]  # [32, 64, 128, 256] por ejemplo
        else:
            hidden_dims = [hidden_dims for _ in range(deep)]  # [32, 32, 32, 32] por ejemplo
    
    if len(hidden_dims) != deep:
        print('errror')
        
    hidden_dims.insert(0, latent_dim)  # [200, 256, 512, 1024]
    deep = len(hidden_dims) - 1        # 3 bloques

    # Down: 200→256, 256→512, 512→1024
    blocks_down = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_dims[i], hidden_dims[i+1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
        ) for i in range(deep)
    ])
    norms_down = nn.ModuleList([
        nn.LayerNorm(hidden_dims[i]) for i in range(deep)
    ])

    # Up: (1024+512)→512, (512+256)→256, (256+200)→200
    blocks_up = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_dims[i+1] + hidden_dims[i], hidden_dims[i]),
            nn.SiLU(),
            nn.Linear(hidden_dims[i], hidden_dims[i]),
        ) for i in reversed(range(deep))
    ])
    norms_up = nn.ModuleList([
        nn.LayerNorm(hidden_dims[i+1] + hidden_dims[i]) for i in reversed(range(deep))
    ])
    
    # Bottleneck
    bottleneck = BottleneckLatent(hidden_dims[-1], hidden_dims[-1]).to(device)

    # Embedder
    embedder = Embedder(num_timesteps=timesteps, embed_dim=emb_dim).to(device)

    # scheduler
    scheduler = Scheduler(num_timesteps=timesteps).to(device)

    # model
    model = LatentDiffusionMLP(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        blocks_up=blocks_up,
        blocks_down=blocks_down,
        norms_up=norms_up,
        norms_down=norms_down,
        embedder=embedder,
        bottleneck=bottleneck,
        config=config
    ).to(device)
    
    return model, scheduler, latent_pipeline
