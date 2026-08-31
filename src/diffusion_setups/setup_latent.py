
import torch
import torch.nn as nn

from src.diffusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latent_pipeline(x):
    if isinstance(x, (list, tuple)):
        x = x[0]        # TensorDataset envuelve en tupla → [B, C, H, W]
    return x             # latente espacial del VAE, ya en la forma correcta

def setup_latent_model(timesteps=1000, emb_dim=128, hidden_dims=32, latent_dim=8,
                        norm_groups=6, deep=4, increase_dims=True):
    '''Devuelve el modelo de difusión latente (2D, sobre el espacio del VAE) listo para entrenar'''

    config = {
        'timesteps': timesteps,
        'emb_dim': emb_dim,
        'hidden_dims': hidden_dims,
        'latent_dim': latent_dim,
        'norm_groups': norm_groups,
        'deep': deep,
        'increase_dims': increase_dims,
    }

    if increase_dims:
        ch_progression = [hidden_dims * (2 ** i) for i in range(deep)]  # [32, 64, 128, 256]
    else:
        ch_progression = [hidden_dims for _ in range(deep)]             # [32, 32, 32, 32]

    ######################
    #### OPCION 1 ########
    ######################

    # Down: hidden_dims -> ch_progression[0] -> ch_progression[1] -> ...
    # down_layers = []
    # in_ch = hidden_dims
    # for out_ch in ch_progression:
    #     down_layers.append(DummyLayer(in_ch, out_ch, norm_groups, emb_dim, skip=True, stride=-2).to(device))
    #     in_ch = out_ch
    # down_layers = nn.ModuleList(down_layers)

    # bottleneck = DummyLayer(in_ch, in_ch, norm_groups, emb_dim).to(device)

    # # Up: espejo del down, sumando los canales del skip correspondiente
    # skip_channels = [hidden_dims] + ch_progression[:-1]
    # up_layers = []
    # for out_ch in reversed(skip_channels):
    #     up_layers.append(DummyLayer(in_ch + out_ch, out_ch, norm_groups, emb_dim, stride=-2).to(device))
    #     in_ch = out_ch
    # up_layers = nn.ModuleList(up_layers)
    
    ######################
    #### OPCION 2 ########
    ######################
    
    # Down
    down_layers = []
    in_ch = hidden_dims
    for i, out_ch in enumerate(ch_progression):
        stride = 1 if i == 0 else 1      # ↓ reduce resolución
        down_layers.append(
            DummyLayer(
                in_ch,
                out_ch,
                norm_groups,
                emb_dim,
                skip=True,
                stride=stride
            ).to(device)
        )
        in_ch = out_ch

    down_layers = nn.ModuleList(down_layers)

    bottleneck = DummyLayer(
        in_ch,
        in_ch,
        norm_groups,
        emb_dim,
        stride=1
    ).to(device)

    # Up
    skip_channels = [hidden_dims] + ch_progression[:-1]
    up_layers = []

    for i, out_ch in enumerate(reversed(skip_channels)):
        stride = 1 if i < len(skip_channels)-1 else 1   # ↑ aumenta resolución
        up_layers.append(
            DummyLayer(
                in_ch + out_ch,
                out_ch,
                norm_groups,
                emb_dim,
                stride=stride
            ).to(device)
        )
        in_ch = out_ch

    up_layers = nn.ModuleList(up_layers)

    # Embedder
    embedder = Embedder(num_timesteps=timesteps, embed_dim=emb_dim).to(device)

    # Scheduler
    scheduler = Scheduler(num_timesteps=timesteps).to(device)

    model = DiffusionModel(
        layer_channels=(hidden_dims, hidden_dims),  # conv_in/conv_out: latent_dim <-> hidden_dims
        norm_groups=norm_groups,
        down_layers=down_layers,
        up_layers=up_layers,
        bottleneck=bottleneck,
        embedder=embedder,
        input_channels=latent_dim,
        output_channels=latent_dim,
        config=config
    ).to(device)

    return model, scheduler, latent_pipeline
