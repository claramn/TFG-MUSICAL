import torch
import torch.nn as nn
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scheduler(nn.Module):
    def __init__(self, num_timesteps, beta_init=1e-4, beta_finish=0.02):
        super().__init__()
        beta = torch.linspace(beta_init, beta_finish, num_timesteps)
        alpha = 1 - beta
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))

    def forward(self, t):
        return self.beta[t], self.alpha[t], self.alpha_bar[t]


class Diffuser(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    # x: input image, t: time step
    def forward(self, x, t):
        # e = torch.randn(1, 1, 32, 32) # ruido gaussiano
        e = torch.randn_like(x)  # ruido gaussiano mismo tamaño que x
        beta_t, alpha_t, alpha_bar_t = self.scheduler(t)
        # para que vaya con tensores
        # beta_t = beta_t.view(-1, 1, 1, 1)
        # alpha_t = alpha_t.view(-1, 1, 1, 1)
        # z = torch.sqrt(alpha_t) * x + torch.sqrt(beta_t) * e
    
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e
        return z, e  
    
class Diffuser1D(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    # x: input image, t: time step
    def forward(self, x, t):
        # e = torch.randn(1, 1, 32, 32) # ruido gaussiano
        e = torch.randn_like(x)  # ruido gaussiano mismo tamaño que x
        beta_t, alpha_t, alpha_bar_t = self.scheduler(t)
        # para que vaya con tensores
        # beta_t = beta_t.view(-1, 1, 1, 1)
        # alpha_t = alpha_t.view(-1, 1, 1, 1)
        # z = torch.sqrt(alpha_t) * x + torch.sqrt(beta_t) * e
    
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e
        return z, e      
    
    
class Embedder(nn.Module):
    def __init__(self, num_timesteps, embed_dim):
        super().__init__()
        position = torch.arange(num_timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(num_timesteps, embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', embeddings)

    def forward(self, t):
        return self.embeddings[t]
    
    
class DummyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, time_dim, kernel_size=3, skip=False, stride=1):
        super().__init__()
        
        if stride < 0:
            stride = -stride
            nnConv = nn.ConvTranspose2d
        else:
            nnConv = nn.Conv2d
                    
        self.conv = nn.Sequential(
            nnConv(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride=stride),
            nn.BatchNorm2d(out_channels), ## EN PRINCIPIO ES MEJOR QUE GROUPNORM
            # nn.GroupNorm(norm_groups, out_channels), ###### TODO se puede cambiar por nn.BatchNorm2d(out_channels) para mayor eficiencia
            # nn.ReLU() ###### TODO se puede cambiar por nn.SiLU() para mayor eficiencia
            nn.SiLU()
        )
        
        self.last_silu = nn.SiLU()

        self.res_conv = nnConv(out_channels, out_channels, 3, padding=1, stride=stride)
        
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.channel_proj = nnConv(in_channels, out_channels, 1)
        
        self.skip_conv = None
        if skip:
            self.skip_conv = nnConv(in_channels, in_channels, kernel_size=1, stride=1)  # Adapta los canales del skip a los canales de salida de la capa
    
    def forward(self, x, t_emb):
        
        skip = None
        if self.skip_conv is not None:
            # sin adaprtar el skip, lo sumamos directamente
            skip = self.skip_conv(x)
        
        out = self.conv(x)
        t = self.time_proj(t_emb)
        t = t[:, :, None, None]
        out = out + t
        out = self.res_conv(out)
        # out = out + self.channel_proj(x)  # ← residual connection, was missing
        out = self.last_silu(out)

        return out, skip
    
class Layer1D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, time_dim, kernel_size=3, skip=False, stride=1):
        super().__init__()
        
        if stride < 0:
            stride = -stride
            nnConv = nn.ConvTranspose1d
        else:
            nnConv = nn.Conv1d
                    
        self.conv = nn.Sequential(
            nnConv(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride=stride),
            nn.BatchNorm1d(out_channels), ## EN PRINCIPIO ES MEJOR QUE GROUPNORM
            # nn.GroupNorm(norm_groups, out_channels), ###### TODO se puede cambiar por nn.BatchNorm2d(out_channels) para mayor eficiencia
            # nn.ReLU() ###### TODO se puede cambiar por nn.SiLU() para mayor eficiencia
            nn.SiLU()
        )
        
        self.last_silu = nn.SiLU()

        self.res_conv = nnConv(out_channels, out_channels, 3, padding=1, stride=stride)
        
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.channel_proj = nnConv(in_channels, out_channels, 1)
        
        self.skip_conv = None
        if skip:
            self.skip_conv = nnConv(in_channels, in_channels, kernel_size=1, stride=1)  # Adapta los canales del skip a los canales de salida de la capa
    
    def forward(self, x, t_emb):
        
        skip = None
        if self.skip_conv is not None:
            # sin adaprtar el skip, lo sumamos directamente
            skip = self.skip_conv(x)
        
        out = self.conv(x)
        t = self.time_proj(t_emb)
        t = t[:, :, None]
        out = out + t
        out = self.res_conv(out)
        # out = out + self.channel_proj(x)  # ← residual connection, was missing
        out = self.last_silu(out)

        return out, skip

class NoopLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t_emb):
        return x, x

class DiffusionModel(nn.Module):
    def __init__(self, 
            layer_channels:tuple[int, int], # canales de entrada y salida de las capas de downsampling y upsampling
            norm_groups:int, 
            embedder:nn.Module, 
            down_layers:list[nn.Module],
            bottleneck:nn.Module=None, 
            up_layers:list[nn.Module]=None, 
            input_channels=1, 
            output_channels=1,
            dims=2
        ):
        '''
            - input_channels: canales de entrada 
            - output_channels: canales de salida
            - layer_chanels: 
                -> 0: canales de salida a las capas de downsampling
                -> 1: canales de entrada de las capas de downsampling
            - layers
                * down_layers: capas de downsampling 
                * up_layers: capas de upsampling
                * bottleneck: capa que va a procesar el resultado de las capas de downsampling antes de pasarlo a las capas de upsampling
                    - transforma lc[0] a lc[1]
            - embedder: objeto que va a hacer el embedding del tiempo
            - channels: 
                0. canales de salida de la primera convolucion
                1. canales de salida de la ultima convolucion
        '''
        
        super().__init__()
        self.relu = nn.ReLU()
        self.channels = layer_channels
        
        nnConvxd = nn.Conv1d if dims == 1 else nn.Conv2d

        
        # aparentemente esto va de hacer redes convolucionales
        # self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),

        
        self.conv_in = nn.Sequential(
            nnConvxd(input_channels, layer_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, layer_channels[0]),
            nn.ReLU()
        )
        
        # self.conv_out = nn.Conv2d(channels[1], output_channels, kernel_size=3, padding=1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(norm_groups, layer_channels[1]),
            nn.ReLU(),
            nnConvxd(layer_channels[1], output_channels, kernel_size=3, padding=1),
        )
        
        self.embedder = embedder
        self.down_layers = down_layers
        self.up_layers = up_layers
        self.bottleneck = bottleneck or NoopLayer() # si no se proporciona un bottleneck, se usa una capa identidad
    
    def forward(self, x, t):
        og_size = x.shape[2:]
        
        # B, C, H, W = x.shape # batch, canales, altura, anchura
        
        x = self.conv_in(x) # adapta el tamaño de x a los canales de entrada de las capas
        t_emb = self.embedder(t) # obtiene el embedding del tiempo
        skips = []
        
        ''' Las capas son las encargadas de decidir si aplican o no el skip o los residuales'''
        
        for layer in self.down_layers:
            x, s = layer(x, t_emb) # procesa x con la capa y devuelve el resultado y el skip (si la capa lo tiene)
            skips.append(s) # guarda el skip para usarlo en las capas de upsampling
            
        x, _ = self.bottleneck(x, t_emb) # procesa el resultado de las capas de downsampling con el bottleneck
        
        for i, layer in enumerate(self.up_layers):
            skip = skips.pop() # obtiene el skip correspondiente a la capa
            # print(f"up[{i}] - x: {x.shape}, skip: {skip.shape if skip is not None else None}")

            if skip is not None:
                # TODO getionar esto
                if skip.shape[2:] == x.shape[2:]: # si el skip tiene el mismo número de canales que x, lo sumamos
                    # TODO arreglar esto en las capas 
                    x = torch.cat([x, skip], dim=1) # concatenamos el skip a la entrada de la capa de upsampling
                    # x = x + skip # sumamos el skip a la entrada de la capa de upsampling
                else: # si el skip tiene un número diferente de canales, lo adaptamos con una convolución y luego lo sumamos
                    # TODO borrar:
                    # print('ha sido necesario interpolar el skip')
                    # print(f'size_pre: {skip[2:]}')
                    skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
                    # print(f'size_post: {skip[2:]}')

                    ##### TODO
                    ''' Posible mejora: nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2) en vez de interpolate'''
                    ''' Tmb decia de guardar el skip fuera o algo asi'''

                    x = torch.cat([x, skip], dim=1) # concatenamos el skip a la entrada de la capa de upsampling
                    # print(f'ERROR: skip shape {skip.shape} != x shape {x.shape}')
                # print(f"up[{i}] - x after cat: {x.shape}")
        
                    
            x, _ = layer(x, t_emb) # procesa x con la capa de upsampling
            # print(f"up[{i}] - x after layer: {x.shape}")

            
        x = self.conv_out(x) # adapta el tamaño de x a los canales de salida  
        if x.shape[2:] != og_size:
            x = nn.functional.interpolate(x, size=og_size, mode="bilinear", align_corners=False)
        return x