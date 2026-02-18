import torch
import torch.nn as nn
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scheduler(nn.Module):
    def __init__(self, num_epochs, device, beta_init=1e-4, beta_finish=0.02):
        super().__init__()
        self.beta = torch.linspace(beta_init, beta_finish, num_epochs, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

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
    
    
class Embeder(nn.Module):
    def __init__(self, num_epochs, embed_dim, device):
        super().__init__()
        position = torch.arange(num_epochs, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(num_epochs, embed_dim, device=device)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings
        self.device = device


    def forward(self, t):
        # t = t.long().view(-1) esto hace que t sea un tensor
        embeds = self.embeddings[t].to(self.device)
        return embeds[:, :, None, None]
    
    
class DummyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, stride=1, size_out=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU()
        )
        self.res_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.size_out = size_out

    def forward(self, x, t_emb):
        out = self.conv(x)
        # res = self.res_conv(out)
        if self.size_out is not None:
        
        return out, x

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
            output_channels=1
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
        
        # aparentemente esto va de hacer redes convolucionales
        # self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),

        
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, layer_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, layer_channels[0]),
            nn.ReLU()
        )
        
        # self.conv_out = nn.Conv2d(channels[1], output_channels, kernel_size=3, padding=1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(norm_groups, layer_channels[1]),
            nn.ReLU(),
            nn.Conv2d(layer_channels[1], output_channels, kernel_size=3, padding=1),
        )
        
        self.embeder = embedder
        self.down_layers = down_layers
        self.up_layers = up_layers
        self.bottleneck = bottleneck or nn.AdaptiveAvgPool2d((layer_channels[0], layer_channels[1]))
        
    def forward(self, x, t):
        return self.true_forward(x, t)
    
    def true_forward(self, x, t):
        # B, C, H, W = x.shape # batch, canales, altura, anchura
        
        x = self.conv_in(x) # adapta el tamaño de x a los canales de entrada de las capas
        t_emb = self.embeder(t) # obtiene el embedding del tiempo
        skips = []
        
        ''' Las capas son las encargadas de decidir si aplican o no el skip o los residuales'''
        
        for layer in self.down_layers:
            x, s = layer(x, t_emb) # procesa x con la capa y devuelve el resultado y el skip (si la capa lo tiene)
            skips.append(s) # guarda el skip para usarlo en las capas de upsampling
            
        x, _ = self.bottleneck(x, t_emb) # procesa el resultado de las capas de downsampling con el bottleneck
        
        for i, layer in enumerate(self.up_layers):
            skip = skips.pop() # obtiene el skip correspondiente a la capa
            if skip is not None:
                if skip.shape[1] == x.shape[1]: # si el skip tiene el mismo número de canales que x, lo sumamos
                    x = torch.cat([x, skip], dim=1) # concatenamos el skip a la entrada de la capa de upsampling
                else: # si el skip tiene un número diferente de canales, lo adaptamos con una convolución y luego lo sumamos
                    print(f'ERROR: skip shape {skip.shape} != x shape {x.shape}')            
            x, _ = layer(x, t_emb) # procesa x con la capa de upsampling
            
        return self.conv_out(x) # adapta el tamaño de x a los canales de salida  
    
    def forward_nores(self, x, t):
        B, C, H, W = x.shape # batch, canales, altura, anchura
        x = self.conv_in(x)
        t_emb = self.embeder(x, t)
        for layer in self.down_layers:
            x, r = layer(x, t_emb)
        for layer in self.up_layers:
            x, r = layer(x, t_emb)
        return self.conv_out(x)
        
    def forward_withres(self, x, t):
        B, C, H, W = x.shape # batch, canales, altura, anchura
        
        x = self.conv_in(x)
        t_emb = self.embeder(x, t)
        residuals = []
        for layer in self.down_layers:
            x, r = layer(x, t_emb)
            residuals.append(r)
        residuals = residuals[::-1] 
        x, _ = self.bottleneck(x, t_emb)
        # residuals.append(r)
        for layer in self.up_layers:
            x = torch.concat((layer(x, t_emb)[0], residuals.pop()), dim=1)
        x = self.relu(x)
        return self.conv_out(x)
    
    def forward_withskip(self, x, t):
        B, C, H, W = x.shape # batch, canales, altura, anchura
        
        x = self.conv_in(x)
        t_emb = self.embeder(x, t)
        
        skips = []       
        for layer in self.down_layers:
            x, _ = layer(x, t_emb)
            skips.append(x)
        # skips = skips[::-1]
        
        x, _ = self.bottleneck(x, t_emb) 
        
        for layer in self.up_layers:
            skip = skips.pop()
            # TODO en principio es mas comun concatenar skips
            x = x + skip  # Suma de skip connection -- alternativa: concatenar
            x, _ = layer(x, t_emb)
        return self.conv_out(x)
         