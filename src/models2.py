import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    H_in, W_in = input_size
    H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return (H_out, W_out)


def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding,
                                        output_padding=(0, 0)):
    H_in, W_in = input_size
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    return (H_out, W_out)


def adjust_shape(x, target_hw, pad_mode='reflect'):
    """Recorta o rellena x para que tenga exactamente target_hw."""
    _, _, H, W = x.shape
    th, tw = target_hw
    if H > th:
        x = x[:, :, :th, :]
    elif H < th:
        x = F.pad(x, (0, 0, 0, th - H), mode=pad_mode)
    if W > tw:
        x = x[:, :, :, :tw]
    elif W < tw:
        x = F.pad(x, (0, tw - W), mode=pad_mode)
    return x


#  ConditionEmbedder 
""" 
las redes neuronales no entienden d instrumento. este bloque coge las etiquetas (instrumento,pitch,velocity,brightness,sustain) y los tritura en una red neuronal mlp d dos capas (muy chikita)
coge 15 nums iniciales y los convierte en un vector d 129 numeros (embedding). asi la red principal puede fusionarlo con el audio comprimido
"""

class ConditionEmbedder(nn.Module):
    """
    Convierte las etiquetas/parámetros en un vector denso de condición.

    Entradas (todas concatenadas en un vector plano):
      - instrument_onehot : (B, 11)   one-hot de familia de instrumento (NSynth tiene 11)
      - pitch_norm        : (B,  1)   MIDI 0-127 normalizado a [0,1]
      - velocity_norm     : (B,  1)   velocity 0-127 normalizado a [0,1]
      - brightness        : (B,  1)   parámetro continuo [0,1]
      - sustain           : (B,  1)   parámetro continuo [0,1]
    Total input = 11 + 1 + 1 + 1 + 1 = 15

    Salida: (B, condition_dim)  por defecto 128
    """

    N_INSTRUMENTS = 11   # familias en NSynth
    N_CONTINUOUS  =  4   # pitch, velocity, brightness, sustain
    INPUT_DIM     = N_INSTRUMENTS + N_CONTINUOUS   # = 15

    def __init__(self, condition_dim: int = 128):
        super().__init__()
        self.condition_dim = condition_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, condition_dim),
            #nn.ReLU(),
        )

    def forward(self, instrument_onehot, pitch_norm, velocity_norm,
                brightness, sustain):
        """
        Todos los tensores tienen shape (B, *).
        pitch_norm, velocity_norm, brightness, sustain: (B,) o (B, 1).
        """
        def _col(t):
            return t.view(t.shape[0], -1).float()

        c = torch.cat([
            _col(instrument_onehot),
            _col(pitch_norm),
            _col(velocity_norm),
            _col(brightness),
            _col(sustain),
        ], dim=1)                  # (B, 15)
        return self.mlp(c)         # (B, condition_dim)


class Encoder(nn.Module):
    """
    Conv2D stack: (B,1,H,W) → μ (B, latent_dim), logvar (B, latent_dim)
    Idéntico al encoder de VAE original, sin BatchNorm (perjudica VAE).
    """

    def __init__(self, input_size, latent_dim, channels):
        super().__init__()
        conv_kernel = (3, 3)
        conv_stride = (2, 2)
        conv_pad    = (1, 1)

        self.sizes = [input_size]
        current    = input_size
        blocks     = []

        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i-1], channels[i],
                             kernel_size=conv_kernel,
                             stride=conv_stride,
                             padding=conv_pad)
            current = compute_conv2D_output_size(current, conv_kernel, conv_stride, conv_pad)
            self.sizes.append(current)
            blocks.append(nn.Sequential(conv, nn.ReLU()))   # sin BatchNorm (VAE)

        self.encoder = nn.Sequential(*blocks)
        self.flatten  = nn.Flatten()
        flat_dim = channels[-1] * current[0] * current[1]
        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar  = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h   = self.encoder(x)
        h   = self.flatten(h)
        mu      = self.fc_mu(h)
        logvar  = self.fc_logvar(h)
        return mu, logvar

    def get_sizes(self):
        return self.sizes

class DecoderMel(nn.Module):
    """
    (B, latent_dim + condition_dim) → (B, 1, H, W)  Mel-spectrogram reconstruido.

    La única diferencia respecto al Decoder original es que fc acepta
    latent_dim + condition_dim en vez de solo latent_dim.
    """

    def __init__(self, sizes, latent_dim, channels, condition_dim: int = 128):
        super().__init__()
        kernel  = (3, 3)
        stride  = (2, 2)
        pad     = (1, 1)

        rev_ch   = list(reversed(channels))
        rev_sz   = list(reversed(sizes))
        in_dim   = latent_dim + condition_dim   # ← recibe z∥c

        self.fc        = nn.Linear(in_dim, rev_ch[0] * rev_sz[0][0] * rev_sz[0][1])
        self.unflatten = nn.Unflatten(1, (rev_ch[0], rev_sz[0][0], rev_sz[0][1]))

        blocks       = []
        current_size = rev_sz[0]

        for i in range(1, len(rev_sz)):
            target     = rev_sz[i]
            calc_no_op = compute_convTranspose2D_output_size(current_size, kernel, stride, pad)
            op_h = 1 if target[0] - calc_no_op[0] > 0 else 0
            op_w = 1 if target[1] - calc_no_op[1] > 0 else 0
            out_pad    = (op_h, op_w)

            deconv = nn.ConvTranspose2d(rev_ch[i-1], rev_ch[i],
                                        kernel_size=kernel, stride=stride,
                                        padding=pad, output_padding=out_pad)
            is_last = (i == len(rev_sz) - 1)
            block   = nn.Sequential(deconv) if is_last else nn.Sequential(deconv, nn.ReLU())
            blocks.append(block)

            current_size = (calc_no_op[0] + op_h, calc_no_op[1] + op_w)
            assert current_size == target, f"Shape mismatch capa {i}: {current_size} vs {target}"

        self.decoder = nn.Sequential(*blocks)

    def forward(self, zc):
        """zc: (B, latent_dim + condition_dim)"""
        x = self.fc(zc)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x


class DecoderDDSP(nn.Module):
    """
    (B, latent_dim + condition_dim) → parámetros DDSP por frame.

    Salidas:
      - f0_scale       : (B, n_frames)       escala multiplicativa sobre el f0 de referencia
      - loudness_scale : (B, n_frames)       escala aditiva sobre loudness_db de referencia
      - harmonics      : (B, n_frames, n_harmonics)  amplitudes relativas de armónicos

    Los valores de referencia (f0_ref, loudness_ref) vienen del dataset (.pt);
    la síntesis diferenciable se hace fuera, en losses.py / ddsp_vocoder.py.

    n_frames y n_harmonics son hiperparámetros; hay q ajustarlos segun nuestra configuración.
    """

    def __init__(self, latent_dim: int, condition_dim: int = 128,
                 n_frames: int = 100, n_harmonics: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_frames     = n_frames
        self.n_harmonics  = n_harmonics
        in_dim = latent_dim + condition_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Cabezas independientes, así puedes anular o escalar cada una
        self.head_f0       = nn.Linear(hidden_dim, n_frames)
        self.head_loudness = nn.Linear(hidden_dim, n_frames)
        self.head_harm     = nn.Linear(hidden_dim, n_frames * n_harmonics)

    def forward(self, zc):
        """
        zc: (B, latent_dim + condition_dim)
        Returns dict con las tres salidas.
        """
        h = self.trunk(zc)                                      # (B, hidden)

        f0_scale       = torch.sigmoid(self.head_f0(h))        # (B, n_frames)   > 0
        # mantener loudness scale en rango
        loudness_scale = torch.sigmoid(self.head_loudness(h))  # forzar [0,1]              # (B, n_frames)   aditivo dB
        harmonics_flat = self.head_harm(h)                      # (B, n_frames*H)
        harmonics      = F.softmax(
            harmonics_flat.view(-1, self.n_frames, self.n_harmonics),
            dim=-1
        )                                                       # (B, n_frames, H) suman 1

        return {
            'f0_scale':       f0_scale,        # multiplica f0_ref
            'loudness_scale': loudness_scale,   # suma a loudness_ref en dB
            'harmonics':      harmonics,        # mezcla de armónicos
        }


class ConditionalVAE(nn.Module):
   
    def __init__(self, input_size=(80, 128), latent_dim=256,
                 channels=None, condition_dim=128,
                 n_frames=100, n_harmonics=64, ddsp_hidden=256):
        super().__init__()

        if channels is None:
            raise ValueError('channels no puede ser None')

        self.latent_dim    = latent_dim
        self.condition_dim = condition_dim

        #  Módulos 
        self.condition_embedder = ConditionEmbedder(condition_dim)

        self.encoder    = Encoder(input_size, latent_dim, channels)
        sizes           = self.encoder.get_sizes()

        self.decoder_mel  = DecoderMel(sizes, latent_dim, channels, condition_dim)
        self.decoder_ddsp = DecoderDDSP(latent_dim, condition_dim,
                                        n_frames, n_harmonics, ddsp_hidden)

        # Prior N(0,I), movemos a device en el primer forward
        self._prior_loc   = None
        self._prior_scale = None

    #  Reparameterización 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # KL divergence 
    @staticmethod
    def kld(mu, logvar):
        """KL( q(z|x) || N(0,I) ), sumada por dimensión latente."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (B,)

    #  Forward 
    def forward(self, mel,
                instrument_onehot, pitch_norm, velocity_norm,
                brightness, sustain):

        B, C, H, W = mel.shape

        # 1. Codificar
        mu, logvar = self.encoder(mel)
        logvar     = torch.clamp(logvar, min=-20, max=20)
        z          = self.reparameterize(mu, logvar)           # (B, latent_dim)

        # 2. Embedding de condición
        c = self.condition_embedder(instrument_onehot,
                                    pitch_norm, velocity_norm,
                                    brightness, sustain)        # (B, condition_dim)

        # 3. Concatenar z y c
        zc = torch.cat([z, c], dim=1)                          # (B, latent_dim + condition_dim)

        # 4. Decoders paralelos
        mel_hat    = self.decoder_mel(zc)
        mel_hat    = adjust_shape(mel_hat, (H, W))             # (B, 1, H, W)
        ddsp_params = self.decoder_ddsp(zc)

        # 5. KL
        kl = self.kld(mu, logvar)                              # (B,)

        return mel_hat, ddsp_params, kl

    #  Sampling (inferencia / demo) 
    @torch.no_grad()
    def sample(self, instrument_onehot, pitch_norm, velocity_norm,
               brightness, sustain, n_samples=1):
        """
        Genera audio desde el prior N(0,I) sin pasar audio de entrada.
        Útil para el demo web: das etiquetas y obtienes síntesis.

        Retorna mel_hat y ddsp_params.
        """
        z = torch.randn(n_samples, self.latent_dim,
                        device=next(self.parameters()).device)
        c = self.condition_embedder(instrument_onehot,
                                    pitch_norm, velocity_norm,
                                    brightness, sustain)
        zc       = torch.cat([z, c], dim=1)
        # Para el Mel necesitamos un tamaño de referencia; aquí asumimos (80,128)
        # Ajusta si tu input_size es diferente
        mel_hat  = self.decoder_mel(zc)
        ddsp_out = self.decoder_ddsp(zc)
        return mel_hat, ddsp_out

    #  Interpolación latente (análisis) 
    @torch.no_grad()
    def interpolate(self, mel_a, mel_b, cond_a, cond_b, steps=8):
        """
        Interpola linealmente entre dos puntos del espacio latente.
        cond_* son tuplas (instrument_oh, pitch_n, vel_n, brightness, sustain).
        Retorna lista de (mel_hat, ddsp_params) para cada paso.
        """
        mu_a, _ = self.encoder(mel_a)
        mu_b, _ = self.encoder(mel_b)
        outputs  = []
        for alpha in torch.linspace(0, 1, steps):
            z  = (1 - alpha) * mu_a + alpha * mu_b
            c_a = self.condition_embedder(*cond_a)
            c_b = self.condition_embedder(*cond_b)
            c   = (1 - alpha) * c_a + alpha * c_b
            zc  = torch.cat([z, c], dim=1)
            mel_hat  = self.decoder_mel(zc)
            ddsp_out = self.decoder_ddsp(zc)
            outputs.append((mel_hat, ddsp_out))
        return outputs


#  Smoke-test rápido 

if __name__ == '__main__':
    B = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = ConditionalVAE(
        input_size   = (80, 128),
        latent_dim   = 256,
        channels     = [1, 32, 64, 128, 256],
        condition_dim= 128,
        n_frames     = 100,
        n_harmonics  = 64,
    ).to(device)

    # Inputs ficticios
    mel       = torch.randn(B, 1, 80, 128).to(device)
    inst_oh   = F.one_hot(torch.randint(0, 11, (B,)), num_classes=11).float().to(device)
    pitch_n   = torch.rand(B, 1).to(device)
    vel_n     = torch.rand(B, 1).to(device)
    bright    = torch.rand(B, 1).to(device)
    sustain   = torch.rand(B, 1).to(device)

    mel_hat, ddsp_params, kl = model(mel, inst_oh, pitch_n, vel_n, bright, sustain)

    print(f'mel_hat shape        : {mel_hat.shape}')           # (4, 1, 80, 128)
    print(f'f0_scale shape       : {ddsp_params["f0_scale"].shape}')      # (4, 100)
    print(f'loudness_scale shape : {ddsp_params["loudness_scale"].shape}') # (4, 100)
    print(f'harmonics shape      : {ddsp_params["harmonics"].shape}')      # (4, 100, 64)
    print(f'kl shape             : {kl.shape}')                # (4,)
    print(f'kl mean              : {kl.mean().item():.4f}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'parámetros totales   : {n_params:,}')
    print('smoke-test OK ✓')
