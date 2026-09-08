import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConditionEmbedder(nn.Module):
    """
    Convierte las etiquetas/parámetros en un mapa denso de condición.

    Entradas (todas concatenadas en un vector plano):
      - instrument_onehot : (B, 11)   one-hot de familia de instrumento (NSynth tiene 11)
      - pitch_norm        : (B,  1)   MIDI 0-127 normalizado a [0,1]
      - velocity_norm     : (B,  1)   velocity 0-127 normalizado a [0,1]
      - brightness        : (B,  1)   parámetro continuo [0,1]
      - sustain           : (B,  1)   parámetro continuo [0,1]
    Total input = 11 + 1 + 1 + 1 + 1 = 15

    Salida: (B, condition_dim, 1, 1) -> se expande a (H,W) fuera de aquí,
    donde haga falta (ver ConditionalVAE._condition_map).
    """

    N_INSTRUMENTS = 11   # familias en NSynth
    N_CONTINUOUS  =  4   # pitch, velocity, brightness, sustain
    INPUT_DIM     = N_INSTRUMENTS + N_CONTINUOUS   # = 15

    def __init__(self, condition_dim: int = 128):
        super().__init__()
        self.condition_dim = condition_dim
        # Conv2d 1x1 == Linear aplicado en cada posicion espacial, pero
        # sin necesitar Flatten porque aqui la "posicion espacial" es 1x1.
        self.embedder = nn.Sequential(
            nn.Conv2d(self.INPUT_DIM, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, condition_dim, kernel_size=1),
            nn.ReLU(),
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
        ], dim=1)                              # (B, 15)

        c = c.view(c.shape[0], c.shape[1], 1, 1)   # (B, 15, 1, 1) sin Flatten
        return self.embedder(c)                # (B, condition_dim, 1, 1)


class Encoder(nn.Module):
    """
    Conv2D stack: (B,1,H,W) → μ (B, latent_dim, H_lat, W_lat),
                             logvar (B, latent_dim, H_lat, W_lat)
    Sin BatchNorm (perjudica VAE) y sin Flatten/Linear (mu/logvar via Conv2d 1x1).
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

        # Cabezas mu/logvar: Conv2d 1x1, preservan (H_lat, W_lat).
        # Sustituyen a "Flatten + Linear(flat_dim, latent_dim)".
        self.conv_mu     = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        h      = self.encoder(x)          # (B, channels[-1], H_lat, W_lat)
        mu     = self.conv_mu(h)          # (B, latent_dim, H_lat, W_lat)
        logvar = self.conv_logvar(h)      # (B, latent_dim, H_lat, W_lat)
        return mu, logvar

    def get_sizes(self):
        return self.sizes


class DecoderMel(nn.Module):
    """
    (B, latent_dim + condition_dim, H_lat, W_lat) → (B, 1, H, W)  Mel-spectrogram reconstruido.

    La diferencia respecto al Decoder original es que en vez de
    "Linear + Unflatten" ahora hay una Conv2d 1x1 (input_conv) que solo
    cambia canales, porque z||c ya tienen la forma espacial correcta.
    """

    def __init__(self, sizes, latent_dim, channels, condition_dim: int = 128):
        super().__init__()
        kernel  = (3, 3)
        stride  = (2, 2)
        pad     = (1, 1)

        rev_ch   = list(reversed(channels))
        rev_sz   = list(reversed(sizes))
        in_dim   = latent_dim + condition_dim   # ← recibe z∥c (por canales)

        self.input_conv = nn.Conv2d(in_dim, rev_ch[0], kernel_size=1)

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
        """zc: (B, latent_dim + condition_dim, H_lat, W_lat)"""
        x = self.input_conv(zc)   # (B, rev_ch[0], H_lat, W_lat) solo cambia canales
        x = self.decoder(x)
        return x


class DecoderDDSP(nn.Module):
    """
    (B, latent_dim + condition_dim, H_lat, W_lat) → parámetros DDSP por frame.

    Salidas:
      - f0_scale       : (B, n_frames)       escala multiplicativa sobre el f0 de referencia
      - loudness_scale : (B, n_frames)       escala aditiva sobre loudness_db de referencia
      - harmonics      : (B, n_frames, n_harmonics)  amplitudes relativas de armónicos

    Los valores de referencia (f0_ref, loudness_ref) vienen del dataset (.pt);
    la síntesis diferenciable se hace fuera, en losses.py / ddsp_vocoder.py.

    n_frames, n_harmonics y latent_hw (H_lat, W_lat del cuello de botella del
    encoder) son necesarios para dimensionar las capas; hay q ajustarlos
    segun nuestra configuración.
    """

    def __init__(self, latent_dim: int, condition_dim: int = 128,
                 n_frames: int = 100, n_harmonics: int = 64,
                 hidden_dim: int = 256, latent_hw=(1, 1)):
        super().__init__()
        self.n_frames    = n_frames
        self.n_harmonics = n_harmonics
        in_dim = latent_dim + condition_dim
        H_lat, W_lat = latent_hw

        # 1) Colapsa el eje de frecuencia (H_lat) a 1, conserva el eje tiempo (W_lat).
        #    Esto sustituye al "Flatten": en vez de aplastar todo, solo se
        #    aplasta la dimension que no nos interesa (frecuencia).
        self.freq_collapse = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=(H_lat, 1)),
            nn.ReLU(),
        )

        # 2) Refinamiento temporal: kernel (1,3) => solo mira vecinos en el eje tiempo.
        self.temporal_trunk = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
        )

        # 3) Cabezas independientes, Conv2d 1x1 (sustituyen a los Linear de cabeza).
        self.head_f0       = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_loudness = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_harm     = nn.Conv2d(hidden_dim, n_harmonics, kernel_size=1)

    def forward(self, zc):
        """
        zc: (B, latent_dim + condition_dim, H_lat, W_lat)
        Returns dict con las tres salidas.
        """
        h = self.freq_collapse(zc)                              # (B, hidden, 1, W_lat)
        h = self.temporal_trunk(h)                               # (B, hidden, 1, W_lat)

        # Ajusta el eje temporal a n_frames exactos (remuestreo, sin pesos).
        h = F.interpolate(h, size=(1, self.n_frames),
                          mode='bilinear', align_corners=False)  # (B, hidden, 1, n_frames)

        f0_raw       = self.head_f0(h).squeeze(2).squeeze(1)        # (B, n_frames)
        loudness_raw = self.head_loudness(h).squeeze(2).squeeze(1)  # (B, n_frames)
        f0_scale       = torch.sigmoid(f0_raw)         # > 0, en [0,1]
        loudness_scale = torch.sigmoid(loudness_raw)   # forzado a [0,1]

        harmonics_map = self.head_harm(h).squeeze(2)             # (B, n_harmonics, n_frames)
        harmonics     = F.softmax(harmonics_map, dim=1)          # normaliza por armónico
        harmonics     = harmonics.permute(0, 2, 1)                # (B, n_frames, n_harmonics), suman 1

        return {
            'f0_scale':       f0_scale,        # multiplica f0_ref
            'loudness_scale': loudness_scale,   # suma a loudness_ref en dB
            'harmonics':      harmonics,        # mezcla de armónicos
        }


class ConditionalVAE(nn.Module):
    
    def __init__(self, input_size=(80, 128), latent_dim=256,
                 channels=None, condition_dim=128,
                 n_frames=100, n_harmonics=64, ddsp_hidden=256, free_bits=0.0):
        super().__init__()

        if channels is None:
            raise ValueError('channels no puede ser None')

        self.latent_dim    = latent_dim
        self.condition_dim = condition_dim
        self.free_bits = free_bits

        #  Módulos 
        self.condition_embedder = ConditionEmbedder(condition_dim)

        self.encoder    = Encoder(input_size, latent_dim, channels)
        sizes           = self.encoder.get_sizes()
        self.latent_hw  = sizes[-1]     # (H_lat, W_lat) del cuello de botella

        self.decoder_mel  = DecoderMel(sizes, latent_dim, channels, condition_dim)
        self.decoder_ddsp = DecoderDDSP(latent_dim, condition_dim,
                                        n_frames, n_harmonics, ddsp_hidden,
                                        latent_hw=self.latent_hw)

    #  Reparameterización (elementwise, funciona igual con mapas 4D) 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # KL divergence 
    @staticmethod
    def kld(mu, logvar,free_bits=0.0):
        """KL( q(z|x) || N(0,I) ), sumada por canal Y por posición espacial."""
        
        # KL individual para cada posición del mapa latente
        # Shape: (B, latent_dim, H_lat, W_lat)
        kl_per_unit = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # Free Bits:
        # Cada unidad latente puede tener hasta `free_bits`
        # de KL sin ser penalizada adicionalmente.
        if free_bits > 0:
            kl_per_unit = torch.clamp(
                kl_per_unit,
                min=free_bits
            )

        # Sumamos todos los canales y posiciones espaciales
        # Shape final: (B,)
        kl = kl_per_unit.sum(dim=[1, 2, 3])

        return kl

    #  Condición: calcula c y la expande de (B,cond,1,1) a (B,cond,H,W) 
    def _condition_map(self, instrument_onehot, pitch_norm, velocity_norm,
                        brightness, sustain, hw):
        c = self.condition_embedder(instrument_onehot, pitch_norm,
                                    velocity_norm, brightness, sustain)   # (B, cond, 1, 1)
        return c.expand(-1, -1, hw[0], hw[1])                            # (B, cond, H, W), broadcast

    #  Forward 
    def forward(self, mel,
                instrument_onehot, pitch_norm, velocity_norm,
                brightness, sustain):

        B, C, H, W = mel.shape

        # 1. Codificar → mapa latente, no vector
        mu, logvar = self.encoder(mel)                          # (B, latent_dim, H_lat, W_lat)
        logvar     = torch.clamp(logvar, min=-20, max=20)
        z          = self.reparameterize(mu, logvar)

        # 2. Embedding de condición, expandido al tamaño espacial de z
        c_map = self._condition_map(instrument_onehot, pitch_norm, velocity_norm,
                                    brightness, sustain, hw=z.shape[2:])

        # 3. Concatenar z y c por canal (ambos son mapas espaciales)
        zc = torch.cat([z, c_map], dim=1)                        # (B, latent_dim+condition_dim, H_lat, W_lat)

        # 4. Decoders paralelos
        mel_hat    = self.decoder_mel(zc)
        mel_hat    = adjust_shape(mel_hat, (H, W))               # (B, 1, H, W)
        ddsp_params = self.decoder_ddsp(zc)

        # 5. KL
        kl = self.kld(mu, logvar,free_bits=self.free_bits)                                # (B,)

        return mel_hat, ddsp_params, kl, mu,logvar

    #  Sampling (inferencia / demo) 
    @staticmethod
    def _prep_condition_tensor(t, n_samples, device):

        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(0)          # (D,) -> (1, D): asumimos 1 sola condicion

        batch = t.shape[0]
        if batch == n_samples:
            return t
        if batch == 1:
            reps = [n_samples] + [1] * (t.dim() - 1)
            return t.repeat(*reps)      # (1, ...) -> (n_samples, ...)
        raise ValueError(
            f'El batch de esta condicion es {batch}, pero n_samples={n_samples}. '
            f'Pasa una condicion por muestra (batch == n_samples) o una sola '
            f'condicion (batch == 1) para repetirla automaticamente.'
        )

    @torch.no_grad()
    def sample(self, instrument_onehot, pitch_norm, velocity_norm,
               brightness, sustain, n_samples=1, z=None):
 
        device = next(self.parameters()).device
        H_lat, W_lat = self.latent_hw

        instrument_onehot = self._prep_condition_tensor(instrument_onehot, n_samples, device)
        pitch_norm        = self._prep_condition_tensor(pitch_norm, n_samples, device)
        velocity_norm      = self._prep_condition_tensor(velocity_norm, n_samples, device)
        brightness         = self._prep_condition_tensor(brightness, n_samples, device)
        sustain            = self._prep_condition_tensor(sustain, n_samples, device)

        if z is None:
            z = torch.randn(n_samples, self.latent_dim, H_lat, W_lat, device=device)
        c_map = self._condition_map(instrument_onehot, pitch_norm, velocity_norm,
                                    brightness, sustain, hw=(H_lat, W_lat))
        zc       = torch.cat([z, c_map], dim=1)
        mel_hat  = self.decoder_mel(zc)
        ddsp_out = self.decoder_ddsp(zc)
        return mel_hat, ddsp_out

    #  Interpolación latente (análisis) 
    @torch.no_grad()
    def interpolate(self, mel_a, mel_b, cond_a, cond_b, steps=8):
        """
        Interpola linealmente entre dos puntos del espacio latente (mapa a mapa).
        cond_* son tuplas (instrument_oh, pitch_n, vel_n, brightness, sustain).
        Retorna lista de (mel_hat, ddsp_params) para cada paso.
        """
        mu_a, _ = self.encoder(mel_a)
        mu_b, _ = self.encoder(mel_b)
        outputs  = []
        for alpha in torch.linspace(0, 1, steps):
            z   = (1 - alpha) * mu_a + alpha * mu_b                       # (B, latent_dim, H_lat, W_lat)
            c_a = self._condition_map(*cond_a, hw=z.shape[2:])
            c_b = self._condition_map(*cond_b, hw=z.shape[2:])
            c   = (1 - alpha) * c_a + alpha * c_b
            zc  = torch.cat([z, c], dim=1)
            mel_hat  = self.decoder_mel(zc)
            ddsp_out = self.decoder_ddsp(zc)
            outputs.append((mel_hat, ddsp_out))
        return outputs
