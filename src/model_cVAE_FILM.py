"""
Conditional VAE multimodal (esqueleto Fase 2) — version "todo convolucional"
+ FiLM en cada etapa de los decoders

=====================================================================
CAMBIOS respecto a la version anterior, y por que:
=====================================================================

PROBLEMA que estabamos viendo: forzar pitch_norm=0 vs pitch_norm=1
(u otras condiciones) casi no cambiaba el audio reconstruido. Con esta
arquitectura hay un motivo estructural muy concreto para eso, ademas
del "z ya conoce el pitch real" que hablamos:

  El embedding de condicion (c_map) SOLO se inyecta UNA VEZ, en el
  cuello de botella (torch.cat([z, c_map], dim=1) -> input_conv).
  A partir de ahi, decoder_mel tiene que "arrastrar" esa informacion
  a traves de 3-4 capas de ConvTranspose2d, y decoder_ddsp a traves
  de freq_collapse + 2 capas temporales, sin que la condicion se
  vuelva a mencionar en ningun punto intermedio. Es exactamente el
  patron cl\u00e1sico de "dilucion de condicionamiento" en decoders
  convolucionales profundos: cuanto mas lejos del punto de inyeccion,
  mas facil es que el gradiente encuentre mas barato tirar de z (que
  SI esta correlacionado en cada capa, porque atraviesa las mismas
  capas) que de una senal que solo aparecio una vez al principio.

CAMBIO: FiLM (Feature-wise Linear Modulation) en cada etapa de ambos
decoders. En vez de inyectar la condicion una sola vez, cada bloque
del decoder recibe tambien c_vec (el embedding SIN expandir, (B,
condition_dim, 1, 1)) y lo usa para modular sus propias activaciones:

    h_mod = h * (1 + gamma) + beta      # gamma, beta = f(c_vec)

gamma/beta se generan con una Conv2d 1x1 (equivalente a un Linear
aplicado canal a canal), asi que sigue sin haber nn.Linear ni
nn.Flatten en ningun sitio, coherente con el resto del modelo. La
inyeccion inicial en el cuello de botella (concat de z y c_map) SE
MANTIENE ademas de FiLM -> mas puntos de entrada para la condicion,
no menos.

Nada de esto cambia el problema de "z puede seguir sabiendo el pitch
real porque viene de codificar el audio real" que hablamos antes:
esto solo asegura que, SI le pides al decoder una condicion distinta,
tenga muchas mas oportunidades de hacerle caso. La prueba real de si
esto funciona sigue siendo la misma que ya planteamos: usar
model.sample() (z desde ruido puro, sin fuga posible desde audio
real) variando una condicion cada vez.

Fuera de eso, la arquitectura es identica: mismo Encoder, mismo
ConditionEmbedder, mismo KLD con free bits, mismo reparameterize.
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers de shape (iguales que antes) ────────────────────────────
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


# ── FiLM (nuevo) ─────────────────────────────────────────────────────
class FiLM(nn.Module):
    """
    Genera (gamma, beta) a partir del embedding de condicion SIN expandir
    (B, condition_dim, 1, 1) y modula un mapa de activaciones h
    (B, C, H, W) canal a canal:

        h_mod = h * (1 + gamma) + beta   (broadcast sobre H, W)

    Se llama en cada etapa de los decoders para que la condicion no se
    diluya tras pasar por varias capas -- en vez de inyectarse una sola
    vez al principio y tener que "sobrevivir" sin refuerzo.

    Conv2d 1x1 == Linear aplicado por canal, igual que el resto del
    modelo: sigue sin haber nn.Linear ni nn.Flatten en ningun sitio.
    """

    def __init__(self, condition_dim: int, out_channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Conv2d(condition_dim, out_channels * 2, kernel_size=1)
        # Empieza como identidad (gamma=0, beta=0 -> h*(1+0)+0 = h): asi el
        # modelo no arranca con una modulacion aleatoria destructiva, y
        # aprende a usar FiLM gradualmente en vez de partir de ruido.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h, c_vec):
        gamma_beta = self.to_gamma_beta(c_vec)          # (B, 2*out_channels, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)         # cada uno (B, out_channels, 1, 1)
        return h * (1 + gamma) + beta


# ── ConditionEmbedder (igual que antes) ─────────────────────────────
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

    Salida: (B, condition_dim, 1, 1)  — se expande a (H,W) fuera de aquí.
    """

    N_INSTRUMENTS = 11
    N_CONTINUOUS = 4
    INPUT_DIM = N_INSTRUMENTS + N_CONTINUOUS

    def __init__(self, condition_dim: int = 128):
        super().__init__()
        self.condition_dim = condition_dim
        self.embedder = nn.Sequential(
            nn.Conv2d(self.INPUT_DIM, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, condition_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, instrument_onehot, pitch_norm, velocity_norm,
                brightness, sustain):
        def _col(t):
            return t.view(t.shape[0], -1).float()

        c = torch.cat([
            _col(instrument_onehot),
            _col(pitch_norm),
            _col(velocity_norm),
            _col(brightness),
            _col(sustain),
        ], dim=1)

        c = c.view(c.shape[0], c.shape[1], 1, 1)
        return self.embedder(c)


# ── Encoder (igual que antes) ────────────────────────────────────────
class Encoder(nn.Module):
    """
    Conv2D stack: (B,1,H,W) → μ (B, latent_dim, H_lat, W_lat),
                             logvar (B, latent_dim, H_lat, W_lat)
    """

    def __init__(self, input_size, latent_dim, channels):
        super().__init__()
        conv_kernel = (3, 3)
        conv_stride = (2, 2)
        conv_pad = (1, 1)

        self.sizes = [input_size]
        current = input_size
        blocks = []

        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i-1], channels[i],
                             kernel_size=conv_kernel,
                             stride=conv_stride,
                             padding=conv_pad)
            current = compute_conv2D_output_size(current, conv_kernel, conv_stride, conv_pad)
            self.sizes.append(current)
            blocks.append(nn.Sequential(conv, nn.ReLU()))

        self.encoder = nn.Sequential(*blocks)
        self.conv_mu = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def get_sizes(self):
        return self.sizes


# ── DecoderMel (MODIFICADO: FiLM en cada etapa) ─────────────────────
class DecoderMel(nn.Module):
    """
    (z, c_map, c_vec) → (B, 1, H, W)  Mel-spectrogram reconstruido.

    Ademas de la inyeccion inicial (concat z||c_map en el cuello de
    botella, igual que antes), cada ConvTranspose2d va seguida de un
    FiLM(c_vec) antes de la no-linealidad -- asi la condicion se
    reafirma en cada resolucion espacial, no solo al principio.
    """

    def __init__(self, sizes, latent_dim, channels, condition_dim: int = 128,
                 film_last_layer: bool = False):
        super().__init__()
        kernel = (3, 3)
        stride = (2, 2)
        pad = (1, 1)

        rev_ch = list(reversed(channels))
        rev_sz = list(reversed(sizes))
        in_dim = latent_dim + condition_dim

        self.input_conv = nn.Conv2d(in_dim, rev_ch[0], kernel_size=1)

        self.deconvs = nn.ModuleList()
        self.films = nn.ModuleList()
        self._is_last = []

        current_size = rev_sz[0]
        for i in range(1, len(rev_sz)):
            target = rev_sz[i]
            calc_no_op = compute_convTranspose2D_output_size(current_size, kernel, stride, pad)
            op_h = 1 if target[0] - calc_no_op[0] > 0 else 0
            op_w = 1 if target[1] - calc_no_op[1] > 0 else 0
            out_pad = (op_h, op_w)

            deconv = nn.ConvTranspose2d(rev_ch[i-1], rev_ch[i],
                                        kernel_size=kernel, stride=stride,
                                        padding=pad, output_padding=out_pad)
            is_last = (i == len(rev_sz) - 1)
            self.deconvs.append(deconv)
            self._is_last.append(is_last)

            # FiLM en todas las capas salvo la ultima (que produce el mel
            # crudo; modularla tambien es valido, pero por defecto lo
            # dejamos fuera para no forzar la escala/offset de la salida
            # final). Pon film_last_layer=True si quieres probarlo.
            if (not is_last) or film_last_layer:
                self.films.append(FiLM(condition_dim, rev_ch[i]))
            else:
                self.films.append(None)

            current_size = (calc_no_op[0] + op_h, calc_no_op[1] + op_w)
            assert current_size == target, f"Shape mismatch capa {i}: {current_size} vs {target}"

    def forward(self, z, c_map, c_vec):
        """
        z     : (B, latent_dim, H_lat, W_lat)
        c_map : (B, condition_dim, H_lat, W_lat)  — para la inyeccion inicial
        c_vec : (B, condition_dim, 1, 1)          — para FiLM en cada etapa
        """
        zc = torch.cat([z, c_map], dim=1)
        x = self.input_conv(zc)

        for deconv, film, is_last in zip(self.deconvs, self.films, self._is_last):
            x = deconv(x)
            if film is not None:
                x = film(x, c_vec)
            if not is_last:
                x = F.relu(x)
        return x


# ── DecoderDDSP (MODIFICADO: FiLM en cada etapa) ────────────────────
class DecoderDDSP(nn.Module):
    """
    (z, c_map, c_vec) → parámetros DDSP por frame.

    Igual que DecoderMel: la condicion se reinyecta via FiLM despues de
    freq_collapse y de cada capa del temporal_trunk, ademas de la
    concatenacion inicial.
    """

    def __init__(self, latent_dim: int, condition_dim: int = 128,
                 n_frames: int = 100, n_harmonics: int = 64,
                 hidden_dim: int = 256, latent_hw=(1, 1)):
        super().__init__()
        self.n_frames = n_frames
        self.n_harmonics = n_harmonics
        in_dim = latent_dim + condition_dim
        H_lat, W_lat = latent_hw

        self.freq_collapse_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=(H_lat, 1))
        self.film_freq = FiLM(condition_dim, hidden_dim)

        self.temporal_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.film_t1 = FiLM(condition_dim, hidden_dim)
        self.temporal_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.film_t2 = FiLM(condition_dim, hidden_dim)

        self.head_f0 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_loudness = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_harm = nn.Conv2d(hidden_dim, n_harmonics, kernel_size=1)

    def forward(self, z, c_map, c_vec):
        zc = torch.cat([z, c_map], dim=1)

        h = self.freq_collapse_conv(zc)          # (B, hidden, 1, W_lat)
        h = self.film_freq(h, c_vec)
        h = F.relu(h)

        h = self.temporal_conv1(h)
        h = self.film_t1(h, c_vec)
        h = F.relu(h)

        h = self.temporal_conv2(h)
        h = self.film_t2(h, c_vec)
        h = F.relu(h)

        h = F.interpolate(h, size=(1, self.n_frames),
                          mode='bilinear', align_corners=False)  # (B, hidden, 1, n_frames)

        f0_raw = self.head_f0(h).squeeze(2).squeeze(1)
        loudness_raw = self.head_loudness(h).squeeze(2).squeeze(1)
        f0_scale = torch.sigmoid(f0_raw)
        loudness_scale = torch.sigmoid(loudness_raw)

        harmonics_map = self.head_harm(h).squeeze(2)
        harmonics = F.softmax(harmonics_map, dim=1)
        harmonics = harmonics.permute(0, 2, 1)

        return {
            'f0_scale': f0_scale,
            'loudness_scale': loudness_scale,
            'harmonics': harmonics,
        }


# ── ConditionalVAE (MODIFICADO: pasa z / c_map / c_vec por separado) ─
class ConditionalVAE(nn.Module):
    """
    VAE condicional con dos decoders paralelos: Mel y DDSP.
    Espacio latente = mapa espacial (B, latent_dim, H_lat, W_lat).
    Condicionamiento inyectado dos veces: concat en el cuello de
    botella + FiLM en cada etapa de ambos decoders.
    """

    def __init__(self, input_size=(80, 128), latent_dim=256,
                 channels=None, condition_dim=128,
                 n_frames=100, n_harmonics=64, ddsp_hidden=256, free_bits=0.0):
        super().__init__()

        if channels is None:
            raise ValueError('channels no puede ser None')

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.free_bits = free_bits

        self.condition_embedder = ConditionEmbedder(condition_dim)

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
        self.latent_hw = sizes[-1]

        self.decoder_mel = DecoderMel(sizes, latent_dim, channels, condition_dim)
        self.decoder_ddsp = DecoderDDSP(latent_dim, condition_dim,
                                        n_frames, n_harmonics, ddsp_hidden,
                                        latent_hw=self.latent_hw)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kld(mu, logvar, free_bits=0.0):
        """KL( q(z|x) || N(0,I) ), sumada por canal Y por posición espacial."""
        kl_per_unit = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_unit = torch.clamp(kl_per_unit, min=free_bits)
        kl = kl_per_unit.sum(dim=[1, 2, 3])
        return kl

    def _condition_vec_and_map(self, instrument_onehot, pitch_norm, velocity_norm,
                                brightness, sustain, hw):
        """Devuelve (c_vec, c_map): el embedding crudo (para FiLM) y su
        version expandida espacialmente (para la concat inicial)."""
        c_vec = self.condition_embedder(instrument_onehot, pitch_norm,
                                        velocity_norm, brightness, sustain)  # (B, cond, 1, 1)
        c_map = c_vec.expand(-1, -1, hw[0], hw[1])                          # (B, cond, H, W)
        return c_vec, c_map

    def forward(self, mel,
                instrument_onehot, pitch_norm, velocity_norm,
                brightness, sustain):
        B, C, H, W = mel.shape

        mu, logvar = self.encoder(mel)
        logvar = torch.clamp(logvar, min=-20, max=20)
        z = self.reparameterize(mu, logvar)

        c_vec, c_map = self._condition_vec_and_map(
            instrument_onehot, pitch_norm, velocity_norm, brightness, sustain,
            hw=z.shape[2:],
        )

        mel_hat = self.decoder_mel(z, c_map, c_vec)
        mel_hat = adjust_shape(mel_hat, (H, W))
        ddsp_params = self.decoder_ddsp(z, c_map, c_vec)

        kl = self.kld(mu, logvar, free_bits=self.free_bits)

        return mel_hat, ddsp_params, kl, mu, logvar

    @staticmethod
    def _prep_condition_tensor(t, n_samples, device):
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(0)

        batch = t.shape[0]
        if batch == n_samples:
            return t
        if batch == 1:
            reps = [n_samples] + [1] * (t.dim() - 1)
            return t.repeat(*reps)
        raise ValueError(
            f'El batch de esta condicion es {batch}, pero n_samples={n_samples}. '
            f'Pasa una condicion por muestra (batch == n_samples) o una sola '
            f'condicion (batch == 1) para repetirla automaticamente.'
        )

    @torch.no_grad()
    def sample(self, instrument_onehot, pitch_norm, velocity_norm,
               brightness, sustain, n_samples=1, z=None):
        """Genera audio desde el prior N(0,I) sin pasar audio de entrada."""
        device = next(self.parameters()).device
        H_lat, W_lat = self.latent_hw

        instrument_onehot = self._prep_condition_tensor(instrument_onehot, n_samples, device)
        pitch_norm = self._prep_condition_tensor(pitch_norm, n_samples, device)
        velocity_norm = self._prep_condition_tensor(velocity_norm, n_samples, device)
        brightness = self._prep_condition_tensor(brightness, n_samples, device)
        sustain = self._prep_condition_tensor(sustain, n_samples, device)

        if z is None:
            z = torch.randn(n_samples, self.latent_dim, H_lat, W_lat, device=device)

        c_vec, c_map = self._condition_vec_and_map(
            instrument_onehot, pitch_norm, velocity_norm, brightness, sustain,
            hw=(H_lat, W_lat),
        )

        mel_hat = self.decoder_mel(z, c_map, c_vec)
        ddsp_out = self.decoder_ddsp(z, c_map, c_vec)
        return mel_hat, ddsp_out

    @torch.no_grad()
    def interpolate(self, mel_a, mel_b, cond_a, cond_b, steps=8):
        """Interpola linealmente entre dos puntos del espacio latente
        Y entre las dos condiciones (antes solo interpolaba z)."""
        mu_a, _ = self.encoder(mel_a)
        mu_b, _ = self.encoder(mel_b)
        outputs = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu_a + alpha * mu_b

            c_vec_a = self.condition_embedder(*cond_a)
            c_vec_b = self.condition_embedder(*cond_b)
            c_vec = (1 - alpha) * c_vec_a + alpha * c_vec_b
            c_map = c_vec.expand(-1, -1, z.shape[2], z.shape[3])

            mel_hat = self.decoder_mel(z, c_map, c_vec)
            ddsp_out = self.decoder_ddsp(z, c_map, c_vec)
            outputs.append((mel_hat, ddsp_out))
        return outputs