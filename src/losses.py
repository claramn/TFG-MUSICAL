import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for perceptually accurate audio reconstruction.
    
    Computes L1 loss on log-magnitude spectrograms at multiple scales,
    inspired by pc-ddsp and DDSP vocoder literature.
    
    Args:
        n_fft_list: list of FFT sizes (e.g., [512, 1024, 2048])
        hop_length_list: list of hop lengths
        win_length_list: list of window lengths
        sample_rate: sample rate (for Hann window registration)
    """
    
    def __init__(
        self,
        n_fft_list=None,
        hop_length_list=None,
        win_length_list=None,
        sample_rate=16000,
    ):
        super().__init__()
        
        if n_fft_list is None:
            n_fft_list = [512, 1024, 2048]
        if hop_length_list is None:
            hop_length_list = [50, 120, 240]
        if win_length_list is None:
            win_length_list = [512, 1024, 2048]
        
        self.n_fft_list = n_fft_list
        self.hop_length_list = hop_length_list
        self.win_length_list = win_length_list
        
        # Register Hann windows as buffers for each scale
        for n_fft in n_fft_list:
            self.register_buffer(
                f"window_{n_fft}",
                torch.hann_window(n_fft)
            )
    
    def forward(self, audio_hat, audio_real):
        """
        Compute multi-scale STFT loss.
        
        Args:
            audio_hat: (B, T) predicted audio
            audio_real: (B, T) reference audio
            
        Returns:
            loss: scalar loss
        """
        if audio_hat.shape != audio_real.shape:
            raise ValueError(
                f"Shape mismatch: audio_hat {audio_hat.shape} vs "
                f"audio_real {audio_real.shape}"
            )
        
        loss = 0.0
        for n_fft, hop_len, win_len in zip(
            self.n_fft_list,
            self.hop_length_list,
            self.win_length_list
        ):
            window = getattr(self, f"window_{n_fft}")
            
            # Compute STFT
            spec_hat = torch.stft(
                audio_hat,
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=win_len,
                window=window,
                center=True,
                return_complex=True,
            )
            spec_real = torch.stft(
                audio_real,
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=win_len,
                window=window,
                center=True,
                return_complex=True,
            )
            
            # Magnitude spectrograms
            mag_hat = torch.abs(spec_hat)
            mag_real = torch.abs(spec_real)
            
            # Log magnitude (with small epsilon for numerical stability)
            log_mag_hat = torch.log(mag_hat + 1e-9)
            log_mag_real = torch.log(mag_real + 1e-9)
            
            # L1 loss on log magnitude
            loss += F.l1_loss(log_mag_hat, log_mag_real)
        
        # Average across scales
        return loss / len(self.n_fft_list)


def spectral_loss(audio_hat, audio_real, sample_rate=16000):
    """
    Quick spectral loss using a default STFT configuration.
    
    Args:
        audio_hat: (B, T) predicted audio
        audio_real: (B, T) reference audio
        sample_rate: sample rate
        
    Returns:
        loss: scalar
    """
    loss_fn = MultiScaleSTFTLoss()
    return loss_fn(audio_hat, audio_real)


def mse_loss(reconstructed, target):
    """
    Mean Squared Error loss for reconstruction.
    """
    return F.mse_loss(reconstructed, target, reduction='mean')

def kl_divergence_loss(mu, log_var):
    """
    KL Divergence loss for VAE.
    """
    kld = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1)
    return kld.mean()

def vae_loss(reconstructed, target, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
    
    kld = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1)
    kld = kld.mean()

    return recon_loss + beta * kld

def diffusion_loss(predicted_noise, actual_noise):
    """
    Loss for diffusion models: MSE between predicted and actual noise.
    """
    return F.mse_loss(predicted_noise, actual_noise, reduction='mean')

class MultiDecoderLoss(nn.Module):
    """
    Función de coste con β-annealing y normalización física de audio.
    """
    def __init__(self, beta_max=0.1, beta_steps=10000, lambda_mel=1.0, lambda_ddsp=1.0):
        super().__init__()
        #Hiperparámetros base
        self.lambda_mel = lambda_mel
        self.lambda_ddsp = lambda_ddsp
        
        #Configuración β-annealing
        self.beta_max = beta_max
        self.beta_steps = beta_steps
        self.current_step = 0
        self.beta = 0.0 # Empieza en 0 para q la red aprenda a reconstruir primero

    def step_beta(self):
        """
        Llama a esta función en cada paso de training loop para subir el beta.
        Ej: loss_fn.step_beta()
        """
        self.current_step += 1
        self.beta = min(self.beta_max, self.beta_max * (self.current_step / self.beta_steps))

    def forward(self, mel_orig, mel_hat, ddsp_params, features_real, kld, mu=None, logvar = None):
        
        # Loss Mel (espectrogramas)
        loss_mel = F.l1_loss(mel_hat, mel_orig)

        #2. Loss KLD (espacio latente)
       # loss_kld = kld.mean()
        if mu is not None and logvar is not None:
            # Per-element KL: shape (B, latent_dim, H_lat, W_lat)
            kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

            # Clamp per-dim KLD to free_bits (in nats per element)
            # free_bits is split across the spatial dims — 0.5 nats per element
            # is a reasonable starting point.
            free_bits = 0.5
            kl_clamped = torch.clamp(kl_per_element, min=free_bits)

            # Sum over latent_dim, H_lat, W_lat → (B,)
            kl_total = kl_clamped.sum(dim=[1, 2, 3])
            loss_kld = kl_total.mean()
        else:
            # Fallback to original behavior if mu/logvar not passed
            loss_kld = kld.mean()

        #3. Loss DDSP 
        f0_pred = ddsp_params['f0_scale']       # Salida sigmoide [0, 1]
        loudness_pred = ddsp_params['loudness_scale'] 
        
        f0_real = features_real['f0'].float()
        loudness_real = features_real['loudness_db'].float()

        # Alinear frames
        f0_real = F.interpolate(f0_real.unsqueeze(1), size=f0_pred.shape[1], mode='linear', align_corners=False).squeeze(1)
        loudness_real = F.interpolate(loudness_real.unsqueeze(1), size=loudness_pred.shape[1], mode='linear', align_corners=False).squeeze(1)

        # FIX 1: Normalización robusta
        # F0: Convertimos Hz a una escala logarítmica (parecido a MIDI) y normalizamos de 0 a 1.
        # F_MIN = 32.7 Hz. Un rango de 6 octavas cubre hasta ~2000 Hz.
        
        #f0_real_norm = (torch.log2(f0_real / 32.7) / 6.0).clamp(0.0, 1.0)
        f0_safe = f0_real.clamp(min=1.0)  # evita log(0)
        f0_real_norm = (torch.log2(f0_safe / 32.7) / 6.0).clamp(0.0, 1.0)
        
        # Loudness: Asumimos un ruido de fondo (silencio absoluto) de -120 dB.
        # Rango [-120, 0] mapeado a [0, 1].
        loudness_real_norm = ((loudness_real + 120.0) / 120.0).clamp(0.0, 1.0)

        # FIX 2: Cálculo de MSE separado para poder monitorizar y balancear
        loss_f0 = F.mse_loss(f0_pred, f0_real_norm)
        
        #Si loudness_pred no tiene un nn.Sigmoid() en el modelo, 
        # asegurarse de que aprenda a predecir valores entre 0 y 1.
        loss_loudness = F.mse_loss(loudness_pred, loudness_real_norm)

        # Balanceo interno del DDSP (el f0 suele ser más crítico al principio que el volumen exacto)
        loss_ddsp = loss_f0 + (0.5 * loss_loudness)

        #LOSS TOTAL con β-annealing (FIX 3)
        total_loss = (self.lambda_mel * loss_mel) + \
                     (self.lambda_ddsp * loss_ddsp) + \
                     (self.beta * loss_kld)

        loss_dict = {
            'loss': total_loss,
            'mel': loss_mel.item(),
            'ddsp': loss_ddsp.item(),
            'f0': loss_f0.item(),
            'loudness': loss_loudness.item(),
            'kld': loss_kld.item(),
            'beta': self.beta  # Útil para loguear en TensorBoard
        }

        return total_loss, loss_dict
    
class MultiDecoderLoss2(nn.Module):
    """
    Loss combinada para el ConditionalVAE con dos decoders (Mel + DDSP).

    Componentes:
      - loss_mel  : L1 entre mel reconstruido y mel original.
      - loss_kld  : KL divergence con free bits (evita posterior collapse)
                    y beta-annealing (0 -> beta_max a lo largo de beta_steps
                    llamadas a step_beta()).
      - loss_ddsp : MSE entre f0/loudness predichos y los reales normalizados.
                    OJO: pese al nombre f0_scale/loudness_scale del decoder,
                    aqui NO se usa ninguna referencia (f0_ref/loudness_ref) —
                    se compara directo contra el valor absoluto normalizado
                    del dataset. Si en algun momento quieres que sean
                    "factores de escala" de verdad, esta loss (y
                    features_real) tendrian que cambiar para traer tambien
                    f0_ref/loudness_ref.
    """

    def __init__(self, beta_max=0.1, beta_steps=10000,
                 lambda_mel=1.0, lambda_ddsp=1.0,
                 free_bits=0.5, free_bits_mode='per_element'):
        super().__init__()
        self.lambda_mel  = lambda_mel
        self.lambda_ddsp = lambda_ddsp

        self.beta_max      = beta_max
        self.beta_steps    = beta_steps
        self.current_step  = 0
        self.beta           = 0.0  # empieza en 0, sube con step_beta()

        assert free_bits_mode in ('per_element', 'per_channel'), \
            "free_bits_mode debe ser 'per_element' o 'per_channel'"
        self.free_bits      = free_bits
        self.free_bits_mode = free_bits_mode

    def step_beta(self):
        """Llamar una vez por batch de training (no por epoch)."""
        self.current_step += 1
        self.beta = min(self.beta_max, self.beta_max * (self.current_step / self.beta_steps))

    def _kld_with_free_bits(self, mu, logvar):
        # (B, latent_dim, H_lat, W_lat)
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits_mode == 'per_element':
            floor = self.free_bits
        else:  # 'per_channel': reparte el presupuesto entre H_lat*W_lat posiciones
            _, _, H_lat, W_lat = mu.shape
            floor = self.free_bits / (H_lat * W_lat)

        kl_clamped = torch.clamp(kl_per_element, min=floor)
        kl_total   = kl_clamped.mean(dim=[1, 2, 3])   # antes: .sum(...)
        return kl_total.mean()

    def forward(self, mel_orig, mel_hat, ddsp_params, features_real, kld, mu, logvar):
        # 1. Mel
        loss_mel = F.l1_loss(mel_hat, mel_orig)

        # 2. KLD (mu/logvar obligatorios: siempre free bits, nunca fallback silencioso)
        loss_kld = self._kld_with_free_bits(mu, logvar)

        # 3. DDSP — se salta entero si lambda_ddsp == 0 (ahorra computo/autograd)
        if self.lambda_ddsp > 0.0:
            f0_pred       = ddsp_params['f0_scale']
            loudness_pred = ddsp_params['loudness_scale']

            f0_real       = features_real['f0'].float()
            loudness_real = features_real['loudness_db'].float()

            f0_real = F.interpolate(f0_real.unsqueeze(1), size=f0_pred.shape[1],
                                     mode='linear', align_corners=False).squeeze(1)
            loudness_real = F.interpolate(loudness_real.unsqueeze(1), size=loudness_pred.shape[1],
                                           mode='linear', align_corners=False).squeeze(1)

            # F0: Hz -> escala log tipo-MIDI, normalizado [0,1] (6 octavas desde 32.7Hz)
            f0_safe      = f0_real.clamp(min=1.0)   # evita log(0)
            f0_real_norm = (torch.log2(f0_safe / 32.7) / 6.0).clamp(0.0, 1.0)

            # Loudness: [-120, 0] dB -> [0, 1]
            loudness_real_norm = ((loudness_real + 120.0) / 120.0).clamp(0.0, 1.0)

            loss_f0       = F.mse_loss(f0_pred, f0_real_norm)
            loss_loudness = F.mse_loss(loudness_pred, loudness_real_norm)
            loss_ddsp     = loss_f0 + 0.5 * loss_loudness   # f0 pesa mas que loudness
        else:
            zero = mel_orig.new_zeros(())
            loss_f0, loss_loudness, loss_ddsp = zero, zero, zero

        # 4. Total, con beta-annealing sobre el KLD
        total_loss = (self.lambda_mel  * loss_mel) + \
                     (self.lambda_ddsp * loss_ddsp) + \
                     (self.beta        * loss_kld)

        loss_dict = {
            'loss':     total_loss,
            'mel':      loss_mel.item(),
            'ddsp':     loss_ddsp.item(),
            'f0':       loss_f0.item(),
            'loudness': loss_loudness.item(),
            'kld':      loss_kld.item(),
            'beta':     self.beta,
        }
        return total_loss, loss_dict