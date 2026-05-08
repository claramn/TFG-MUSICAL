import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, mel_orig, mel_hat, ddsp_params, features_real, kld):
        
        # Loss Mel (espectrogramas)
        loss_mel = F.l1_loss(mel_hat, mel_orig)

        #2. Loss KLD (espacio latente)
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