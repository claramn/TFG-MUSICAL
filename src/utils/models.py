import torch.nn as nn
from src.utils.models import *
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels, variational=False):
        super().__init__()

        # Is the encoder serving an autoencoder or a variational autoencoder?
        self.variational = variational

        conv_kernel_size = (3, 3)
        conv_stride = (2, 2)  # Stride 2 for downsampling
        conv_padding = (1, 1)

        self.sizes = [input_size]
        current_size = input_size

        blocks = []
        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size, stride=conv_stride,padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            self.sizes.append(current_size)

            if variational: # BatchNorm can hurt VAE
                block = nn.Sequential(conv, nn.ReLU())
            else:
                block = nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(channels[i]))

            blocks.append(block)

        self.encoder = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()

        # If varational=True, fc1 represents the mean layer
        self.fc1 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

        # fc2 represents the log_var layer
        self.fc2 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)

        if self.variational:
            mu = self.fc1(x)
            log_var = self.fc2(x)
            return x, mu, log_var
        
        x = self.fc1(x)
        return x

    def get_sizes(self):
        return self.sizes


class Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels, variational=False):
        super().__init__()

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        output_padding = 0

        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))

        expected_size = rev_sizes[0]
        self.fc = nn.Linear(latent_dim, rev_channels[0] * expected_size[0] * expected_size[1])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(rev_channels[0], expected_size[0], expected_size[1]))

        deconv_blocks = []
        current_size = expected_size    #empieza en rev_sizes[0]
        for i in range(1, len(rev_sizes)):
            target = rev_sizes[i]  # el tamaño que queremos alcanzar en esta capa
            calc_no_op = compute_convTranspose2D_output_size(
                current_size, kernel_size, stride, padding
            )   #cálculo del size esperado sin output_padding

            #decidir output padding por eje (0 o 1)
            op_h = target[0] - calc_no_op[0]
            op_w = target[1] - calc_no_op[1]
            op_h = 1 if op_h > 0 else 0
            op_w = 1 if op_w > 0 else 0
            output_padding = (op_h, op_w)
            #opcional: debug para ver tamaños capa a capa
            
            #crear la capa transposta con ese output padding, antes output padding era siempre 0
            deconv = nn.ConvTranspose2d(
                rev_channels[i - 1],
                rev_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )

            # We don't add a ReLU in last layer in the case of a Variational Decoder
            if i < len(rev_sizes) - 1 or not variational:
                block = nn.Sequential(deconv, nn.ReLU())
            else:
                block = nn.Sequential(deconv)

            deconv_blocks.append(block)
            
            #actualizar current_size para la siguiente capa. nuestro helper no acepta output padding, recomputa a mano
            current_size = (
                calc_no_op[0] + output_padding[0],
                calc_no_op[1] + output_padding[1],
            )
            
            assert current_size == target, f"Mismatch en capa {i}: got {current_size} vs target {target}"
        self.decoder = nn.Sequential(*deconv_blocks) #, nn.Signmoid())

    def forward(self, x):
       # print("[DEC FWD] input z:", x.shape)
        x = self.fc(x)
        x = self.unflatten(x)
        ##print("[DEC FWD] after unflatten:", x.shape)
        x = self.decoder(x)
       # print("[DEC FWD] after convT stack:", x.shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()

        if not channels:
            raise ValueError('channels argument in AutoEncoder class must be valid')

        self.input_size = input_size
        self.channels = channels

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
      #  print("[ENCODER] sizes:", sizes)    #para ver los tamaños, t dice como van quedando las h, w en cada capa
        self.decoder = Decoder(sizes, latent_dim, channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        #print("[AE FWD] x:", x.shape, "rec:", reconstructed.shape)

        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()
        
        if not channels:
            raise ValueError('channels argument in VAE class must be valid')

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels

        self.encoder = Encoder(input_size, latent_dim, self.channels, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, self.channels, variational=True)  

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        Z = mean + eps * std
        return Z
    
    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        return kld

    def calculate_ELBO_terms(self, x, sigma=1.0):
        B, C, H, W = x.shape

        # Encode to get posterior parameters
        _, mean, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-20, max=20)

        # Sample z ~ q(z|x)
        z = self.reparameterization(mean, log_var)

        # Decode to reconstruction
        x_hat = self.decoder(z)
        x_hat = adjust_shape(x_hat, (H, W), pad_mode='reflect')  # [B,C,H,W]

        # Reconstruction term
        recon = - 1/(2*sigma**2) * F.mse_loss(x_hat, x, reduction='none').view(B, -1).sum(dim=1)

        # KL divergence term
        kld = self.compute_kld(mean, log_var) # (B,)

        return recon, kld

    def forward(self, x, sigma=1.0, return_reconstruction=False):
        """
        B: batch size
        C: number of channels (normalmente 1 en espectrogramas)
        H: height de la imagen (frecuencias en el caso de espectrogramas)
        W: width de la imagen (tiempo en el caso de espectrogramas)
        """

        B,C,H,W = x.shape
        _,mean, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-20, max=20)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        x_hat = adjust_shape(x_hat, (H, W), pad_mode='reflect')
        kld = self.compute_kld(mean, log_var)

        if return_reconstruction:
            return x_hat, kld

        recon_term = -1/(2*sigma**2) * F.mse_loss(
            x_hat, x, reduction='none'
        ).view(B, -1).sum(dim=1)
        return recon_term, kld

    def loss_function(self, recon, kld):
        # We return -ELBO, since we'retrying to maximize ELBO
        return (-recon + kld).mean()


def compute_magnitude_and_phase(stft_spec):
    """
    Compute magnitude and phase from complex STFT spectrogram.
    stft_spec: torch.Tensor, shape [B, F, T] or [B, 2*F, T] if complex
    Returns: mag, phase
    """
    if stft_spec.dtype == torch.complex64:
        mag = torch.abs(stft_spec)
        phase = torch.angle(stft_spec)
    else:
        # Assuming real and imaginary parts are concatenated
        # stft_spec shape [B, 2, F, T], where [:,0,:,:] is real, [:,1,:,:] is imag
        real = stft_spec[:, 0, :, :]
        imag = stft_spec[:, 1, :, :]
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
    return mag, phase


def compute_magnitude_and_phase_sin_cos(stft_spec):
    """
    Compute magnitude and phase using sin/cos from complex STFT spectrogram.
    stft_spec: torch.Tensor, shape [B, F, T] complex or [B, 2, F, T]
    Returns: mag, sin_phase, cos_phase
    """
    mag, phase = compute_magnitude_and_phase(stft_spec)
    sin_phase = torch.sin(phase)
    cos_phase = torch.cos(phase)
    return mag, sin_phase, cos_phase


def adjust_shape(x, target_shape, pad_mode='constant', pad_value=0):
    """
    Adjust the shape of x to match target_shape by padding or cropping.
    x: torch.Tensor, shape [B, C, H, W]
    target_shape: tuple (H, W)
    """
    B, C, H, W = x.shape
    target_H, target_W = target_shape
    
    if H < target_H:
        pad_H = target_H - H
        x = F.pad(x, (0, 0, 0, pad_H), mode=pad_mode, value=pad_value)
    elif H > target_H:
        x = x[:, :, :target_H, :]
    
    if W < target_W:
        pad_W = target_W - W
        x = F.pad(x, (0, pad_W, 0, 0), mode=pad_mode, value=pad_value)
    elif W > target_W:
        x = x[:, :, :, :target_W]
    
    return x


def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    """
    Compute output size of a 2D convolution.
    input_size: tuple (H, W)
    """
    H_in, W_in = input_size
    H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return (H_out, W_out)

