import torch.nn as nn
from torch.distributions import Normal
from src.utils.models import *
import torch.nn.functional as F
import torch
from src.losses import mse_loss

class Latent_Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels, variational=False):
        super().__init__()
        self.variational = variational
        conv_kernel_size = (3, 3)
        conv_stride = (2, 2)
        conv_padding = (1, 1)
        self.sizes = [input_size]
        current_size = input_size
        blocks = []
        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            self.sizes.append(current_size)
            if variational:
                block = nn.Sequential(conv, nn.ReLU())
            else:
                block = nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(channels[i]))
            blocks.append(block)
        self.encoder = nn.Sequential(*blocks)

        if variational:
            self.conv_mu = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)
            self.conv_log_var = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)
        else:
            self.conv_out = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)         # [B, C, H, W]
        if self.variational:
            mu = self.conv_mu(x)        # [B, latent_dim, H, W]
            log_var = self.conv_log_var(x)  # [B, latent_dim, H, W]
            return x, mu, log_var
        x = self.conv_out(x)        # [B, latent_dim, H, W]
        return x

    def get_sizes(self):
        return self.sizes


class Latent_Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels, variational=False):
        super().__init__()
        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))
        expected_size = rev_sizes[0]

        # Conv 1x1 para proyectar latent_dim → primer número de canales del decoder
        self.conv_in = nn.Conv2d(latent_dim, rev_channels[0], kernel_size=1)

        deconv_blocks = []
        current_size = expected_size
        for i in range(1, len(rev_sizes)):
            target = rev_sizes[i]
            calc_no_op = compute_convTranspose2D_output_size(
                current_size, kernel_size, stride, padding
            )
            op_h = target[0] - calc_no_op[0]
            op_w = target[1] - calc_no_op[1]
            op_h = 1 if op_h > 0 else 0
            op_w = 1 if op_w > 0 else 0
            output_padding = (op_h, op_w)

            deconv = nn.ConvTranspose2d(
                rev_channels[i - 1],
                rev_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )

            if i < len(rev_sizes) - 1 or not variational:
                block = nn.Sequential(deconv, nn.ReLU())
            else:
                block = nn.Sequential(deconv)
            deconv_blocks.append(block)

            current_size = (
                calc_no_op[0] + output_padding[0],
                calc_no_op[1] + output_padding[1],
            )
            assert current_size == target, f"Mismatch en capa {i}: got {current_size} vs target {target}"

        self.decoder = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        x = self.conv_in(x)   # [B, latent_dim, H, W] → [B, rev_channels[0], H, W]
        x = self.decoder(x)   # [B, C, H, W] → imagen reconstruida
        return x

class Latent_AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()

        if not channels:
            raise ValueError('channels argument in AutoEncoder class must be valid')

        self.input_size = input_size
        self.channels = channels

        self.encoder = Latent_Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
      #  print("[ENCODER] sizes:", sizes)    #para ver los tamaños, t dice como van quedando las h, w en cada capa
        self.decoder = Latent_Decoder(sizes, latent_dim, channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        #print("[AE FWD] x:", x.shape, "rec:", reconstructed.shape)

        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        return reconstructed

    def loss_function(self, x, reconstructed):
        return mse_loss(reconstructed, x)

class latent_VAE(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()
        
        if not channels:
            raise ValueError('channels argument in VAE class must be valid')

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels

        self.normal = Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        self.encoder = Latent_Encoder(input_size, latent_dim, self.channels, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = Latent_Decoder(sizes, latent_dim, self.channels, variational=True)  

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = self.normal.sample(mean.shape)
        Z = mean + eps * std
        return Z.to(device=std.device)
    
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

        # Compute loss
        loss = vae_loss(x_hat, x, mean, log_var, sigma)

        return loss

    def forward(self, x):
        return self.calculate_ELBO_terms(x)

