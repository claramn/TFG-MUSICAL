import torch.nn as nn
from torch.distributions import Normal
from src.utils.models import *


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
        for i in range(1, len(rev_sizes)):
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
        self.decoder = nn.Sequential(*deconv_blocks) #, nn.Signmoid())

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
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
        self.decoder = Decoder(sizes, latent_dim, channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
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

        self.normal = Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        self.encoder = Encoder(input_size, latent_dim, self.channels, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, self.channels, variational=True)  

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

        # Reconstruction term
        recon = - 1/(2*sigma**2) * F.mse_loss(x_hat, x, reduction='none').view(B, -1).sum(dim=1)

        # KL divergence term
        kld = self.compute_kld(mean, log_var) # (B,)

        return recon, kld

    def forward(self, x):
        return self.calculate_ELBO_terms(x)

    def loss_function(self, recon, kld):
        # We return -ELBO, since we'retrying to maximize ELBO
        return (-recon + kld).mean()

