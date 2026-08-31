import torch.nn as nn
from torch.distributions import Normal
from src.utils.models import *
import torch.nn.functional as F
import torch

class latent_Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels, strides=None, variational=False):
        super().__init__()
        self.variational = variational

        conv_kernel_size = (3, 3)
        conv_padding = (1, 1)

        if strides is None:
            strides = [(2, 2)] * (len(channels) - 1)
        assert len(strides) == len(channels) - 1

        self.sizes = [input_size]
        current_size = input_size

        blocks = []
        for i in range(1, len(channels)):
            layer_stride = strides[i - 1]
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size,
                              stride=layer_stride, padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, layer_stride, conv_padding)
            self.sizes.append(current_size)

            if variational:
                block = nn.Sequential(conv, nn.ReLU())
            else:
                block = nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(channels[i]))

            blocks.append(block)

        self.encoder = nn.Sequential(*blocks)
        self.fc1 = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        if self.variational:
            mu = self.fc1(x)
            log_var = self.fc2(x)
            return x, mu, log_var
        x = self.fc1(x)
        return x

    def get_sizes(self):
        return self.sizes


class latent_Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels, strides=None, variational=False):
        super().__init__()

        kernel_size = (3, 3)
        padding = (1, 1)

        # strides tiene una entrada por cada capa conv del encoder: len(channels) - 1
        if strides is None:
            strides = [(2, 2)] * (len(channels) - 1)
        assert len(strides) == len(channels) - 1, \
            f"strides debe tener {len(channels)-1} elementos, tiene {len(strides)}"

        # el decoder deshace las capas en orden inverso al encoder
        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))
        rev_strides = list(reversed(strides))

        expected_size = rev_sizes[0]
        self.fc = nn.Conv2d(latent_dim, channels[-1], kernel_size=1)

        deconv_blocks = []
        current_size = expected_size
        for i in range(1, len(rev_sizes)):
            target = rev_sizes[i]
            layer_stride = rev_strides[i - 1]   # <- stride correspondiente a esta capa

            calc_no_op = compute_convTranspose2D_output_size(
                current_size, kernel_size, layer_stride, padding
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
                stride=layer_stride,
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
        x = self.fc(x)
        x = self.decoder(x)
        return x
    
    


class latent_AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()

        if not channels:
            raise ValueError('channels argument in AutoEncoder class must be valid')

        self.input_size = input_size
        self.channels = channels

        self.encoder = latent_Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
      #  print("[ENCODER] sizes:", sizes)    #para ver los tamaños, t dice como van quedando las h, w en cada capa
        self.decoder = latent_Decoder(sizes, latent_dim, channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        #print("[AE FWD] x:", x.shape, "rec:", reconstructed.shape)

        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        return reconstructed

class latent_VAE(nn.Module):
    def __init__(self, input_size, latent_dim, strides=None, channels=None):
        super().__init__()
        
        if not channels:
            raise ValueError('channels argument in VAE class must be valid')

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels

        self.normal = Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        self.encoder = latent_Encoder(input_size, latent_dim, self.channels, strides=strides, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = latent_Decoder(sizes, latent_dim, self.channels, strides=strides, variational=True)

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = self.normal.sample(mean.shape)
        return mean + eps * std
        # return Z.to(device=std.device)
    
    # def compute_kld(self, mu, logvar):
    #     kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
    #     return kld
    
    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp(), dim=(1, 2, 3))
        return kld  # forma [B]

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
        # recon = - 1/(2*sigma**2) * F.mse_loss(x_hat, x, reduction='none').view(B, -1).sum(dim=1)
        recon = F.mse_loss(x_hat, x, reduction='none').mean(dim=(1, 2, 3))  # forma [B]

        # KL divergence term
        kld = self.compute_kld(mean, log_var) # (B,)

        return recon, kld

    """
    def forward(self, x):
        return self.calculate_ELBO_terms(x)
    """ 

    """
    esta funcion define como pasa el dato por el modelo cuando llamamos a model(x)
    en teoria deberia ir mejor pq no nos atamos a mse dentro del modelo y podemos montar una loss d audio meojor luego en el nb
    """
    def forward(self, x):
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
        x_hat = self.decoder(z) #en vez d muestrear directamente de la normal, pasamos el z reparameterizado por el decoder para obtener la reconstrucción, que es lo que realmente nos interesa en el forward. El cálculo de ELBO lo hacemos en la función loss_function, que se llama desde el training loop
        x_hat = adjust_shape(x_hat, (H, W), pad_mode='reflect')
        kld = self.compute_kld(mean, log_var)
        return x_hat, kld

    def loss_function(self, recon, kld):
        # We return -ELBO, since we'retrying to maximize ELBO
        return (recon + kld).mean()


def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
        """
        Compute output size of a 2D convolution.
        input_size: tuple (H, W)
        """
        H_in, W_in = input_size
        H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        return (H_out, W_out)
    
def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding, output_padding=(0, 0)):
    """
    Calcula el tamaño de salida de una capa ConvTranspose2d (Deconvolución).
    Fórmula: H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
    """
    H_in, W_in = input_size
    
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    
    return (H_out, W_out)
