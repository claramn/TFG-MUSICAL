import torch
import torch.nn as nn
import torch.nn.functional as F
"""
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

"""