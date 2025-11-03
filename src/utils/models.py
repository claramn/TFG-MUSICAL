# ----------------------------------------------------
# Helper functions src/models.py
# ----------------------------------------------------

from math import floor
import torch

MAX_MAGNITUDE = 165.65298461914062
MAX_PHASE = torch.pi

def compute_magnitude_and_phase(stft_spec, normalize=False):
    magnitude = stft_spec.abs()  
    phase = torch.angle(stft_spec)
    if normalize:
        magnitude /= MAX_MAGNITUDE
        phase /= MAX_PHASE
    return magnitude, phase

def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a convolutional layer based on the formula:
        `output size = 1 + floor((input_size + 2 * padding - kernel_size) / stride)`
    """
    def _compute_conv2D_output_size(input_size, kernel_size, stride, padding):
        return 1 + floor((input_size + 2 * padding - kernel_size) / stride)
    
    if stride == 0: raise ZeroDivisionError("Cannot compute output size with stride = 0")
    output_height = _compute_conv2D_output_size(input_size[0], kernel_size[0], stride[0], padding[0])
    output_width = _compute_conv2D_output_size(input_size[1], kernel_size[1], stride[1], padding[1])
    return (output_height, output_width)

def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a transposed convolutional layer based on the formula:
        `output size = (input_size - 1) * stride - 2 * padding + kernel_size`
    """
    def _compute_convTransposed2D_output_size(input_size, kernel_size, stride, padding):
        return (input_size - 1) * stride - 2 * padding + kernel_size
    
    output_height = _compute_convTransposed2D_output_size(input_size[0], kernel_size[0], stride[0], padding[0])
    output_width = _compute_convTransposed2D_output_size(input_size[1], kernel_size[1], stride[1], padding[1])
    return (output_height, output_width)

def adjust_shape(x, target_size, pad_mode='hold'):
    """
    Adjust a 4D tensor to (B, C, H_target, W_target) by cropping or padding.

    Args:
        x           -- input Tensor of shape (B, C, H, W)
        target_size -- (H_target, W_target)
        pad_mode    -- one of:
                        'hold'    : repeat the last row/col
                        'tail'    : replay the last d rows/cols as a block
                        'reflect' : mirror the last d rows/cols
    Returns:
        Tensor of shape (B, C, H_target, W_target)
    """
    assert pad_mode in ('hold', 'tail', 'reflect'), f"unknown pad_mode {pad_mode}"

    B, C, H, W = x.shape
    H_t, W_t = target_size

    # --- adjust height ---
    if H > H_t:
        x = x[:, :, :H_t, :]
    elif H < H_t:
        d = H_t - H
        if pad_mode == 'hold':
            pad = x[:, :, -1:, :].repeat(1, 1, d, 1)
        elif pad_mode == 'tail':
            pad = x[:, :, -d:, :]  # shape [B,C,d,W]
        else:  # reflect
            pad = x[:, :, -d:, :].flip(2)
        x = torch.cat([x, pad], dim=2)

    # --- adjust width ---
    if W > W_t:
        x = x[:, :, :, :W_t]
    elif W < W_t:
        d = W_t - W
        if pad_mode == 'hold':
            pad = x[:, :, :, -1:].repeat(1, 1, 1, d)
        elif pad_mode == 'tail':
            pad = x[:, :, :, -d:]  # shape [B,C,H,d]
        else:  # reflect
            pad = x[:, :, :, -d:].flip(3)
        x = torch.cat([x, pad], dim=3)

    return x