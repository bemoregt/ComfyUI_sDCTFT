"""
Differentiable 2D DCT-II (orthonormal norm) using PyTorch FFT.
Matches scipy.fft.dctn(x, type=2, norm='ortho').
Fully compatible with torch.autograd.
"""

import math
import numpy as np
import torch


def dct1d(x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
    """1D DCT-II along last dimension.

    Args:
        x:    (..., N) real tensor
        norm: None → unnormalized;  'ortho' → orthonormal

    Returns:
        X: (..., N) DCT-II coefficients
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().reshape(-1, N)

    # Makhoul reordering: even-indexed first, then odd-indexed reversed
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # FFT
    Vc = torch.view_as_real(torch.fft.fft(v.to(torch.float32), dim=1))

    # Phase shift weights:  W[k] = exp(-j * pi * k / (2N))
    k = torch.arange(N, dtype=torch.float32, device=x.device).unsqueeze(0)
    W_r = torch.cos(-math.pi * k / (2.0 * N))
    W_i = torch.sin(-math.pi * k / (2.0 * N))

    # X[k] = 2 * Re(V[k] * W[k])
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2.0
        V[:, 1:] /= math.sqrt(N / 2.0) * 2.0

    V = 2.0 * V.reshape(x_shape)
    return V.to(x_shape[0] if isinstance(x_shape[0], type) else x.dtype)


def idct1d(X: torch.Tensor, norm: str | None = None) -> torch.Tensor:
    """1D iDCT-II along last dimension (inverse of dct1d).

    Args:
        X:    (..., N) DCT-II coefficients
        norm: None → unnormalized inverse;  'ortho' → orthonormal inverse

    Returns:
        x: (..., N) reconstructed values
    """
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().reshape(-1, N).to(torch.float32) / 2.0

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2.0
        X_v[:, 1:] *= math.sqrt(N / 2.0) * 2.0

    k = torch.arange(N, dtype=torch.float32, device=X.device).unsqueeze(0)
    W_r = torch.cos(math.pi * k / (2.0 * N))
    W_i = torch.sin(math.pi * k / (2.0 * N))

    # Build Hermitian-symmetric V for IFFT
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0.0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.view_as_complex(torch.stack([V_r, V_i], dim=-1))
    v = torch.fft.ifft(V, dim=1).real

    x = v.new_zeros(v.shape)
    x[:, ::2] = v[:, : N - (N // 2)]
    x[:, 1::2] = v.flip([1])[:, : N // 2]

    return x.reshape(x_shape)


def dct2d(x: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """2D DCT-II (separable row then column transform).

    Args:
        x:    (..., H, W) real tensor
        norm: 'ortho' (default) → orthonormal normalization

    Returns:
        X_F: (..., H, W) 2D DCT-II coefficients
    """
    return dct1d(dct1d(x, norm=norm).transpose(-2, -1), norm=norm).transpose(-2, -1)


def idct2d(X: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """2D iDCT-II (inverse of dct2d).

    Args:
        X_F: (..., H, W) 2D DCT-II coefficients
        norm: 'ortho' (default) → orthonormal normalization

    Returns:
        x: (..., H, W) reconstructed values
    """
    return idct1d(idct1d(X, norm=norm).transpose(-2, -1), norm=norm).transpose(-2, -1)
