"""
sDCTFT (Selective DCT Fine-Tuning) parametrization module.

Implements Algorithm 1 from:
  "MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection"
  arXiv:2410.09103 / 2505.23870

Usage:
    import torch.nn.utils.parametrize as P
    P.register_parametrization(linear_layer, 'weight',
                                sDCTFTParametrization(linear_layer.weight, n=700))
"""

import math
import numpy as np
import torch
import torch.nn as nn

from .dct import dct2d, idct2d


class sDCTFTParametrization(nn.Module):
    """PyTorch parametrization that adds a learnable sDCTFT update to a weight matrix.

    The effective weight during forward pass is:
        W_eff = W_frozen + iDCT(ΔW_F) * alpha

    where ΔW_F is sparse in frequency space (only n selected components are
    trainable; all others are fixed at zero).

    Args:
        weight:  Original weight tensor (any shape; reshaped to 2D internally).
        n:       Total number of trainable DCT coefficients per layer.
        delta:   Fraction of n selected by energy ranking (default 0.7).
                 Remaining (1-delta)*n are selected randomly per band.
        alpha:   Fixed scaling factor applied to the iDCT weight update.
        seed:    RNG seed for the random portion of the selection.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        n: int,
        delta: float = 0.7,
        alpha: float = 16.0,
        seed: int = 42,
    ):
        super().__init__()
        self.alpha = alpha
        self.original_shape = weight.shape

        W_2d = weight.detach().float().reshape(weight.shape[0], -1)
        self.M, self.N = W_2d.shape

        # Compute 2D DCT-II of the base weight (for energy-based selection only)
        W_F = dct2d(W_2d.unsqueeze(0)).squeeze(0)  # (M, N)

        # Select n frequency indices hierarchically
        flat_indices, n_actual = self._select_indices(
            W_F.cpu().numpy(), n, delta, seed
        )
        self.register_buffer(
            "flat_indices", torch.tensor(flat_indices, dtype=torch.long)
        )

        # Trainable parameters: n selected DCT-domain coefficients (Kaiming init)
        self.dW_params = nn.Parameter(torch.empty(n_actual))
        nn.init.kaiming_uniform_(self.dW_params.unsqueeze(0), a=math.sqrt(5))
        self.dW_params.data = self.dW_params.data.squeeze(0)

    # ------------------------------------------------------------------
    # Index selection
    # ------------------------------------------------------------------

    def _select_indices(
        self,
        W_F_np: np.ndarray,
        n: int,
        delta: float,
        seed: int,
    ):
        """Hierarchical frequency selection (Algorithm 1, lines 2-5).

        Three concentric frequency bands defined by Euclidean distance from
        the DC origin:
            d_max = sqrt((M/2)^2 + (N/2)^2)     (Eq. 4 – uses half-dimensions)
            low  = d ≤ d_max/3
            mid  = d_max/3 < d ≤ 2*d_max/3
            high = d > 2*d_max/3

        Within each band, top delta*n_band indices are selected by DCT
        energy magnitude, the rest are selected randomly.
        """
        M, N = W_F_np.shape
        d_max = math.sqrt((M / 2) ** 2 + (N / 2) ** 2)

        ii, jj = np.mgrid[0:M, 0:N]
        distances = np.sqrt(ii**2 + jj**2)

        # Partition boundaries
        bounds = [0.0, d_max / 3.0, 2.0 * d_max / 3.0, d_max + 1e-6]
        band_masks = [
            (distances >= bounds[i]) & (distances < bounds[i + 1])
            for i in range(3)
        ]

        # n per band (distribute evenly; give remainder to high-freq band)
        n_base = n // 3
        n_per_band = [n_base, n_base, n - 2 * n_base]

        rng = np.random.default_rng(seed)
        energies_sq = W_F_np**2
        all_selected = []

        for mask, n_band in zip(band_masks, n_per_band):
            coords = np.argwhere(mask)  # (num_in_band, 2)
            if len(coords) == 0 or n_band <= 0:
                continue

            n_band = min(n_band, len(coords))
            n_energy = min(int(math.ceil(n_band * delta)), len(coords))
            n_random = n_band - n_energy

            band_energies = energies_sq[mask]
            sorted_idx = np.argsort(-band_energies)

            energy_sel = coords[sorted_idx[:n_energy]]

            if n_random > 0 and n_energy < len(coords):
                remaining = coords[sorted_idx[n_energy:]]
                rand_pick = rng.choice(
                    len(remaining),
                    size=min(n_random, len(remaining)),
                    replace=False,
                )
                random_sel = remaining[rand_pick]
                band_selected = np.vstack([energy_sel, random_sel])
            else:
                band_selected = energy_sel

            all_selected.append(band_selected)

        if not all_selected:
            raise ValueError("No frequency indices selected; increase n or check weight shape.")

        all_coords = np.vstack(all_selected)
        flat_indices = all_coords[:, 0] * N + all_coords[:, 1]
        return flat_indices, len(flat_indices)

    # ------------------------------------------------------------------
    # PyTorch parametrize forward
    # ------------------------------------------------------------------

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Called by nn.utils.parametrize on every forward pass.

        Returns:
            W_eff = W_frozen + iDCT(ΔW_F) * alpha   (same dtype/shape as weight)
        """
        orig_dtype = weight.dtype
        W_base = weight.detach().float().reshape(self.M, self.N)

        # Build sparse DCT update in float32
        dW_F = torch.zeros(
            self.M * self.N, dtype=torch.float32, device=weight.device
        )
        dW_F.scatter_(0, self.flat_indices, self.dW_params.float())
        dW_F = dW_F.reshape(1, self.M, self.N)

        # iDCT → spatial weight update
        dW_T = idct2d(dW_F).squeeze(0)  # (M, N)

        W_eff = (W_base + dW_T * self.alpha).reshape(self.original_shape)
        return W_eff.to(orig_dtype)

    def extra_repr(self) -> str:
        n = len(self.flat_indices)
        return (
            f"shape={tuple(self.original_shape)}, "
            f"n={n}, alpha={self.alpha}"
        )


# ---------------------------------------------------------------------------
# Helpers: apply / remove sDCTFT to a model
# ---------------------------------------------------------------------------

import torch.nn.utils.parametrize as P


def apply_sdctft(
    model: nn.Module,
    n: int,
    delta: float = 0.7,
    alpha: float = 16.0,
    target_suffixes: tuple[str, ...] = (
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
        "proj_in", "proj_out",
    ),
    min_size: int = 64,
) -> list[str]:
    """Wrap eligible linear layers in a model with sDCTFT parametrizations.

    Args:
        model:           PyTorch model (UNet, DiT, …)
        n:               Number of trainable DCT coefficients per layer.
        delta:           Energy fraction (0–1).
        alpha:           Scaling factor.
        target_suffixes: Layer name suffixes to wrap (attention + FF projections).
        min_size:        Skip layers whose weight has fewer than min_size elements.

    Returns:
        List of wrapped layer names.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    wrapped = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.numel() < min_size:
            continue
        # Check if name ends with any of the target suffixes
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue

        try:
            param = sDCTFTParametrization(
                module.weight, n=n, delta=delta, alpha=alpha
            )
            P.register_parametrization(module, "weight", param)
            wrapped.append(name)
        except Exception as exc:
            print(f"[sDCTFT] Skipping {name}: {exc}")

    return wrapped


def remove_sdctft(model: nn.Module) -> None:
    """Remove all sDCTFT parametrizations and restore original weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and P.is_parametrized(module, "weight"):
            P.remove_parametrizations(module, "weight", leave_parametrized=True)


def get_sdctft_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable sDCTFT parameters across the model."""
    params = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and P.is_parametrized(module, "weight"):
            for p_module in module.parametrizations.weight:
                if isinstance(p_module, sDCTFTParametrization):
                    params.append(p_module.dW_params)
    return params
