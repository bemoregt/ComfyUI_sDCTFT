"""
Diffusion model fine-tuning loop using sDCTFT.

Supports:
  - SD 1.x / 2.x  (UNetModel, context via cross-attention)
  - SDXL           (same UNet but with pooled text conditioning y)

The training follows the standard DDPM v-prediction / epsilon-prediction
objective used in Stable Diffusion:

    L = E[||ε - ε_θ(sqrt(ᾱ_t)·z + sqrt(1-ᾱ_t)·ε, t, c)||²]
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Noise schedule (linear β schedule — matches SD 1.x defaults)
# ---------------------------------------------------------------------------

def build_linear_schedule(
    T: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, device=device) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
    }


def add_noise(
    z: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """Forward diffusion q(z_t | z_0) = sqrt(ᾱ_t)·z + sqrt(1-ᾱ_t)·ε"""
    sqrt_ab = schedule["sqrt_alphas_cumprod"][t].reshape(-1, 1, 1, 1)
    sqrt_1m_ab = schedule["sqrt_one_minus_alphas_cumprod"][t].reshape(-1, 1, 1, 1)
    return sqrt_ab * z + sqrt_1m_ab * noise


# ---------------------------------------------------------------------------
# Image → latent helpers
# ---------------------------------------------------------------------------

def preprocess_images(
    images: torch.Tensor,
    target_size: int = 512,
) -> torch.Tensor:
    """Resize ComfyUI images (B, H, W, C) [0,1] to (B, C, H, W) [0,1] square."""
    # ComfyUI format: (B, H, W, C) float32 [0, 1]
    images = images.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    if images.shape[-1] != target_size or images.shape[-2] != target_size:
        images = F.interpolate(
            images, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
    return images


def encode_images_to_latents(
    vae,
    images_comfy: torch.Tensor,
    target_size: int = 512,
    batch_size: int = 4,
) -> torch.Tensor:
    """Encode ComfyUI IMAGE tensor to VAE latents.

    Args:
        vae:          ComfyUI VAE object
        images_comfy: (B, H, W, C) in [0, 1]
        target_size:  Resize to this square resolution before encoding
        batch_size:   Sub-batch size for VAE encoding

    Returns:
        latents: (B, 4, H//8, W//8)
    """
    # Resize to target_size
    imgs_bchw = preprocess_images(images_comfy, target_size)  # (B, C, H, W) [0,1]

    latents_list = []
    for i in range(0, imgs_bchw.shape[0], batch_size):
        chunk = imgs_bchw[i : i + batch_size]
        # vae.encode expects (B, H, W, C) in [0, 1]
        chunk_bhwc = chunk.permute(0, 2, 3, 1)
        with torch.no_grad():
            lat = vae.encode(chunk_bhwc)
        latents_list.append(lat)

    return torch.cat(latents_list, dim=0)  # (B, 4, H//8, W//8)


# ---------------------------------------------------------------------------
# Text conditioning helpers
# ---------------------------------------------------------------------------

def encode_prompt(
    clip,
    prompt: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a text prompt to CLIP conditioning tensors.

    Returns:
        cond:   (B, seq_len, d_model) tensor
        pooled: (B, d_pool) tensor or None for SD1.x
    """
    tokens = clip.tokenize(prompt)
    with torch.no_grad():
        cond_out = clip.encode_from_tokens(tokens, return_pooled=True)
    if isinstance(cond_out, tuple):
        cond, pooled = cond_out
    else:
        cond, pooled = cond_out, None

    # Expand to batch
    cond = cond.expand(batch_size, -1, -1)
    if pooled is not None:
        pooled = pooled.expand(batch_size, -1)
    return cond, pooled


def get_uncond_conditioning(
    clip,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Get unconditional (empty prompt) conditioning."""
    return encode_prompt(clip, "", batch_size)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_sdctft(
    unet: torch.nn.Module,
    vae,
    clip,
    images: torch.Tensor,
    captions: list[str] | None,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    target_size: int,
    T: int,
    guidance_scale: float,
    save_every: int,
    output_dir: str | None,
    progress_callback=None,
) -> None:
    """Fine-tune a UNet with sDCTFT.

    Args:
        unet:              The diffusion UNet module (already has sDCTFT parametrizations).
        vae:               ComfyUI VAE instance.
        clip:              ComfyUI CLIP instance.
        images:            (N, H, W, C) training images in [0, 1].
        captions:          List of N prompts, or None for unconditional.
        num_epochs:        Number of training epochs.
        learning_rate:     AdamW learning rate.
        batch_size:        Training batch size.
        target_size:       Spatial resolution for latent encoding (pixels).
        T:                 Number of diffusion timesteps (default 1000).
        guidance_scale:    Unused during training (kept for future CFG-aware loss).
        save_every:        Save checkpoint every N steps (0 = never).
        output_dir:        Directory for intermediate checkpoints.
        progress_callback: Optional fn(step, total_steps, loss) for UI feedback.
    """
    from .algorithm import get_sdctft_params

    device = next(unet.parameters()).device
    schedule = build_linear_schedule(T=T, device=device)

    # Collect only trainable sDCTFT parameters
    sdctft_params = get_sdctft_params(unet)
    if not sdctft_params:
        raise RuntimeError(
            "No sDCTFT parameters found. "
            "Did you call apply_sdctft() before training?"
        )

    total_trainable = sum(p.numel() for p in sdctft_params)
    total_all = sum(p.numel() for p in unet.parameters())
    print(
        f"[sDCTFT] Trainable params: {total_trainable:,} / {total_all:,} "
        f"({100*total_trainable/total_all:.4f}%)"
    )

    optimizer = torch.optim.AdamW(sdctft_params, lr=learning_rate, weight_decay=1e-4)

    # Pre-encode all images to latents (no grad needed)
    print("[sDCTFT] Encoding training images to latents …")
    all_latents = encode_images_to_latents(vae, images, target_size=target_size)
    N = all_latents.shape[0]
    print(f"[sDCTFT] Encoded {N} images → latents shape {tuple(all_latents.shape)}")

    # Pre-encode captions
    if captions is not None and len(captions) == N:
        print("[sDCTFT] Encoding captions …")
        all_conds = []
        all_pooled = []
        for cap in captions:
            cond, pooled = encode_prompt(clip, cap, batch_size=1)
            all_conds.append(cond)
            all_pooled.append(pooled)
        all_conds = torch.cat(all_conds, dim=0).to(device)   # (N, seq, d)
        all_pooled = (
            torch.cat(all_pooled, dim=0).to(device)          # (N, d_pool)
            if all_pooled[0] is not None else None
        )
    else:
        # Single unconditional or single shared caption
        shared_prompt = (captions[0] if captions and len(captions) == 1 else "")
        print(f"[sDCTFT] Using shared prompt: '{shared_prompt}'")
        all_conds, all_pooled = encode_prompt(clip, shared_prompt, batch_size=N)
        all_conds = all_conds.to(device)
        all_pooled = all_pooled.to(device) if all_pooled is not None else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    unet.train()
    step = 0
    total_steps = num_epochs * math.ceil(N / batch_size)
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=device)

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            z0 = all_latents[idx].to(device)               # (B, 4, h, w)
            cond = all_conds[idx] if all_conds.shape[0] == N else all_conds
            y = all_pooled[idx] if all_pooled is not None and all_pooled.shape[0] == N else all_pooled

            B = z0.shape[0]
            noise = torch.randn_like(z0)
            t = torch.randint(0, T, (B,), device=device)
            z_t = add_noise(z0, noise, t, schedule)

            # UNet forward
            # ComfyUI UNet forward signature:
            #   diffusion_model(x, timesteps, context, y=None, **kwargs)
            unet_kwargs = {}
            if y is not None:
                unet_kwargs["y"] = y

            noise_pred = unet(z_t, t, context=cond, **unet_kwargs)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sdctft_params, max_norm=1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t0
                speed = step / elapsed
                eta = (total_steps - step) / speed if speed > 0 else 0
                print(
                    f"[sDCTFT] epoch {epoch+1}/{num_epochs}  "
                    f"step {step}/{total_steps}  "
                    f"loss={loss.item():.5f}  "
                    f"eta={eta/60:.1f}min"
                )

            if progress_callback is not None:
                progress_callback(step, total_steps, loss.item())

            # Checkpoint
            if save_every > 0 and step % save_every == 0 and output_dir:
                ckpt_path = Path(output_dir) / f"sdctft_step{step}.pt"
                _save_sdctft_state(unet, str(ckpt_path))
                print(f"[sDCTFT] Checkpoint saved: {ckpt_path}")

    unet.eval()
    print(f"[sDCTFT] Training complete. {step} steps in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_sdctft_state(unet: torch.nn.Module, path: str) -> None:
    """Save only the sDCTFT trainable parameters (not full model weights)."""
    from .algorithm import get_sdctft_params
    import torch.nn.utils.parametrize as P

    state = {}
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear) and P.is_parametrized(module, "weight"):
            for p_mod in module.parametrizations.weight:
                from .algorithm import sDCTFTParametrization
                if isinstance(p_mod, sDCTFTParametrization):
                    state[name] = {
                        "dW_params": p_mod.dW_params.detach().cpu(),
                        "flat_indices": p_mod.flat_indices.cpu(),
                        "M": p_mod.M,
                        "N": p_mod.N,
                        "alpha": p_mod.alpha,
                        "original_shape": p_mod.original_shape,
                    }
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(state, path)
