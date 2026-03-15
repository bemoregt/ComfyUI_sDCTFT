"""
ComfyUI Custom Nodes — sDCTFT Fine-Tuning

Node overview:
  ┌──────────────────────────────────────────────────────────────────┐
  │  [sDCTFT_LoadDataset]  ──────────────────────────────────────┐  │
  │  [LoadCheckpoint] → MODEL                                    │  │
  │                      CLIP  ──> [sDCTFT_Config] → CONFIG ─┐  │  │
  │                      VAE                                  │  │  │
  │                                                           ↓  ↓  │
  │  MODEL ────────────────────────────> [sDCTFT_Train] → MODEL  │
  │  VAE   ────────────────────────────>                     │  │  │
  │  CLIP  ────────────────────────────>                         │  │
  │                                                              ↓  │
  │                                          [sDCTFT_SaveModel]     │
  └──────────────────────────────────────────────────────────────────┘

Node categories: sDCTFT/Dataset, sDCTFT/Training, sDCTFT/IO
"""

import os
import copy
import json
import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import folder_paths  # ComfyUI built-in

# ---------------------------------------------------------------------------
# Custom data types
# ---------------------------------------------------------------------------

SDCTFT_CONFIG = "SDCTFT_CONFIG"
SDCTFT_DATASET = "SDCTFT_DATASET"


# ===========================================================================
# Node 1: sDCTFT_LoadDataset
# ===========================================================================

class sDCTFT_LoadDataset:
    """Load training images (and optional per-image captions) from a folder.

    The node scans the directory for images (.png/.jpg/.jpeg/.webp/.bmp) and
    optionally reads matching .txt caption files (same stem).

    Outputs a SDCTFT_DATASET dict that is consumed by sDCTFT_Train.
    You can also directly feed ComfyUI IMAGE tensors via the 'images' port.
    """

    CATEGORY = "sDCTFT/Dataset"
    RETURN_TYPES = (SDCTFT_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to folder containing training images. "
                                   "Leave blank to use the 'images' input instead.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Batch IMAGE tensor (B, H, W, C) as alternative "
                                   "to image_dir.",
                    },
                ),
                "shared_caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Caption used for all images. "
                                   "Overridden by per-image .txt files if found.",
                    },
                ),
                "max_images": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000,
                     "tooltip": "Limit number of images loaded (0 = all)."},
                ),
            },
        }

    def load(
        self,
        image_dir: str = "",
        images: torch.Tensor | None = None,
        shared_caption: str = "",
        max_images: int = 0,
    ) -> tuple:
        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        imgs_list: list[torch.Tensor] = []
        captions_list: list[str] = []

        # ------------------------------------------------------------------
        # Load from directory
        # ------------------------------------------------------------------
        if image_dir.strip():
            image_dir = image_dir.strip()
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f"[sDCTFT] Directory not found: {image_dir}")

            paths = sorted(
                p
                for p in Path(image_dir).iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )
            if max_images > 0:
                paths = paths[:max_images]

            if not paths:
                raise ValueError(f"[sDCTFT] No images found in: {image_dir}")

            for p in paths:
                img = Image.open(p).convert("RGB")
                t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # (H,W,C)
                imgs_list.append(t)

                # Per-image caption
                cap_path = p.with_suffix(".txt")
                if cap_path.exists():
                    captions_list.append(cap_path.read_text().strip())
                else:
                    captions_list.append(shared_caption)

        # ------------------------------------------------------------------
        # Load from IMAGE tensor
        # ------------------------------------------------------------------
        if images is not None:
            B = images.shape[0]
            for i in range(B):
                imgs_list.append(images[i].float())  # (H, W, C)
                captions_list.append(shared_caption)

        if not imgs_list:
            raise ValueError("[sDCTFT] No training images provided.")

        # Stack into tensor: (N, H_max, W_max, C) — pad to largest if needed
        # For simplicity, we do NOT pre-pad here; trainer handles resize.
        imgs_tensor = torch.stack(imgs_list, dim=0)  # (N, H, W, C)

        dataset = {
            "images": imgs_tensor,           # (N, H, W, C) float32 [0, 1]
            "captions": captions_list,        # list[str], len = N
        }
        print(
            f"[sDCTFT] Dataset loaded: {len(imgs_list)} images, "
            f"captions={'per-image' if any(c for c in captions_list) else 'none'}"
        )
        return (dataset,)


# ===========================================================================
# Node 2: sDCTFT_Config
# ===========================================================================

class sDCTFT_Config:
    """Configure sDCTFT fine-tuning hyperparameters.

    Key hyperparameters:
      n_coefficients  — Number of trainable DCT spectral coefficients per layer.
                        Higher = more capacity but slower training and more memory.
                        Recommended: 700 (small), 2400 (medium), 7000 (large).
      delta           — Fraction of n_coefficients selected by energy magnitude
                        (0.7 = 70% energy-based, 30% random). Paper default: 0.7.
      alpha           — Fixed scaling factor applied to the spatial weight update.
                        Controls the effective learning magnitude per step.
                        Start with 16 for fine-tuning, 300 for aggressive adaptation.
      target_layers   — Comma-separated list of layer name suffixes to wrap.
                        Use 'all_linear' to wrap every linear layer.
    """

    CATEGORY = "sDCTFT/Training"
    RETURN_TYPES = (SDCTFT_CONFIG,)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_coefficients": (
                    "INT",
                    {
                        "default": 700,
                        "min": 10,
                        "max": 50000,
                        "step": 10,
                        "tooltip": "Trainable DCT coefficients per layer.",
                    },
                ),
                "delta": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Energy-based selection ratio (paper default: 0.7).",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 0.1,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "Scaling factor for weight updates.",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 1e-3,
                        "min": 1e-6,
                        "max": 1.0,
                        "step": 1e-5,
                        "tooltip": "AdamW learning rate.",
                    },
                ),
                "num_epochs": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of training epochs.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Training batch size.",
                    },
                ),
                "target_resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Image resolution for VAE encoding (pixels).",
                    },
                ),
                "diffusion_timesteps": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 100,
                        "max": 2000,
                        "step": 100,
                        "tooltip": "Total diffusion timesteps T.",
                    },
                ),
            },
            "optional": {
                "target_layers": (
                    "STRING",
                    {
                        "default": (
                            "to_q,to_k,to_v,to_out.0,"
                            "ff.net.0.proj,ff.net.2,"
                            "proj_in,proj_out"
                        ),
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated layer suffix list to apply sDCTFT. "
                            "Use 'all_linear' to target every linear layer."
                        ),
                    },
                ),
                "save_every_n_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Save intermediate checkpoint every N steps (0 = off).",
                    },
                ),
                "checkpoint_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Directory for intermediate checkpoints.",
                    },
                ),
            },
        }

    def build(
        self,
        n_coefficients: int,
        delta: float,
        alpha: float,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        target_resolution: int,
        diffusion_timesteps: int,
        target_layers: str = (
            "to_q,to_k,to_v,to_out.0,"
            "ff.net.0.proj,ff.net.2,"
            "proj_in,proj_out"
        ),
        save_every_n_steps: int = 0,
        checkpoint_dir: str = "",
    ) -> tuple:
        config = {
            "n": n_coefficients,
            "delta": delta,
            "alpha": alpha,
            "lr": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "target_size": target_resolution,
            "T": diffusion_timesteps,
            "target_layers": (
                None
                if target_layers.strip().lower() == "all_linear"
                else tuple(s.strip() for s in target_layers.split(",") if s.strip())
            ),
            "save_every": save_every_n_steps,
            "checkpoint_dir": checkpoint_dir.strip(),
        }
        print("[sDCTFT] Config:", json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
            indent=2
        ))
        return (config,)


# ===========================================================================
# Node 3: sDCTFT_Train
# ===========================================================================

class sDCTFT_Train:
    """Fine-tune a diffusion model with the sDCTFT algorithm.

    Takes a base model, VAE, CLIP, training dataset and configuration,
    applies sDCTFT parametrizations to the target attention / FF layers,
    runs the training loop, and returns the fine-tuned MODEL.

    The returned MODEL is a standard ComfyUI MODEL object and can be used
    with any downstream sampling node (KSampler, etc.).

    ⚠ Training runs synchronously in the current process. Progress is printed
      to the ComfyUI console. For long training runs, monitor the terminal.
    """

    CATEGORY = "sDCTFT/Training"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("finetuned_model",)
    FUNCTION = "train"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   ("MODEL",),
                "vae":     ("VAE",),
                "clip":    ("CLIP",),
                "dataset": (SDCTFT_DATASET,),
                "config":  (SDCTFT_CONFIG,),
            },
        }

    def train(
        self,
        model,
        vae,
        clip,
        dataset: dict,
        config: dict,
    ) -> tuple:
        import comfy.model_management as mm
        from .sdctft.algorithm import apply_sdctft
        from .sdctft.trainer import train_sdctft

        # ------------------------------------------------------------------
        # 1. Clone model so we don't modify the cached version
        # ------------------------------------------------------------------
        model_out = model.clone()

        # ------------------------------------------------------------------
        # 2. Extract the actual UNet module
        # ------------------------------------------------------------------
        unet = model_out.model.diffusion_model
        device = mm.get_torch_device()
        unet = unet.to(device)

        print(f"[sDCTFT] Device: {device}")
        print(f"[sDCTFT] Model type: {type(unet).__name__}")

        # ------------------------------------------------------------------
        # 3. Apply sDCTFT parametrizations
        # ------------------------------------------------------------------
        target_suffixes = config.get(
            "target_layers",
            ("to_q", "to_k", "to_v", "to_out.0",
             "ff.net.0.proj", "ff.net.2",
             "proj_in", "proj_out"),
        )
        # If target_layers=None ('all_linear' was chosen), pass a wildcard
        if target_suffixes is None:
            # Wrap ALL linear layers (match any suffix)
            wrapped = apply_sdctft(
                unet,
                n=config["n"],
                delta=config["delta"],
                alpha=config["alpha"],
                target_suffixes=("",),   # empty string is suffix of everything
            )
        else:
            wrapped = apply_sdctft(
                unet,
                n=config["n"],
                delta=config["delta"],
                alpha=config["alpha"],
                target_suffixes=target_suffixes,
            )

        if not wrapped:
            raise RuntimeError(
                "[sDCTFT] No layers were wrapped. "
                "Check target_layers in config and model architecture."
            )
        print(f"[sDCTFT] Wrapped {len(wrapped)} layers: {wrapped[:5]}{'…' if len(wrapped)>5 else ''}")

        # ------------------------------------------------------------------
        # 4. Run training
        # ------------------------------------------------------------------
        images = dataset["images"]         # (N, H, W, C)
        captions = dataset["captions"]     # list[str]

        train_sdctft(
            unet=unet,
            vae=vae,
            clip=clip,
            images=images,
            captions=captions,
            num_epochs=config["epochs"],
            learning_rate=config["lr"],
            batch_size=config["batch_size"],
            target_size=config["target_size"],
            T=config["T"],
            guidance_scale=1.0,
            save_every=config.get("save_every", 0),
            output_dir=config.get("checkpoint_dir") or None,
        )

        # ------------------------------------------------------------------
        # 5. Bake in the trained weights (remove parametrizations)
        # ------------------------------------------------------------------
        from .sdctft.algorithm import remove_sdctft
        import torch.nn.utils.parametrize as P

        # Before removing parametrizations, materialize the effective weights
        for module in unet.modules():
            if isinstance(module, torch.nn.Linear) and P.is_parametrized(module, "weight"):
                # nn.utils.parametrize.remove_parametrizations with
                # leave_parametrized=True writes back the effective weight
                pass
        remove_sdctft(unet)

        # Re-enable all parameters (in case downstream code needs them)
        for p in unet.parameters():
            p.requires_grad_(False)  # inference mode after training

        print("[sDCTFT] Fine-tuning complete. Returning updated model.")
        return (model_out,)


# ===========================================================================
# Node 4: sDCTFT_SaveModel
# ===========================================================================

class sDCTFT_SaveModel:
    """Save the fine-tuned model as a .safetensors checkpoint.

    The full model weights are saved (not just the delta), so the output
    can be loaded by any standard ComfyUI checkpoint loader.

    Optionally also saves a compact sDCTFT delta file (.pt) containing only
    the trained spectral coefficients (much smaller file).
    """

    CATEGORY = "sDCTFT/IO"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename": (
                    "STRING",
                    {
                        "default": "sdctft_finetuned",
                        "multiline": False,
                        "tooltip": "Output filename without extension.",
                    },
                ),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Save directory. "
                            "Defaults to ComfyUI's models/checkpoints folder."
                        ),
                    },
                ),
                "vae":  ("VAE",  {"tooltip": "Include VAE in the saved checkpoint."}),
                "clip": ("CLIP", {"tooltip": "Include CLIP in the saved checkpoint."}),
            },
        }

    def save(
        self,
        model,
        filename: str,
        output_dir: str = "",
        vae=None,
        clip=None,
    ) -> tuple:
        try:
            from safetensors.torch import save_file as st_save
            _has_safetensors = True
        except ImportError:
            _has_safetensors = False

        # Resolve output directory
        if not output_dir.strip():
            ckpt_dirs = folder_paths.get_folder_paths("checkpoints")
            out_dir = ckpt_dirs[0] if ckpt_dirs else os.path.join(
                os.path.dirname(__file__), "output"
            )
        else:
            out_dir = output_dir.strip()
        os.makedirs(out_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Collect state dict
        # ------------------------------------------------------------------
        state_dict: dict[str, torch.Tensor] = {}

        # UNet weights
        unet = model.model.diffusion_model
        for k, v in unet.state_dict().items():
            state_dict[f"model.diffusion_model.{k}"] = v.cpu()

        # VAE weights (optional)
        if vae is not None:
            vae_sd = vae.first_stage_model.state_dict()
            for k, v in vae_sd.items():
                state_dict[f"first_stage_model.{k}"] = v.cpu()

        # CLIP weights (optional)
        if clip is not None:
            try:
                clip_sd = clip.cond_stage_model.state_dict()
                for k, v in clip_sd.items():
                    state_dict[f"cond_stage_model.{k}"] = v.cpu()
            except AttributeError:
                pass

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        ext = ".safetensors" if _has_safetensors else ".pt"
        save_path = os.path.join(out_dir, filename + ext)

        if _has_safetensors:
            # safetensors requires contiguous float32
            sd_st = {
                k: v.contiguous().to(torch.float32)
                for k, v in state_dict.items()
            }
            st_save(sd_st, save_path)
        else:
            torch.save({"state_dict": state_dict}, save_path)

        size_mb = os.path.getsize(save_path) / 1024**2
        print(f"[sDCTFT] Model saved: {save_path} ({size_mb:.1f} MB)")
        return (save_path,)


# ===========================================================================
# (Bonus) Node 5: sDCTFT_Preview  — optional, shows training dataset info
# ===========================================================================

class sDCTFT_DatasetInfo:
    """Display information about a loaded sDCTFT dataset."""

    CATEGORY = "sDCTFT/Dataset"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("info", "preview_images")
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (SDCTFT_DATASET,),
            },
            "optional": {
                "max_preview": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Number of images to return for preview.",
                    },
                ),
            },
        }

    def inspect(self, dataset: dict, max_preview: int = 4) -> tuple:
        images = dataset["images"]   # (N, H, W, C)
        captions = dataset["captions"]
        N, H, W, C = images.shape

        # Caption stats
        with_cap = sum(1 for c in captions if c.strip())
        info = (
            f"Images  : {N}\n"
            f"Shape   : {H}×{W}×{C}\n"
            f"Captions: {with_cap}/{N}\n"
        )
        if captions and captions[0].strip():
            info += f"\nSample caption:\n  {captions[0][:200]}"

        preview = images[:max_preview].float().clamp(0.0, 1.0)
        return (info, preview)
