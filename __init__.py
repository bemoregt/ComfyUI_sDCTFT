"""
ComfyUI-sDCTFT
==============
Custom node set for fine-tuning diffusion models using the
sDCTFT (Selective Discrete Cosine Transform Fine-Tuning) algorithm.

Reference:
  "MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection"
  arXiv:2410.09103 / 2505.23870  (ACL 2025 Best Theme Paper)
"""

from .nodes import (
    sDCTFT_LoadDataset,
    sDCTFT_Config,
    sDCTFT_Train,
    sDCTFT_SaveModel,
    sDCTFT_DatasetInfo,
)

NODE_CLASS_MAPPINGS = {
    "sDCTFT_LoadDataset":  sDCTFT_LoadDataset,
    "sDCTFT_Config":       sDCTFT_Config,
    "sDCTFT_Train":        sDCTFT_Train,
    "sDCTFT_SaveModel":    sDCTFT_SaveModel,
    "sDCTFT_DatasetInfo":  sDCTFT_DatasetInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "sDCTFT_LoadDataset":  "Load Dataset (sDCTFT)",
    "sDCTFT_Config":       "Training Config (sDCTFT)",
    "sDCTFT_Train":        "Fine-Tune Model (sDCTFT)",
    "sDCTFT_SaveModel":    "Save Fine-Tuned Model (sDCTFT)",
    "sDCTFT_DatasetInfo":  "Dataset Info (sDCTFT)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(
    "\033[92m[sDCTFT]\033[0m Loaded "
    f"{len(NODE_CLASS_MAPPINGS)} nodes: "
    + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
)
