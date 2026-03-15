# ComfyUI-sDCTFT

Fine-tune diffusion models inside ComfyUI using the **sDCTFT** (Selective Discrete Cosine Transform Fine-Tuning) algorithm from the ACL 2025 Best Theme Paper award winner *MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection* ([arXiv:2410.09103](https://arxiv.org/abs/2410.09103)).

sDCTFT achieves competitive fine-tuning quality with **up to 99.97 % fewer trainable parameters than LoRA** by working directly in the frequency domain of each weight matrix.

![ComfyUI-sDCTFT workflow screenshot](ScrShot%207.png)

---

## How it works

Standard fine-tuning updates every weight.
LoRA approximates updates with two low-rank matrices.
sDCTFT instead selects a tiny set of DCT spectral coefficients and only trains those:

```
W_eff = W_frozen + iDCT(ΔW_F) × α
```

1. **DCT projection** — Apply 2D DCT-II (orthonormal) to each weight matrix.
2. **Hierarchical selection** — Divide the frequency plane into three concentric bands
   (low / mid / high) using Euclidean distance from the DC origin.
   Within each band pick the top 70 % of coefficients by energy magnitude,
   and fill the remaining 30 % randomly.
3. **Train selected coefficients** — Only the chosen `n` spectral values receive
   gradients (Kaiming-initialized). Every other parameter is frozen.
4. **Reconstruct at runtime** — The sparse DCT update is mapped back to weight-space
   via iDCT and scaled by `α` before each forward pass.

| Method   | Trainable params (LLaMA 3.1 8B) | Relative size |
|----------|--------------------------------:|:-------------:|
| Full FT  | 8 000 M                         | 100 %         |
| LoRA r=8 | 183 M                           | 2.3 %         |
| **sDCTFT** | **0.05 M**                    | **0.0006 %**  |

---

## Nodes

| Node | Category | Description |
|------|----------|-------------|
| **Load Dataset (sDCTFT)** | `sDCTFT/Dataset` | Load training images from a folder or an IMAGE tensor; attach optional per-image captions |
| **Training Config (sDCTFT)** | `sDCTFT/Training` | Set all hyperparameters (n, δ, α, lr, epochs …) |
| **Fine-Tune Model (sDCTFT)** | `sDCTFT/Training` | Run the training loop; returns a standard ComfyUI MODEL |
| **Save Fine-Tuned Model (sDCTFT)** | `sDCTFT/IO` | Save the result as `.safetensors` or `.pt` |
| **Dataset Info (sDCTFT)** | `sDCTFT/Dataset` | Inspect a loaded dataset; returns a text summary and preview images |

---

## Installation

```bash
# 1. Clone into ComfyUI's custom_nodes folder
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-sDCTFT

# 2. Install dependencies
pip install -r ComfyUI-sDCTFT/requirements.txt

# 3. Restart ComfyUI
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, Pillow.
`safetensors` is optional but strongly recommended for saving.

---

## Recommended workflow

```
[Load Checkpoint] ── MODEL ─────────────────────────────────────┐
                  ── VAE  ──────────────────────────────────┐   │
                  ── CLIP ────────────────────────────────┐ │   │
                                                          ↓ ↓   ↓
[Load Dataset]  ── dataset ──────────> [Fine-Tune Model (sDCTFT)] ── MODEL
[Training Config] ── config ─────────>                              │
                                                                    ↓
                                                   [Save Fine-Tuned Model]
                                                                    │
                                                                    ↓
                                                         sdctft_finetuned.safetensors
```

The output MODEL flows directly into any KSampler node for immediate inference.

---

## Key hyperparameters

| Parameter | Widget | Default | Notes |
|-----------|--------|---------|-------|
| `n_coefficients` | INT | 700 | Trainable DCT coefficients per layer. More = higher capacity but slower. Try 700 → 2400 → 7000. |
| `delta` | FLOAT | 0.7 | Fraction of `n` chosen by energy ranking. Paper default. |
| `alpha` | FLOAT | 16.0 | Scales the spatial weight update. Raise to 300 for aggressive style transfer. |
| `learning_rate` | FLOAT | 1e-3 | AdamW LR. Lower to 1e-4 if loss is unstable. |
| `num_epochs` | INT | 5 | Full passes over the dataset. |
| `batch_size` | INT | 1 | Increase if VRAM permits. |
| `target_resolution` | INT | 512 | Pixel size images are resized to before VAE encoding. |
| `target_layers` | STRING | `to_q,to_k,to_v,…` | Comma-separated layer name suffixes to wrap. Use `all_linear` to target every linear layer. |

### Layer targeting

By default only the cross-attention and feed-forward projection layers are wrapped:

```
to_q, to_k, to_v, to_out.0
ff.net.0.proj, ff.net.2
proj_in, proj_out
```

These match the standard SD 1.x / 2.x / SDXL UNet naming.
For custom architectures inspect `model.model.diffusion_model.named_modules()` and adjust accordingly.

### Caption files

Place a `.txt` file with the same stem next to each image:

```
dataset/
  cat_01.png
  cat_01.txt   ← "a tabby cat sitting on a windowsill"
  cat_02.jpg
  cat_02.txt
```

If no `.txt` file is found the **shared_caption** widget value is used.
Leave both blank for unconditional fine-tuning.

---

## Algorithm details

### Frequency band partitioning (Eq. 3–7 of the paper)

```
d(u, v)  = √(u² + v²)
d_max    = √((M/2)² + (N/2)²)      # half-dimension norm

low   = { (u,v) : d(u,v) ≤ d_max/3 }
mid   = { (u,v) : d_max/3 < d(u,v) ≤ 2·d_max/3 }
high  = { (u,v) : d(u,v) > 2·d_max/3 }
```

`n` is split evenly across the three bands.
Within each band the hybrid selection is:

```
top ⌈n_band × δ⌉   →  highest |W_F[u,v]|²  (deterministic)
remaining           →  uniform random sample  (seed=42)
```

### Forward pass (Algorithm 1 of the paper)

```python
dW_F = zeros(M, N)
dW_F[selected_indices] = dW_params          # scatter trainable values
dW_T = iDCT_2d(dW_F) * alpha               # inverse DCT + scale
W_eff = W_frozen + dW_T                     # add to frozen base
```

All DCT / iDCT operations are implemented with `torch.fft` and are fully differentiable.

---

## Running the validation test

```bash
python test_dct.py
```

Expected output:
```
DCT round-trip tests
  roundtrip (16, 16)              PASS
  roundtrip (32, 64)              PASS
  ...
Scipy comparison tests
  vs scipy (8, 8)                 PASS
  vs scipy (16, 32)               PASS
Autograd test
  autograd                        PASS
sDCTFT layer end-to-end test
  sDCTFTParametrization E2E       PASS
    trainable: 100 / 4096 params (2.4%)
```

---

## Tips

- **Low VRAM:** Keep `n_coefficients` at 700, `batch_size` at 1, and `target_resolution` at 512. sDCTFT's memory footprint is minimal because almost all parameters are frozen.
- **Style transfer:** Raise `alpha` to 100–300 and use 10–20 epochs with a high-quality style image set.
- **Subject fine-tuning (DreamBooth-style):** Use 10–30 images, 5–10 epochs, `alpha=16`, and include a class-preservation prompt as the shared caption.
- **Unstable loss:** Lower `learning_rate` to `1e-4` and reduce `alpha`.
- **Intermediate checkpoints:** Set `save_every_n_steps` and `checkpoint_dir` in the Config node.

---

## Reference

```bibtex
@inproceedings{shen2025macp,
  title     = {MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection},
  author    = {Shen, Yixian and Bi, Qi and Huang, Jia-Hong and Zhu, Hongyi
               and Pimentel, Andy D. and Pathania, Anuj},
  booktitle = {Proceedings of ACL 2025},
  year      = {2025},
  note      = {Best Theme Paper Award},
  url       = {https://arxiv.org/abs/2410.09103}
}
```

---

## License

MIT
