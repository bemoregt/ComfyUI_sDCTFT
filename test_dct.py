"""
Quick validation: checks that dct2d / idct2d match scipy and are invertible.
Run from repo root:  python test_dct.py
"""
import sys, math, time
import numpy as np
import torch

sys.path.insert(0, ".")
from sdctft.dct import dct2d, idct2d


def test_roundtrip():
    for shape in [(16, 16), (32, 64), (7, 13), (1, 512, 512)]:
        x = torch.randn(*shape)
        X = dct2d(x, norm="ortho")
        x_rec = idct2d(X, norm="ortho")
        err = (x - x_rec).abs().max().item()
        status = "PASS" if err < 1e-4 else f"FAIL (err={err:.2e})"
        print(f"  roundtrip {str(shape):20s}  {status}")


def test_vs_scipy():
    try:
        from scipy.fft import dctn, idctn
    except ImportError:
        print("  scipy not available, skipping scipy comparison")
        return

    for shape in [(8, 8), (16, 32)]:
        x_np = np.random.randn(*shape).astype(np.float32)
        x_t = torch.from_numpy(x_np)

        # scipy reference
        ref = dctn(x_np, type=2, norm="ortho")

        # our implementation
        out = dct2d(x_t, norm="ortho").numpy()

        err = np.abs(ref - out).max()
        status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
        print(f"  vs scipy {str(shape):20s}  {status}")


def test_autograd():
    x = torch.randn(8, 16, requires_grad=True)
    X = dct2d(x, norm="ortho")
    loss = X.sum()
    loss.backward()
    status = "PASS" if x.grad is not None else "FAIL (no grad)"
    print(f"  autograd                        {status}")


def test_sdctft_layer():
    sys.path.insert(0, ".")
    from sdctft.algorithm import sDCTFTParametrization
    import torch.nn as nn
    import torch.nn.utils.parametrize as P

    layer = nn.Linear(64, 64, bias=False)
    param = sDCTFTParametrization(layer.weight, n=100, delta=0.7, alpha=16.0)
    P.register_parametrization(layer, "weight", param)

    # Forward pass
    x = torch.randn(4, 64)
    y = layer(x)

    # Backward pass
    loss = y.sum()
    loss.backward()

    has_grad = param.dW_params.grad is not None
    status = "PASS" if has_grad else "FAIL (no grad on dW_params)"
    print(f"  sDCTFTParametrization E2E       {status}")

    n_trainable = sum(p.numel() for p in [param.dW_params])
    n_total = layer.weight.numel()
    print(f"    trainable: {n_trainable} / {n_total} params ({100*n_trainable/n_total:.1f}%)")


if __name__ == "__main__":
    print("=" * 50)
    print("DCT round-trip tests")
    print("=" * 50)
    test_roundtrip()

    print("\nScipy comparison tests")
    print("=" * 50)
    test_vs_scipy()

    print("\nAutograd test")
    print("=" * 50)
    test_autograd()

    print("\nsDCTFT layer end-to-end test")
    print("=" * 50)
    test_sdctft_layer()

    print("\nAll tests complete.")
