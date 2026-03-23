#!/usr/bin/env python3
"""Smoke test: verify FuXi model forward pass, shapes, and gradient flow."""

import sys
import torch

from model import FuXi


def test_forward():
    """Test forward pass with typical data dimensions."""
    # Your data: 20 variables, 121 lat, 240 lon, 2 time-steps
    B, V, T, H, W = 2, 20, 2, 121, 240

    model = FuXi(
        num_variables=V,
        embed_dim=128,       # small for testing
        num_heads=4,
        window_size=5,
        depth_pre=2,
        depth_mid=4,
        depth_post=2,
        drop_path_rate=0.1,
        input_height=H,
        input_width=W,
    )

    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    x = torch.randn(B, V, T, H, W)
    print(f"Input shape:  {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (B, V, H, W), f"Expected {(B, V, H, W)}, got {tuple(y.shape)}"
    print("✓ Output shape correct")

    # Test gradient flow
    x.requires_grad_(False)
    model.train()
    y = model(torch.randn(B, V, T, H, W))
    loss = y.mean()
    loss.backward()
    print("✓ Gradient flow OK")

    # Test autoregressive forecast
    model.eval()
    with torch.no_grad():
        x_prev = torch.randn(1, V, H, W)
        x_curr = torch.randn(1, V, H, W)
        preds = model.forecast(x_prev, x_curr, num_steps=3)
    print(f"✓ 3-step forecast: {len(preds)} predictions, each {tuple(preds[0].shape)}")

    # Print architecture summary
    print(f"\nArchitecture summary:")
    print(f"  Cube embedding:  ({V}, {T}, {H}, {W}) → "
          f"({model.cube_embedding.embed_dim}, {model.embed_h}, {model.embed_w})")
    print(f"  Down resolution: ({model.embed_h}, {model.embed_w}) → "
          f"({model.embed_h // 2}, {model.embed_w // 2})")
    print(f"  Up resolution:   ({model.embed_h // 2}, {model.embed_w // 2}) → "
          f"({model.embed_h}, {model.embed_w})")
    print(f"  FC reconstruct:  ({model.embed_h}, {model.embed_w}) → "
          f"({model.recon_h}, {model.recon_w})")
    print(f"  Bilinear interp: ({model.recon_h}, {model.recon_w}) → ({H}, {W})")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_forward()
