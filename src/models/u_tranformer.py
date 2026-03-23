"""
U-Transformer for the FuXi model – paper-accurate.

Paper §Methods:
"The U-Transformer is constructed using 48 repeated Swin Transformer V2
 blocks" with a single DownBlock and UpBlock for temporary spatial
 reduction.

Layout
------
    [pre Swin blocks at full res]
        → DownBlock  (halve spatial dims)
    [mid Swin blocks at half res]   ← most blocks live here (cheaper)
        → UpBlock    (restore dims, with skip from DownBlock output)
    [post Swin blocks at full res]

All blocks operate at constant channel dimension C.
"""

import torch
import torch.nn as nn

try:
    from .blocks import DownBlock, UpBlock
    from .swin import SwinStage
except ImportError:
    from blocks import DownBlock, UpBlock
    from swin import SwinStage


class UTransformer(nn.Module):
    """
    U-Transformer backbone.

    Parameters
    ----------
    channels : int
        Channel dimension C (paper: 1536).
    input_resolution : (int, int)
        (H, W) after cube embedding.  Paper: (180, 360).
    num_heads : int
        Attention heads.  Paper uses C / head_dim.
    window_size : int
        Swin local-window size.  Must divide both full and half resolutions.
    depth_pre, depth_mid, depth_post : int
        Block counts for each section.  Paper total = 48.
    mlp_ratio : float
        MLP expansion ratio.  Paper: 4.0.
    drop_path_rate : float
        Maximum stochastic-depth rate (linearly ramped).  Paper: 0.2.
    use_checkpoint : bool
        Gradient checkpointing (paper uses it to fit on A100s).
    """

    def __init__(
        self,
        channels: int = 1536,
        input_resolution: tuple = (180, 360),
        num_heads: int = 8,
        window_size: int = 5,
        depth_pre: int = 4,
        depth_mid: int = 40,
        depth_post: int = 4,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        total_depth = depth_pre + depth_mid + depth_post
        H, W = input_resolution
        half_res = (H // 2, W // 2)

        # Linearly-ramped drop-path rates across ALL blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        idx = 0

        # ---- Pre-down Swin blocks (full resolution) ----------------------
        self.pre_stage = SwinStage(
            dim=channels,
            input_resolution=input_resolution,
            depth=depth_pre,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=dpr[idx: idx + depth_pre],
            use_checkpoint=use_checkpoint,
        )
        idx += depth_pre

        # ---- Down Block (C×H×W → C×H/2×W/2) -----------------------------
        self.down_block = DownBlock(channels)

        # ---- Middle Swin blocks (half resolution – most blocks here) -----
        self.mid_stage = SwinStage(
            dim=channels,
            input_resolution=half_res,
            depth=depth_mid,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=dpr[idx: idx + depth_mid],
            use_checkpoint=use_checkpoint,
        )
        idx += depth_mid

        # ---- Up Block (C×H/2×W/2 → C×H×W, with skip connection) --------
        self.up_block = UpBlock(channels)

        # ---- Post-up Swin blocks (full resolution) -----------------------
        self.post_stage = SwinStage(
            dim=channels,
            input_resolution=input_resolution,
            depth=depth_post,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=dpr[idx: idx + depth_post],
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # Pre-down transformer blocks
        x = self.pre_stage(x)                 # (B, C, H, W)

        # Downsample + save skip
        skip = self.down_block(x)             # (B, C, H/2, W/2)

        # Middle transformer blocks at reduced resolution
        x = self.mid_stage(skip)              # (B, C, H/2, W/2)

        # Upsample with skip connection (concat DownBlock output + mid output)
        x = self.up_block(x, skip)            # (B, C, H, W)

        # Post-up transformer blocks
        x = self.post_stage(x)                # (B, C, H, W)
        return x

    def _init_respostnorm(self):
        """Zero-init SwinV2 post-norm weights for stable training."""
        self.pre_stage._init_respostnorm()
        self.mid_stage._init_respostnorm()
        self.post_stage._init_respostnorm()
