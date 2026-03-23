"""
FuXi model – paper-accurate.

Architecture (paper §Methods, Fig. 5a)
--------------------------------------
1. Cube Embedding : (B, C_var, 2, H, W) → (B, embed_dim, H', W')
2. U-Transformer  : (B, embed_dim, H', W') → (B, embed_dim, H', W')
3. FC Layer        : per-position linear mapping  → (B, C_var, H_recon, W_recon)
4. Bilinear interp : restore to original (H, W)

The model is autoregressive: it takes (X_{t-1}, X_t) and predicts X_{t+1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import CubeEmbedding
    from .u_transformer import UTransformer
    from .swin_v2 import trunc_normal_
except ImportError:
    from blocks import CubeEmbedding
    from u_transformer import UTransformer
    from swin_v2 import trunc_normal_


class FuXi(nn.Module):
    """
    FuXi weather forecasting model.

    Parameters
    ----------
    num_variables : int
        Total input/output channels  (paper: 5 upper-air × 13 levels + 5 surface = 70).
    embed_dim : int
        Channel width C of the U-Transformer  (paper: 1536).
    num_heads : int
        Attention heads in Swin V2 blocks.
    window_size : int
        Swin local-window size.  Must divide spatial dims at both full
        and half resolution.
    depth_pre, depth_mid, depth_post : int
        Swin block counts  (paper total: 48).
    mlp_ratio : float
        MLP expansion ratio (paper: 4.0).
    drop_path_rate : float
        Max stochastic-depth rate (paper: 0.2).
    input_height, input_width : int
        Original spatial grid size (for FC head reconstruction).
    patch_size : tuple
        Cube-embedding kernel/stride  (paper: (2, 4, 4)).
    use_checkpoint : bool
        Gradient checkpointing to save memory (paper uses it).
    mc_dropout : float
        Monte-Carlo dropout rate for ensemble generation (paper: 0.2).
        Set to 0.0 for deterministic inference.
    """

    def __init__(
        self,
        num_variables: int = 20,
        embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 5,
        depth_pre: int = 2,
        depth_mid: int = 12,
        depth_post: int = 2,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        input_height: int = 121,
        input_width: int = 240,
        patch_size: tuple = (2, 4, 4),
        use_checkpoint: bool = False,
        mc_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size

        # ---- 1. Cube Embedding -------------------------------------------
        self.cube_embedding = CubeEmbedding(
            in_channels=num_variables,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # Compute embedded spatial dimensions (raw, before padding)
        _, ph, pw = patch_size
        self.embed_h = (input_height - ph) // ph + 1
        self.embed_w = (input_width - pw) // pw + 1

        # Pad embedded dims so they are divisible by window_size * 2.
        # Factor of 2 because DownBlock halves spatial dims, and the
        # half-resolution Swin blocks also need divisibility by window_size.
        factor = window_size * 2
        self.pad_h = (factor - self.embed_h % factor) % factor
        self.pad_w = (factor - self.embed_w % factor) % factor
        self.padded_h = self.embed_h + self.pad_h
        self.padded_w = self.embed_w + self.pad_w

        # ---- 2. U-Transformer -------------------------------------------
        self.u_transformer = UTransformer(
            channels=embed_dim,
            input_resolution=(self.padded_h, self.padded_w),
            num_heads=num_heads,
            window_size=window_size,
            depth_pre=depth_pre,
            depth_mid=depth_mid,
            depth_post=depth_post,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

        # ---- 3. FC output head -------------------------------------------
        # Each embedded spatial position maps to a (ph × pw) output patch.
        # FC: embed_dim → num_variables * ph * pw
        self.fc = nn.Linear(embed_dim, num_variables * ph * pw)

        # ---- 4. MC Dropout for ensemble (paper §Methods – FuXi ensemble) -
        self.mc_dropout = nn.Dropout(mc_dropout) if mc_dropout > 0 else None

        # ---- Reconstruction target size ----------------------------------
        self.recon_h = self.embed_h * ph
        self.recon_w = self.embed_w * pw

        # ---- Weight initialization ---------------------------------------
        self.apply(self._init_weights)
        self.u_transformer._init_respostnorm()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_variables, T=2, H, W)
               Two consecutive time-steps stacked along dim-2.
        Returns:
            (B, num_variables, H, W)  – forecast for next time-step.
        """
        B = x.shape[0]
        _, ph, pw = self.patch_size

        # 1. Cube embedding: (B, V, 2, H, W) → (B, C, H', W')
        x = self.cube_embedding(x)

        # Pad to make spatial dims divisible by window_size * 2
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

        # 2. U-Transformer: (B, C, padded_H, padded_W) → same
        x = self.u_transformer(x)

        # Remove padding
        if self.pad_h > 0 or self.pad_w > 0:
            x = x[:, :, : self.embed_h, : self.embed_w]

        # Optional MC dropout for ensemble generation
        if self.mc_dropout is not None:
            x = self.mc_dropout(x)

        # 3. FC head: per-position mapping C → V*ph*pw, then pixel-shuffle
        x = x.permute(0, 2, 3, 1)   # (B, H', W', C)
        x = self.fc(x)              # (B, H', W', V*ph*pw)

        # Reshape → pixel shuffle
        x = x.view(B, self.embed_h, self.embed_w,
                    self.num_variables, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.num_variables, self.recon_h, self.recon_w)

        # 4. Bilinear interpolation to original spatial size
        if (self.recon_h != self.input_height) or (self.recon_w != self.input_width):
            x = F.interpolate(
                x,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=True,
            )

        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forecast(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        num_steps: int = 1,
    ) -> list:
        """
        Autoregressive multi-step forecast.

        Args:
            x_prev : (B, V, H, W) – time step t-1
            x_curr : (B, V, H, W) – time step t
            num_steps : number of 6-hour steps to predict
        Returns:
            list of (B, V, H, W) tensors
        """
        predictions = []
        for _ in range(num_steps):
            inp = torch.stack([x_prev, x_curr], dim=2)  # (B, V, 2, H, W)
            pred = self(inp)
            predictions.append(pred)
            x_prev = x_curr
            x_curr = pred
        return predictions
