"""
Building blocks for the FuXi model – paper-accurate.

Components
----------
- CubeEmbedding : 3-D convolution + LayerNorm  (spatio-temporal → 2-D)
- ResidualBlock : two Conv→GN→SiLU units with a skip connection
- DownBlock     : 3×3 stride-2 conv followed by ResidualBlock
- UpBlock       : concat skip → 1×1 proj → ResidualBlock → TransposedConv

All blocks keep the channel dimension constant (paper: C = 1536).
Down/Up only change spatial resolution.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_groups(channels: int, desired: int = 32) -> int:
    """Largest divisor of *channels* that is ≤ *desired*."""
    g = min(desired, channels)
    while channels % g != 0:
        g -= 1
    return g


# ---------------------------------------------------------------------------
# Cube Embedding
# ---------------------------------------------------------------------------

class CubeEmbedding(nn.Module):
    """
    Space-time cube embedding (paper §Methods – Cube Embedding).

    A 3-D convolution with kernel = stride = (2, 4, 4) reduces the temporal
    dimension (2 time-steps → 1) and spatial dimensions by 4× each.
    Followed by LayerNorm.

    Input  : (B, C_var, T=2, H, W)        e.g. (B, 70, 2, 721, 1440)
    Output : (B, embed_dim, H', W')        e.g. (B, 1536, 180, 360)
    """

    def __init__(self, in_channels: int = 70, embed_dim: int = 1536,
                 patch_size=(2, 4, 4)):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Conv3d: treats variables as input channels, time as depth
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_var, T, H, W)
        Returns:
            (B, embed_dim, H', W')
        """
        # Conv3d → (B, embed_dim, 1, H', W') → squeeze time dim
        x = self.proj(x).squeeze(2)  # (B, embed_dim, H', W')

        # LayerNorm on channel dimension (permute → norm → permute back)
        x = x.permute(0, 2, 3, 1)   # (B, H', W', embed_dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # (B, embed_dim, H', W')
        return x

    def output_spatial(self, H: int, W: int):
        """Compute output (H', W') for given input spatial dims."""
        _, ph, pw = self.patch_size
        return (H - ph) // ph + 1, (W - pw) // pw + 1


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Paper §Methods – Down Block:
    "a residual block that has two 3×3 convolution layers followed by a
     group normalization (GN) layer and a sigmoid-weighted linear unit
     (SiLU) activation"

    Structure: Conv3×3 → GN → SiLU → Conv3×3 → GN → SiLU  + residual
    Channels remain constant.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        g = _safe_groups(channels, num_groups)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(g, channels)
        self.silu1 = nn.SiLU()

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(g, channels)
        self.silu2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.silu1(self.gn1(self.conv1(x)))
        x = self.silu2(self.gn2(self.conv2(x)))
        return x + residual


# ---------------------------------------------------------------------------
# Down Block
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Paper §Methods:
    "The Down Block consists of a 3×3 2D convolution layer with a stride
     of 2, and a residual block."

    Halves spatial resolution; channels stay constant.
    C × H × W  →  C × H/2 × W/2
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.downsample = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1,
        )
        self.residual = ResidualBlock(channels, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.residual(x)
        return x


# ---------------------------------------------------------------------------
# Up Block
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    """
    Paper §Methods:
    "The Up Block has the same residual block as used in the Down Block,
     along with a 2D transposed convolution with a kernel of 2 and a
     stride of 2."

    "a skip connection is included that concatenates the outputs from the
     Down Block with those of the transformer blocks before being fed
     into the Up Block."

    Skip concat (2C) → 1×1 proj (C) → ResidualBlock → TransposedConv(k=2,s=2)
    C × H/2 × W/2  →  C × H × W
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        # Fuse concatenated skip (2C → C)
        self.skip_proj = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.residual = ResidualBlock(channels, num_groups)
        self.upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=2, stride=2,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H/2, W/2) – output from middle transformer blocks
            skip : (B, C, H/2, W/2) – output from DownBlock (saved earlier)
        Returns:
            (B, C, H, W)
        """
        x = torch.cat([x, skip], dim=1)  # (B, 2C, H/2, W/2)
        x = self.skip_proj(x)            # (B, C,  H/2, W/2)
        x = self.residual(x)             # (B, C,  H/2, W/2)
        x = self.upsample(x)             # (B, C,  H,   W)
        return x
