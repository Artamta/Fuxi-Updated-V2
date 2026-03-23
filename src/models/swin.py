"""
Swin Transformer V2 components for FuXi U-Transformer.

Extracted and adapted from the official Microsoft Swin-Transformer repo:
https://github.com/microsoft/Swin-Transformer
Licensed under The MIT License.

Key SwinV2 features (vs V1):
  - Residual post-normalization (LN at END of residual unit)
  - Scaled cosine attention (not dot-product)
  - Log-spaced continuous relative position bias (CPB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_2tuple(x):
    """Convert scalar to 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=0.02, a=-2.0, b=2.0):
    """Truncated normal initialization (from timm)."""
    with torch.no_grad():
        l = (1.0 + torch.erf(torch.tensor((a - mean) / std / np.sqrt(2)))) / 2.0
        u = (1.0 + torch.erf(torch.tensor((b - mean) / std / np.sqrt(2)))) / 2.0
        tensor.uniform_(2 * l.item() - 1, 2 * u.item() - 1)
        tensor.erfinv_()
        tensor.mul_(std * np.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    """Stochastic Depth / Drop Path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP with GELU activation (used inside Swin blocks)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Window operations
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size (int)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Merge windows back into a feature map.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int)
        H, W: spatial dimensions
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Scaled Cosine Attention (SwinV2)
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with Swin Transformer V2 features:
      - Scaled cosine attention: Sim(q,k) = cos(q,k) / tau + B
      - Learnable per-head temperature tau (clamped)
      - Continuous relative position bias via MLP on log-spaced coordinates
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, pretrained_window_size=(0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Learnable temperature (per head, not shared across layers)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # MLP for continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Log-spaced relative coordinates table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to [-8, 8]
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, N, N) or None
        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # ---- Scaled cosine attention (SwinV2 Eq. 1) ----
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=np.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale

        # ---- Continuous relative position bias ----
        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table
        ).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Swin Transformer V2 Block
# ---------------------------------------------------------------------------

class SwinTransformerV2Block(nn.Module):
    """
    A single Swin Transformer V2 block.

    Key V2 changes vs V1:
      - Residual post-normalization (LN at END of residual, not beginning)
      - Scaled cosine attention
      - Log-spaced continuous position bias

    Paper: "The U-Transformer is constructed using 48 repeated
    Swin Transformer V2 blocks"
    """

    def __init__(
        self, dim, input_resolution, num_heads, window_size=5,
        shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
        attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        # Post-normalization (SwinV2: LN at end of residual)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )

        # Pre-compute attention mask for shifted-window attention
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        Args:
            x: (B, L, C) where L = H * W
        Returns:
            (B, L, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input size {L} != {H}*{W}"

        shortcut = x
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Window partition → attention → window reverse
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual post-norm (SwinV2)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def _init_respostnorm(self):
        """Zero-init post-norm weights for stable training (SwinV2)."""
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm1.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)
        nn.init.constant_(self.norm2.weight, 0)


# ---------------------------------------------------------------------------
# SwinStage: flat sequence of SwinV2 blocks (no PatchMerging)
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """
    A flat sequence of Swin Transformer V2 blocks at a FIXED resolution.

    Unlike the hierarchical Swin (which uses PatchMerging to halve resolution
    between stages), this stage keeps channels and resolution constant.
    This is what the FuXi paper uses inside the U-Transformer.

    Handles format conversion: (B, C, H, W) ↔ (B, H*W, C).
    """

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size=5,
        mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
        drop_path=0.0, norm_layer=nn.LayerNorm,
        use_checkpoint=False, pretrained_window_size=0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # Linearly-ramped drop-path rates for this stage
        if isinstance(drop_path, (list, tuple)):
            dpr = list(drop_path)
        else:
            dpr = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)  – conv format
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Convert to transformer format: (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        # Convert back to conv format: (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def _init_respostnorm(self):
        """Zero-init post-norm weights in all blocks."""
        for blk in self.blocks:
            blk._init_respostnorm()
