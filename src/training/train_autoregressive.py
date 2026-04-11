#!/usr/bin/env python3
"""
Autoregressive FuXi training.

Key features
- Multi-step forecast training with free-running rollout (no teacher forcing by default)
- Optional teacher forcing probability to stabilize early epochs
- Mixed precision (fp16) and gradient clipping
- Per-step and averaged MAE tracking (first step + last step)
- Checkpointing (last + best) with resume support
- Config-driven overrides via YAML/JSON (same shape as `train.py`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    PEFT_AVAILABLE = False

try:
    from ..models.fuxi_model import make_fuxi
    from .loss import LatitudeWeightedL1Loss
    from ..pretraining.pretrain import (
        FuXiZarrDataset,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        DEFAULT_PRESSURE_LEVELS,
    )
except ImportError:
    from src.models.fuxi_model import make_fuxi
    from src.training.loss import LatitudeWeightedL1Loss
    from src.pretraining.pretrain import (
        FuXiZarrDataset,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        DEFAULT_PRESSURE_LEVELS,
    )

# Default WeatherBench2 Zarr (paper path)
ZARR_STORE = "/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
DEFAULT_LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]


def parse_csv_list(value: Optional[str]):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_cli_overrides(argv: List[str]) -> set:
    """Collect explicit --flag tokens provided on the command line."""
    overrides = set()
    for token in argv:
        if token == "--":
            break
        if token.startswith("--") and len(token) > 2:
            if "=" in token:
                overrides.add(token.split("=", 1)[0])
            else:
                overrides.add(token)
    return overrides


def load_config(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs. Install pyyaml.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    if not cfg:
        return args

    cli_overrides = getattr(args, "_cli_overrides", set())

    def set_if(section: str, key: str, attr: str = None):
        attr = attr or key
        cli_flag = f"--{attr.replace('_', '-')}"
        if cli_flag in cli_overrides:
            return
        if section in cfg and key in cfg[section]:
            setattr(args, attr, cfg[section][key])

    for key in [
        "zarr_store", "train_start", "train_end", "val_start", "val_end",
        "test_start", "test_end", "history_steps", "pressure_vars",
        "surface_vars", "pressure_levels",
    ]:
        set_if("data", key)

    for key in [
        "preset", "embed_dim", "window_size", "drop_path_rate", "num_heads",
        "depth_pre", "depth_mid", "depth_post", "mlp_ratio", "mc_dropout",
        "use_checkpoint",
    ]:
        set_if("model", key)

    for key in [
        "max_epochs", "max_iters", "batch_size", "num_workers", "patience",
        "lr", "weight_decay", "beta1", "beta2", "fp16", "gpus",
        "forecast_steps", "teacher_forcing", "grad_clip", "eval_every",
        "input_noise_std", "noise_coarse_factor", "target_offset_steps",
        "curriculum_start_steps", "curriculum_max_steps", "curriculum_step_size",
        "curriculum_epoch_interval", "constant_lr", "prefetch_factor",
        "status_every_steps",
        "skip_final_test_eval",
    ]:
        set_if("train", key)

    for key in ["output_root", "exp_name", "resume", "seed"]:
        set_if("output", key)

    for key in [
        "enable_lora", "lora_rank", "lora_alpha", "lora_dropout",
        "lora_target_modules", "lora_bias", "lora_train_base",
    ]:
        set_if("finetuning", key)

    set_if("logging", "tensorboard")
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FuXi autoregressive training")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML/JSON config file; CLI overrides config")

    # Data
    p.add_argument("--zarr-store", type=str, default=ZARR_STORE)
    p.add_argument("--train-start", type=str, default="1979-01-01")
    p.add_argument("--train-end", type=str, default="2015-12-31")
    p.add_argument("--val-start", type=str, default="2016-01-01")
    p.add_argument("--val-end", type=str, default="2017-12-31")
    p.add_argument("--test-start", type=str, default="2018-01-01")
    p.add_argument("--test-end", type=str, default="2018-12-31")
    p.add_argument("--history-steps", type=int, default=2)

    # Training
    p.add_argument("--forecast-steps", type=int, default=20,
                   help="Number of 6h steps to roll out during training")
    p.add_argument(
        "--target-offset-steps",
        type=int,
        default=0,
        help="Offset in 6h steps before history starts (0=0-5d stage, 20=5-10d, 40=10-15d)",
    )
    p.add_argument("--teacher-forcing", type=float, default=0.0,
                   help="Probability of using ground truth as next input during training")
    p.add_argument(
        "--curriculum-start-steps",
        type=int,
        default=0,
        help="Curriculum start horizon; 0 disables curriculum",
    )
    p.add_argument(
        "--curriculum-max-steps",
        type=int,
        default=0,
        help="Curriculum max horizon; used with --curriculum-start-steps",
    )
    p.add_argument(
        "--curriculum-step-size",
        type=int,
        default=2,
        help="Increase horizon by this many steps per curriculum interval",
    )
    p.add_argument(
        "--curriculum-epoch-interval",
        type=int,
        default=1,
        help="Number of epochs between curriculum horizon increases",
    )
    p.add_argument(
        "--input-noise-std",
        type=float,
        default=0.0,
        help="Std-dev of smooth Perlin-like noise added to history inputs during training",
    )
    p.add_argument(
        "--noise-coarse-factor",
        type=int,
        default=16,
        help="Coarse downsample factor for Perlin-like noise texture (>=2 for smooth noise)",
    )
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor per worker (used when num_workers > 0)",
    )
    p.add_argument(
        "--status-every-steps",
        type=int,
        default=200,
        help="Print in-epoch status every N optimizer steps (0 disables)",
    )
    p.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers across epochs",
    )
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--constant-lr", action="store_true",
                   help="Disable cosine scheduler and keep LR constant")

    # Model
    p.add_argument("--preset", choices=["paper", "mini"], default="paper")
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--drop-path-rate", type=float, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--depth-pre", type=int, default=None)
    p.add_argument("--depth-mid", type=int, default=None)
    p.add_argument("--depth-post", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=None)
    p.add_argument("--mc-dropout", type=float, default=None)
    p.add_argument("--use-checkpoint", action="store_true")

    # I/O
    p.add_argument("--output-root", type=str, default="results")
    p.add_argument("--exp-name", type=str, default="autoregressive")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--resume-model-only",
        action="store_true",
        help="When resuming, load only model weights and reset epoch/step/optimizer state",
    )
    p.add_argument(
        "--skip-final-test-eval",
        action="store_true",
        help="Skip final full test-set evaluation to speed stage chaining",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gpus", type=int, default=None)

    # Parameter-efficient fine-tuning (LoRA)
    p.add_argument("--enable-lora", action="store_true",
                   help="Enable LoRA adapters for parameter-efficient fine-tuning")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank (r)")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA scaling alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="LoRA dropout")
    p.add_argument("--lora-target-modules", type=str, default=",".join(DEFAULT_LORA_TARGET_MODULES),
                   help="Comma-separated target module suffixes for LoRA")
    p.add_argument("--lora-bias", choices=["none", "lora_only", "all"], default="none",
                   help="Bias handling in LoRA")
    p.add_argument("--lora-train-base", action="store_true",
                   help="Train base model parameters in addition to LoRA adapters")

    # Variables / levels
    p.add_argument("--pressure-vars", type=str, default=None,
                   help="Comma-separated pressure variable names")
    p.add_argument("--surface-vars", type=str, default=None,
                   help="Comma-separated surface variable names")
    p.add_argument("--pressure-levels", type=str, default=None,
                   help="Comma-separated pressure levels (ints)")

    args = p.parse_args()
    args._cli_overrides = parse_cli_overrides(sys.argv[1:])
    return args


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_epoch_metrics_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "epoch", "horizon", "global_step",
        "train_loss", "train_mae", "train_first_mae", "train_last_mae",
        "val_loss", "val_mae", "val_first_mae", "val_last_mae",
        "it_per_sec", "samples_per_sec", "eta_hours",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_training_curves(rows: List[Dict[str, float]], out_path: str) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    train_mae = [float(r["train_mae"]) for r in rows]
    val_loss_points = [(int(r["epoch"]), float(r["val_loss"])) for r in rows if np.isfinite(float(r["val_loss"]))]
    val_mae_points = [(int(r["epoch"]), float(r["val_mae"])) for r in rows if np.isfinite(float(r["val_mae"]))]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, train_loss, label="Train loss", linewidth=2.0)
    if val_loss_points:
        axes[0].plot(
            [x for x, _ in val_loss_points],
            [y for _, y in val_loss_points],
            label="Val loss",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )
    axes[0].set_ylabel("Lat-weighted L1")
    axes[0].set_title("Loss Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_mae, label="Train MAE", linewidth=2.0)
    if val_mae_points:
        axes[1].plot(
            [x for x, _ in val_mae_points],
            [y for _, y in val_mae_points],
            label="Val MAE",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE Curves")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_device(enable_fp16: bool) -> Tuple[torch.device, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = enable_fp16 and device.type == "cuda"
    return device, use_fp16


class AutoregressiveFuXiDataset(FuXiZarrDataset):
    """Extends the one-step dataset to return multi-step targets."""

    def __init__(self, *args, forecast_steps: int = 20, target_offset_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be >= 1")
        if target_offset_steps < 0:
            raise ValueError("target_offset_steps must be >= 0")
        self.forecast_steps = int(forecast_steps)
        self.target_offset_steps = int(target_offset_steps)

    def __len__(self) -> int:
        return self.n_times - (self.target_offset_steps + self.history_steps + self.forecast_steps - 1)

    def _load_sequence_raw(self, start_in_slice: int, num_frames: int) -> np.ndarray:
        """Load contiguous time-window frames in one shot to reduce Zarr I/O overhead."""
        t_indices = np.asarray(self._time_indices[start_in_slice:start_in_slice + num_frames], dtype=np.int64)
        if t_indices.shape[0] != num_frames:
            raise IndexError("Requested sequence window exceeds available time range")

        t0 = int(t_indices[0])
        t1 = int(t_indices[-1]) + 1
        contiguous = (t1 - t0) == num_frames

        parts = []
        for arr in self._pressure_arrays:
            if contiguous:
                block = np.asarray(arr[t0:t1, self._level_indices, :, :], dtype=np.float32)
            else:
                block = np.stack(
                    [
                        np.asarray(arr[int(ti), self._level_indices, :, :], dtype=np.float32)
                        for ti in t_indices
                    ],
                    axis=0,
                )
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block)

        for arr in self._surface_arrays:
            if contiguous:
                block = np.asarray(arr[t0:t1, :, :], dtype=np.float32)
            else:
                block = np.stack(
                    [np.asarray(arr[int(ti), :, :], dtype=np.float32) for ti in t_indices],
                    axis=0,
                )
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block[:, np.newaxis, :, :])

        return np.concatenate(parts, axis=1)

    def __getitem__(self, index: int):
        base = index + self.target_offset_steps

        seq = self._load_sequence_raw(base, self.history_steps + self.forecast_steps)  # (T, C, H, W)
        seq = torch.from_numpy(seq)
        seq = (seq - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)

        history = seq[: self.history_steps].permute(1, 0, 2, 3).contiguous()  # (C, history, H, W)
        target_seq = seq[self.history_steps: self.history_steps + self.forecast_steps]  # (forecast, C, H, W)
        return history, target_seq


def build_loaders(args):
    dataset_forecast_steps = int(args.forecast_steps)
    if args.curriculum_max_steps and int(args.curriculum_max_steps) > 0:
        # Only load targets up to the maximum horizon that can actually be used.
        dataset_forecast_steps = max(1, min(dataset_forecast_steps, int(args.curriculum_max_steps)))

    train_set = AutoregressiveFuXiDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.train_start,
        time_end=args.train_end,
        forecast_steps=dataset_forecast_steps,
        target_offset_steps=args.target_offset_steps,
        pressure_vars=args.pressure_vars,
        surface_vars=args.surface_vars,
        pressure_levels=args.pressure_levels,
    )

    val_set = AutoregressiveFuXiDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.val_start,
        time_end=args.val_end,
        forecast_steps=dataset_forecast_steps,
        target_offset_steps=args.target_offset_steps,
        mean=train_set.mean,
        std=train_set.std,
        pressure_vars=args.pressure_vars,
        surface_vars=args.surface_vars,
        pressure_levels=args.pressure_levels,
    )

    test_set = AutoregressiveFuXiDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.test_start,
        time_end=args.test_end,
        forecast_steps=dataset_forecast_steps,
        target_offset_steps=args.target_offset_steps,
        mean=train_set.mean,
        std=train_set.std,
        pressure_vars=args.pressure_vars,
        surface_vars=args.surface_vars,
        pressure_levels=args.pressure_levels,
    )

    persistent_workers = (args.num_workers > 0) and (not args.no_persistent_workers)
    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        loader_common["prefetch_factor"] = max(2, int(args.prefetch_factor))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common,
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def build_model(num_vars: int, spatial_h: int, spatial_w: int, args, device: torch.device):
    overrides = {}
    if args.embed_dim is not None:
        overrides["embed_dim"] = args.embed_dim
    if args.window_size is not None:
        overrides["window_size"] = args.window_size
    if args.drop_path_rate is not None:
        overrides["drop_path_rate"] = args.drop_path_rate
    if args.num_heads is not None:
        overrides["num_heads"] = args.num_heads
    if args.depth_pre is not None:
        overrides["depth_pre"] = args.depth_pre
    if args.depth_mid is not None:
        overrides["depth_mid"] = args.depth_mid
    if args.depth_post is not None:
        overrides["depth_post"] = args.depth_post
    if args.mlp_ratio is not None:
        overrides["mlp_ratio"] = args.mlp_ratio
    if args.mc_dropout is not None:
        overrides["mc_dropout"] = args.mc_dropout

    model = make_fuxi(
        preset=args.preset,
        num_variables=num_vars,
        input_height=spatial_h,
        input_width=spatial_w,
        use_checkpoint=args.use_checkpoint,
        **overrides,
    ).to(device)

    if args.enable_lora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA was requested (--enable-lora) but PEFT is not installed. "
                "Install dependencies from requirements.txt or run: pip install peft"
            )
        if args.lora_rank <= 0:
            raise ValueError("lora_rank must be > 0 when LoRA is enabled")

        target_modules = args.lora_target_modules or list(DEFAULT_LORA_TARGET_MODULES)
        lora_cfg = LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=float(args.lora_alpha),
            target_modules=target_modules,
            lora_dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
        )
        model = get_peft_model(model, lora_cfg)

        if args.lora_train_base:
            for p in model.parameters():
                p.requires_grad = True

        mode = "LoRA + full base" if args.lora_train_base else "LoRA adapter-only"
        print(
            "LoRA enabled | "
            f"mode={mode} | rank={args.lora_rank} | alpha={args.lora_alpha} "
            f"| dropout={args.lora_dropout} | targets={target_modules}"
        )
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    num_gpus = args.gpus or torch.cuda.device_count()
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    raw_model = model.module if hasattr(model, "module") else model
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Parameters: trainable={trainable_params:,} / total={total_params:,} ({100.0 * trainable_params / max(total_params, 1):.2f}%)")
    return model, raw_model


def rollout_autoregressive(model, history, forecast_steps: int, teacher_forcing: float = 0.0, target_seq=None):
    preds = []
    prev = history[:, :, -2, :, :]
    curr = history[:, :, -1, :, :]

    for step in range(forecast_steps):
        inp = torch.stack([prev, curr], dim=2)
        pred = model(inp)
        preds.append(pred)

        if teacher_forcing > 0.0 and target_seq is not None:
            use_teacher = bool(torch.rand(()) < teacher_forcing)
            next_curr = target_seq[:, step].detach() if use_teacher else pred.detach()
        else:
            next_curr = pred.detach()

        prev, curr = curr, next_curr

    return torch.stack(preds, dim=1)


def build_perlin_like_noise(shape: torch.Size, device: torch.device, coarse_factor: int):
    """
    Build smooth, spatially correlated noise by upsampling a coarse random field.
    This is not exact Perlin noise but provides similar low-frequency perturbations.
    """
    if coarse_factor < 2:
        return torch.randn(shape, device=device)

    *prefix, h, w = shape
    coarse_h = max(2, h // coarse_factor)
    coarse_w = max(2, w // coarse_factor)

    coarse = torch.randn((*prefix, coarse_h, coarse_w), device=device, dtype=torch.float32)
    up = F.interpolate(
        coarse.view(-1, 1, coarse_h, coarse_w),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).view(*prefix, h, w)

    up = up - up.mean(dim=(-2, -1), keepdim=True)
    up = up / up.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return up


def maybe_add_history_noise(history: torch.Tensor, input_noise_std: float, noise_coarse_factor: int):
    if input_noise_std <= 0.0:
        return history
    noise = build_perlin_like_noise(history.shape, history.device, noise_coarse_factor).to(history.dtype)
    return history + (input_noise_std * noise)


def resolve_active_forecast_steps(args, epoch: int) -> int:
    start = int(args.curriculum_start_steps or 0)
    max_steps = int(args.curriculum_max_steps or 0)

    if start <= 0 or max_steps <= 0:
        return int(args.forecast_steps)

    interval = max(1, int(args.curriculum_epoch_interval))
    step_size = max(1, int(args.curriculum_step_size))
    active = start + ((max(1, epoch) - 1) // interval) * step_size
    active = min(active, max_steps)
    active = min(active, int(args.forecast_steps))
    return max(1, active)


def compute_sequence_metrics(preds, targets, criterion):
    step_losses = []
    step_mae = []
    for step in range(preds.shape[1]):
        l = criterion(preds[:, step], targets[:, step])
        step_losses.append(l)
        step_mae.append(torch.abs(preds[:, step].float() - targets[:, step].float()).mean())
    loss = torch.stack(step_losses).mean()
    mae = torch.stack(step_mae).mean()
    first_mae = step_mae[0]
    last_mae = step_mae[-1]
    return loss, mae, first_mae, last_mae


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_fp16,
                    forecast_steps, teacher_forcing, grad_clip, max_iters, global_step,
                    input_noise_std, noise_coarse_factor, status_every_steps):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_first = 0.0
    total_last = 0.0
    count = 0
    updates = 0
    epoch_t0 = time.perf_counter()

    for history, targets in loader:
        history = history.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets[:, :forecast_steps]
        history = maybe_add_history_noise(history, input_noise_std, noise_coarse_factor)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            preds = rollout_autoregressive(
                model,
                history,
                forecast_steps,
                teacher_forcing=teacher_forcing,
                target_seq=targets,
            )
            loss, mae, first_mae, last_mae = compute_sequence_metrics(preds, targets, criterion)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = history.shape[0]
        total_loss += loss.item() * bs
        total_mae += mae.item() * bs
        total_first += first_mae.item() * bs
        total_last += last_mae.item() * bs
        count += bs
        updates += 1
        global_step += 1

        if status_every_steps > 0 and (updates % status_every_steps == 0):
            elapsed = max(time.perf_counter() - epoch_t0, 1e-6)
            it_per_sec = updates / elapsed
            print(
                f"  [train] global_step={global_step} epoch_updates={updates} "
                f"it/s={it_per_sec:.2f} loss={loss.item():.4f} mae={mae.item():.4f}",
                flush=True,
            )

        if max_iters is not None and global_step >= max_iters:
            break

    denom = max(count, 1)
    return (
        total_loss / denom,
        total_mae / denom,
        total_first / denom,
        total_last / denom,
        global_step,
        updates,
    )


def evaluate(model, loader, criterion, device, use_fp16, forecast_steps):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_first = 0.0
    total_last = 0.0
    count = 0

    with torch.no_grad():
        for history, targets in loader:
            history = history.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets[:, :forecast_steps]
            with torch.cuda.amp.autocast(enabled=use_fp16):
                preds = rollout_autoregressive(model, history, forecast_steps, teacher_forcing=0.0)
                loss, mae, first_mae, last_mae = compute_sequence_metrics(preds, targets, criterion)

            bs = history.shape[0]
            total_loss += loss.item() * bs
            total_mae += mae.item() * bs
            total_first += first_mae.item() * bs
            total_last += last_mae.item() * bs
            count += bs

    denom = max(count, 1)
    return (
        total_loss / denom,
        total_mae / denom,
        total_first / denom,
        total_last / denom,
    )


def load_checkpoint_if_any(
    raw_model,
    optimizer,
    device,
    resume_path: Optional[str],
    resume_model_only: bool = False,
    expect_lora: bool = False,
):
    if not resume_path:
        return None
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state", ckpt)

    loaded = False
    if expect_lora and hasattr(raw_model, "get_base_model"):
        base_model = raw_model.get_base_model()

        if "base_model_state" in ckpt:
            base_model.load_state_dict(ckpt["base_model_state"], strict=False)
            print("Loaded base model state from LoRA checkpoint.")
            loaded = True

        if isinstance(model_state, dict):
            has_peft_keys = any(k.startswith("base_model.") or "lora_" in k for k in model_state)
            if has_peft_keys:
                missing, unexpected = raw_model.load_state_dict(model_state, strict=False)
                if missing:
                    print(f"Resume warning: missing {len(missing)} keys while loading LoRA model state.")
                if unexpected:
                    print(f"Resume warning: unexpected {len(unexpected)} keys while loading LoRA model state.")
                loaded = True
            else:
                base_missing, base_unexpected = base_model.load_state_dict(model_state, strict=False)
                if base_missing:
                    print(f"Resume warning: missing {len(base_missing)} keys while loading base model state.")
                if base_unexpected:
                    print(f"Resume warning: unexpected {len(base_unexpected)} keys while loading base model state.")
                loaded = True

    if not loaded:
        strict = not expect_lora
        missing, unexpected = raw_model.load_state_dict(model_state, strict=strict)
        if missing:
            print(f"Resume warning: missing {len(missing)} keys while loading model state.")
        if unexpected:
            print(f"Resume warning: unexpected {len(unexpected)} keys while loading model state.")

    if (not resume_model_only) and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as exc:
            print(f"Resume warning: failed to load optimizer state ({exc}). Continuing with fresh optimizer state.")
    return ckpt


def main():
    args = parse_args()
    cfg = load_config(args.config)
    args = apply_config(args, cfg)
    if hasattr(args, "_cli_overrides"):
        delattr(args, "_cli_overrides")

    args.pressure_vars = args.pressure_vars if isinstance(args.pressure_vars, list) else parse_csv_list(args.pressure_vars)
    args.surface_vars = args.surface_vars if isinstance(args.surface_vars, list) else parse_csv_list(args.surface_vars)

    if args.pressure_vars is None:
        args.pressure_vars = list(DEFAULT_PRESSURE_VARS)
    if args.surface_vars is None:
        args.surface_vars = list(DEFAULT_SURFACE_VARS)

    if isinstance(args.pressure_levels, list):
        args.pressure_levels = [int(v) for v in args.pressure_levels]
    else:
        pl = parse_csv_list(args.pressure_levels)
        args.pressure_levels = [int(v) for v in pl] if pl else list(DEFAULT_PRESSURE_LEVELS)

    if isinstance(args.lora_target_modules, list):
        args.lora_target_modules = [str(v).strip() for v in args.lora_target_modules if str(v).strip()]
    else:
        parsed = parse_csv_list(args.lora_target_modules)
        args.lora_target_modules = parsed if parsed else list(DEFAULT_LORA_TARGET_MODULES)

    set_seeds(args.seed)

    run_dir = ensure_dir(os.path.join(args.output_root, args.exp_name))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device, use_fp16 = select_device(args.fp16)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_loaders(args)
    spatial_h, spatial_w = train_set.spatial_shape
    num_vars = train_set.channels

    model, raw_model = build_model(num_vars, spatial_h, spatial_w, args, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check LoRA/base training configuration.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if not args.constant_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    criterion = LatitudeWeightedL1Loss(
        num_lat=spatial_h,
        lat_range=(float(np.min(train_set.latitudes)), float(np.max(train_set.latitudes))),
    ).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    ckpt = load_checkpoint_if_any(
        raw_model,
        optimizer,
        device,
        args.resume,
        args.resume_model_only,
        expect_lora=bool(args.enable_lora),
    )
    if ckpt and not args.resume_model_only:
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("val_loss", float("inf")))
    else:
        start_epoch = 1
        global_step = 0
        best_val = float("inf")
    no_improve = 0

    print(
        f"Autoregressive training | horizon={args.forecast_steps} | teacher_forcing={args.teacher_forcing} "
        f"| noise_std={args.input_noise_std} | noise_coarse_factor={args.noise_coarse_factor} "
        f"| target_offset={args.target_offset_steps} | constant_lr={args.constant_lr}"
    )
    if args.curriculum_start_steps and args.curriculum_max_steps:
        print(
            f"Curriculum: {args.curriculum_start_steps}->{args.curriculum_max_steps} "
            f"step={args.curriculum_step_size} every {args.curriculum_epoch_interval} epoch(s)"
        )
    prefetch_used = max(2, int(args.prefetch_factor)) if args.num_workers > 0 else 0
    print(
        f"Dataloader: workers={args.num_workers} | prefetch={prefetch_used} "
        f"| persistent={args.num_workers > 0 and not args.no_persistent_workers}"
    )
    print(f"Data: {num_vars} vars, {spatial_h}x{spatial_w}; train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}")

    epoch_rows: List[Dict[str, float]] = []
    epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.csv")
    loss_curve_path = os.path.join(run_dir, "loss_curve.png")

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_t0 = time.perf_counter()
        active_forecast_steps = resolve_active_forecast_steps(args, epoch)

        train_loss, train_mae, train_first, train_last, global_step, updates = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_fp16,
            active_forecast_steps,
            args.teacher_forcing,
            args.grad_clip,
            args.max_iters,
            global_step,
            args.input_noise_std,
            args.noise_coarse_factor,
            args.status_every_steps,
        )

        did_eval = (epoch % args.eval_every == 0)
        if did_eval:
            val_loss, val_mae, val_first, val_last = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_fp16,
                active_forecast_steps,
            )
        else:
            val_loss = val_mae = val_first = val_last = float("nan")

        epoch_sec = max(time.perf_counter() - epoch_t0, 1e-6)
        iter_per_sec = updates / epoch_sec
        samples_per_sec = (updates * args.batch_size) / epoch_sec
        if args.max_iters is not None:
            remaining_updates = max(int(args.max_iters) - int(global_step), 0)
            eta_hours = remaining_updates / max(iter_per_sec, 1e-9) / 3600.0
        else:
            remaining_epochs = max(args.max_epochs - epoch, 0)
            eta_hours = (remaining_epochs * epoch_sec) / 3600.0

        print(
            f"Epoch {epoch:03d} | horizon={active_forecast_steps} | train loss={train_loss:.4f} mae={train_mae:.4f} "
            f"(first={train_first:.4f} last={train_last:.4f}) | "
            f"val loss={val_loss:.4f} mae={val_mae:.4f} "
            f"(first={val_first:.4f} last={val_last:.4f}) | "
            f"sec={epoch_sec:.1f} | it/s={iter_per_sec:.2f} | sample/s={samples_per_sec:.2f} "
            f"| ETA={eta_hours:.2f}h"
        )

        epoch_rows.append(
            {
                "epoch": float(epoch),
                "horizon": float(active_forecast_steps),
                "global_step": float(global_step),
                "train_loss": float(train_loss),
                "train_mae": float(train_mae),
                "train_first_mae": float(train_first),
                "train_last_mae": float(train_last),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_first_mae": float(val_first),
                "val_last_mae": float(val_last),
                "it_per_sec": float(iter_per_sec),
                "samples_per_sec": float(samples_per_sec),
                "eta_hours": float(eta_hours),
            }
        )
        save_epoch_metrics_csv(epoch_rows, epoch_metrics_path)
        plot_training_curves(epoch_rows, loss_curve_path)

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        if did_eval:
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  ★ New best val loss: {best_val:.4f}")
            else:
                no_improve += 1

        if args.max_iters is not None and global_step >= args.max_iters:
            print(f"Reached max iterations ({args.max_iters}). Stopping.")
            break

        if scheduler is not None:
            scheduler.step()
        if did_eval and args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping after {no_improve} epochs without improvement.")
            break

    # Final test evaluation
    if args.skip_final_test_eval:
        print("Skipping final test evaluation (--skip-final-test-eval).")
    elif len(test_loader) > 0:
        test_loss, test_mae, test_first, test_last = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_fp16,
            args.forecast_steps,
        )
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(
                {
                    "test_loss": test_loss,
                    "test_mae": test_mae,
                    "test_first_mae": test_first,
                    "test_last_mae": test_last,
                    "best_val_loss": best_val,
                },
                f,
                indent=2,
            )
        print(
            f"Test | loss={test_loss:.4f} mae={test_mae:.4f} "
            f"(first={test_first:.4f} last={test_last:.4f})"
        )

    print(f"Done. Run directory: {run_dir}")
    print(f"Saved epoch metrics: {epoch_metrics_path}")
    if MATPLOTLIB_AVAILABLE:
        print(f"Saved training curves: {loss_curve_path}")
    else:
        print("Skipped loss curve plotting because matplotlib is unavailable.")


if __name__ == "__main__":
    main()
