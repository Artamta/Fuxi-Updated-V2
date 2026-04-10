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
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    yaml = None

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


def parse_csv_list(value: Optional[str]):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


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

    def set_if(section: str, key: str, attr: str = None):
        attr = attr or key
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
        "skip_final_test_eval",
    ]:
        set_if("train", key)

    for key in ["output_root", "exp_name", "resume", "seed"]:
        set_if("output", key)

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

    # Variables / levels
    p.add_argument("--pressure-vars", type=str, default=None,
                   help="Comma-separated pressure variable names")
    p.add_argument("--surface-vars", type=str, default=None,
                   help="Comma-separated surface variable names")
    p.add_argument("--pressure-levels", type=str, default=None,
                   help="Comma-separated pressure levels (ints)")

    return p.parse_args()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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
    dataset_forecast_steps = args.forecast_steps
    if args.curriculum_max_steps and args.curriculum_max_steps > 0:
        dataset_forecast_steps = max(dataset_forecast_steps, int(args.curriculum_max_steps))

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

    num_gpus = args.gpus or torch.cuda.device_count()
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    raw_model = model.module if hasattr(model, "module") else model
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
                    input_noise_std, noise_coarse_factor):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_first = 0.0
    total_last = 0.0
    count = 0
    updates = 0

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


def load_checkpoint_if_any(raw_model, optimizer, device, resume_path: Optional[str], resume_model_only: bool = False):
    if not resume_path:
        return None
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt.get("model_state", ckpt))
    if (not resume_model_only) and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def main():
    args = parse_args()
    cfg = load_config(args.config)
    args = apply_config(args, cfg)

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

    optimizer = optim.AdamW(
        model.parameters(),
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

    ckpt = load_checkpoint_if_any(raw_model, optimizer, device, args.resume, args.resume_model_only)
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


if __name__ == "__main__":
    main()
