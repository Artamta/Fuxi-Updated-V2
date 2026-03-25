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
from typing import Optional, Tuple

import numpy as np
import torch
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
    p.add_argument("--teacher-forcing", type=float, default=0.0,
                   help="Probability of using ground truth as next input during training")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

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

    def __init__(self, *args, forecast_steps: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be >= 1")
        self.forecast_steps = int(forecast_steps)

    def __len__(self) -> int:
        return self.n_times - (self.history_steps + self.forecast_steps - 1)

    def __getitem__(self, index: int):
        history_frames = []
        for t in range(self.history_steps):
            frame = self._load_frame_raw(index + t)
            history_frames.append((torch.from_numpy(frame) - self.mean) / self.std)
        history = torch.stack(history_frames, dim=1)  # (C, history, H, W)

        targets = []
        for t in range(self.forecast_steps):
            frame = self._load_frame_raw(index + self.history_steps + t)
            targets.append((torch.from_numpy(frame) - self.mean) / self.std)
        target_seq = torch.stack(targets, dim=0)  # (forecast, C, H, W)
        return history, target_seq


def build_loaders(args):
    train_set = AutoregressiveFuXiDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.train_start,
        time_end=args.train_end,
        forecast_steps=args.forecast_steps,
        pressure_vars=args.pressure_vars,
        surface_vars=args.surface_vars,
        pressure_levels=args.pressure_levels,
    )

    val_set = AutoregressiveFuXiDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.val_start,
        time_end=args.val_end,
        forecast_steps=args.forecast_steps,
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
        forecast_steps=args.forecast_steps,
        mean=train_set.mean,
        std=train_set.std,
        pressure_vars=args.pressure_vars,
        surface_vars=args.surface_vars,
        pressure_levels=args.pressure_levels,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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
                    forecast_steps, teacher_forcing, grad_clip, max_iters, global_step):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_first = 0.0
    total_last = 0.0
    count = 0

    for history, targets in loader:
        history = history.to(device)
        targets = targets.to(device)
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
            history = history.to(device)
            targets = targets.to(device)
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


def load_checkpoint_if_any(raw_model, optimizer, device, resume_path: Optional[str]):
    if not resume_path:
        return None
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt.get("model_state", ckpt))
    if "optimizer_state" in ckpt:
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
        json.dump(vars(args), f, indent=2)

    device, use_fp16 = select_device(args.fp16)

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    criterion = LatitudeWeightedL1Loss(
        num_lat=spatial_h,
        lat_range=(float(np.min(train_set.latitudes)), float(np.max(train_set.latitudes))),
    ).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    ckpt = load_checkpoint_if_any(raw_model, optimizer, device, args.resume)
    start_epoch = int(ckpt.get("epoch", 0)) + 1 if ckpt else 1
    global_step = int(ckpt.get("global_step", 0)) if ckpt else 0
    best_val = float("inf") if ckpt is None else float(ckpt.get("val_loss", float("inf")))
    no_improve = 0

    print(f"Autoregressive training | horizon={args.forecast_steps} | teacher_forcing={args.teacher_forcing}")
    print(f"Data: {num_vars} vars, {spatial_h}x{spatial_w}; train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}")

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss, train_mae, train_first, train_last, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_fp16,
            args.forecast_steps,
            args.teacher_forcing,
            args.grad_clip,
            args.max_iters,
            global_step,
        )

        did_eval = (epoch % args.eval_every == 0)
        if did_eval:
            val_loss, val_mae, val_first, val_last = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_fp16,
                args.forecast_steps,
            )
        else:
            val_loss = val_mae = val_first = val_last = float("nan")

        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f} mae={train_mae:.4f} "
            f"(first={train_first:.4f} last={train_last:.4f}) | "
            f"val loss={val_loss:.4f} mae={val_mae:.4f} "
            f"(first={val_first:.4f} last={val_last:.4f})"
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

        scheduler.step()
        if did_eval and args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping after {no_improve} epochs without improvement.")
            break

    # Final test evaluation
    if len(test_loader) > 0:
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
