#!/usr/bin/env python3
"""
FuXi training script – paper-faithful (Zarr version).

Directly uses Zarr store for lazy data loading.
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional: TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from model import FuXi
from loss import LatitudeWeightedL1Loss
# Import the Zarr dataset class (adjust import if needed)
from fuxi_train import FuXiZarrDataset

# ---- Zarr store --------------------------------------------------------------
ZARR_STORE = "/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

def parse_args():
    p = argparse.ArgumentParser(description="FuXi Zarr training")

    # --- Mode ---
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--stage", choices=["short", "medium", "long"], default="short")

    # --- Data ---
    p.add_argument("--zarr-store", type=str, default=ZARR_STORE, help="Path to Zarr store")
    p.add_argument("--train-start", type=str, default="1979-01-01")
    p.add_argument("--train-end", type=str, default="1979-12-31")
    p.add_argument("--val-start", type=str, default="2016-01-01")
    p.add_argument("--val-end", type=str, default="2016-06-30")
    p.add_argument("--test-start", type=str, default="2018-01-01")
    p.add_argument("--test-end", type=str, default="2018-06-30")
    p.add_argument("--history-steps", type=int, default=2)

    # --- Training ---
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)

    # --- Optimizer ---
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)

    # --- Model ---
    p.add_argument("--embed-dim", type=int, default=1536)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--drop-path-rate", type=float, default=0.2)
    p.add_argument("--use-checkpoint", action="store_true")

    # --- I/O ---
    p.add_argument("--output-root", type=str, default="Models")
    p.add_argument("--exp-name", type=str, default="paper_zarr")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--tensorboard", action="store_true")

    return p.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def plot_losses(train_losses, val_losses, outdir):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Lat-weighted L1")
    plt.legend()
    plt.title("Training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)
    plt.close()

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_fp16):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            pred = model(history)
            loss = criterion(pred, target)
        total_loss += loss.item()
        total_mae += torch.abs(pred.float() - target.float()).mean().item()
        count += 1
    return total_loss / max(count, 1), total_mae / max(count, 1)

def pretrain_one_epoch(model, loader, optimizer, criterion, device, scaler,
                       use_fp16, max_iters=None, global_step=0):
    model.train()
    total_loss = 0.0
    count = 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            pred = model(history)
            loss = criterion(pred, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        count += 1
        global_step += 1
        if max_iters is not None and global_step >= max_iters:
            break
    return total_loss / max(count, 1), global_step

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage_name = args.exp_name
    run_dir = ensure_dir(os.path.join(args.output_root, stage_name))
    plots_dir = ensure_dir(os.path.join(run_dir, "Plots"))

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"

    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        print(f"TensorBoard logging enabled at {os.path.join(run_dir, 'tensorboard')}")

    # ---- Data: Use Zarr Dataset ----
    print(f"Loading training data from Zarr: {args.zarr_store}")
    train_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.train_start,
        time_end=args.train_end,
    )
    print(f"Loading validation data from Zarr: {args.zarr_store}")
    val_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        mean=train_set.mean,
        std=train_set.std,
        time_start=args.val_start,
        time_end=args.val_end,
    )
    print(f"Loading test data from Zarr: {args.zarr_store}")
    test_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        mean=train_set.mean,
        std=train_set.std,
        time_start=args.test_start,
        time_end=args.test_end,
    )

    spatial_h, spatial_w = train_set.spatial_shape
    num_vars = train_set.channels
    print(f"Data: {num_vars} variables, {spatial_h}×{spatial_w} grid, "
          f"{len(train_set)} train / {len(val_set)} val samples")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ----
    model = FuXi(
        num_variables=num_vars,
        embed_dim=args.embed_dim,
        num_heads=8,  # Default for paper
        window_size=args.window_size,
        depth_pre=2,
        depth_mid=48,  # Paper-faithful
        depth_post=2,
        mlp_ratio=4.0,
        drop_path_rate=args.drop_path_rate,
        input_height=spatial_h,
        input_width=spatial_w,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    num_gpus = args.gpus or torch.cuda.device_count()
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    raw_model = model.module if hasattr(model, 'module') else model

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            raw_model.load_state_dict(ckpt["model_state"])
        else:
            raw_model.load_state_dict(ckpt)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    print(f"Optimizer: AdamW  lr={args.lr}  wd={args.weight_decay}  "
          f"betas=({args.beta1}, {args.beta2})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32)
    criterion = LatitudeWeightedL1Loss(
        num_lat=spatial_h,
        lat_range=(latitudes.min().item(), latitudes.max().item()),
    ).to(device)

    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    best_val = float("inf")
    train_losses, val_losses = [], []
    no_improve = 0
    global_step = 0

    print("\n" + "=" * 60)
    print("PRE-TRAINING  (single-step prediction, Zarr)")
    print("=" * 60)

    for epoch in range(1, args.max_epochs + 1):
        train_loss, global_step = pretrain_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, use_fp16, args.max_iters, global_step,
        )
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device, use_fp16,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MAE/val', val_mae, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:3d} | train={train_loss:.5f}  "
              f"val={val_loss:.5f}  mae={val_mae:.5f}  "
              f"step={global_step}")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"  ★ New best: {best_val:.5f}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs w/o improvement.")
            break

        if args.max_iters and global_step >= args.max_iters:
            print(f"Reached max iterations ({args.max_iters}).")
            break

        scheduler.step()

    if train_losses:
        plot_losses(train_losses, val_losses, plots_dir)

    if writer is not None:
        writer.close()

    if len(test_set) > 0:
        print(f"\nEvaluating on test set")
        test_loss, test_mae = evaluate(
            model, test_loader, criterion, device, use_fp16,
        )
        print(f"Test: loss={test_loss:.5f}  mae={test_mae:.5f}")

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({
                "test_loss": test_loss,
                "test_mae": test_mae,
                "best_val_loss": best_val,
                "num_params": raw_model.count_parameters(),
            }, f, indent=2)

    print(f"\nDone!  Results saved to: {run_dir}")

if __name__ == "__main__":
    main()