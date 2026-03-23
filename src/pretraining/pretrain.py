#!/usr/bin/env python3
"""
FuXi one-step pretraining (Accelerate).

Scope:
- Pretraining only: (t-1, t) -> (t+1)
- Multi-GPU with Hugging Face Accelerate
- Mixed precision, gradient clipping, checkpointing
- Validation-driven best checkpoint

No rollout curriculum or post-training/fine-tuning logic is included here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed as hf_set_seed
from torch.utils.data import DataLoader, Dataset
import zarr

try:
    from .loss import LatitudeWeightedL1Loss
    from .model import FuXi
except ImportError:
    from loss import LatitudeWeightedL1Loss
    from model import FuXi


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_ZARR_STORE = (
    "/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/"
    "1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"
)

DEFAULT_PRESSURE_VARS = [
    "temperature",
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]
DEFAULT_SURFACE_VARS = ["t2m", "u10", "v20", "mslp", "tcwv"]
DEFAULT_PRESSURE_LEVELS = [850, 500, 250]

PRESSURE_VAR_ALIASES: Dict[str, str] = {
    "temperature": "temperature",
    "t": "temperature",
    "geopotential": "geopotential",
    "z": "geopotential",
    "specific_humidity": "specific_humidity",
    "q": "specific_humidity",
    "u_component_of_wind": "u_component_of_wind",
    "u": "u_component_of_wind",
    "v_component_of_wind": "v_component_of_wind",
    "v": "v_component_of_wind",
}

SURFACE_VAR_ALIASES: Dict[str, str] = {
    "2m_temperature": "2m_temperature",
    "t2m": "2m_temperature",
    "10m_u_component_of_wind": "10m_u_component_of_wind",
    "u10": "10m_u_component_of_wind",
    "10m_v_component_of_wind": "10m_v_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "v20": "10m_v_component_of_wind",
    "mean_sea_level_pressure": "mean_sea_level_pressure",
    "mslp": "mean_sea_level_pressure",
    "surface_pressure": "surface_pressure",
    "sp": "surface_pressure",
    "total_column_water_vapour": "total_column_water_vapour",
    "tcwv": "total_column_water_vapour",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FuXi one-step pretraining on WeatherBench2 Zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--zarr-store", type=str, default=DEFAULT_ZARR_STORE)
    data.add_argument("--train-start", type=str, default="1979-01-01")
    data.add_argument("--train-end", type=str, default="2018-12-31")
    data.add_argument("--val-start", type=str, default="2019-01-01")
    data.add_argument("--val-end", type=str, default="2020-12-31")
    data.add_argument("--test-start", type=str, default="2021-01-01")
    data.add_argument("--test-end", type=str, default="2022-12-31")
    data.add_argument("--history-steps", type=int, default=2)
    data.add_argument("--stats-samples", type=int, default=256)
    data.add_argument(
        "--pressure-vars",
        type=parse_csv_strings,
        default=list(DEFAULT_PRESSURE_VARS),
        help="Comma-separated pressure-level variables",
    )
    data.add_argument(
        "--surface-vars",
        type=parse_csv_strings,
        default=list(DEFAULT_SURFACE_VARS),
        help="Comma-separated surface variables",
    )
    data.add_argument(
        "--pressure-levels",
        type=parse_csv_ints,
        default=list(DEFAULT_PRESSURE_LEVELS),
        help="Comma-separated pressure levels",
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--embed-dim", type=int, default=1536)
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--window-size", type=int, default=8)
    model.add_argument("--depth-pre", type=int, default=2)
    model.add_argument("--depth-mid", type=int, default=44)
    model.add_argument("--depth-post", type=int, default=2)
    model.add_argument("--mlp-ratio", type=float, default=4.0)
    model.add_argument("--drop-path-rate", type=float, default=0.2)
    model.add_argument("--use-checkpoint", action="store_true")

    train = parser.add_argument_group("Training")
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--accum-steps", type=int, default=1)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--max-iters", type=int, default=40000)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=2.5e-4)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")
    train.add_argument(
        "--loss",
        type=str,
        choices=["l1", "rmse"],
        default="l1",
        help="If rmse is chosen, latitude-area weighting is applied.",
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--runs-dir", type=str, default="Models_paper/pretrain")
    io.add_argument("--exp-name", type=str, default=None)
    io.add_argument("--resume", type=str, default=None)
    io.add_argument("--seed", type=int, default=42)
    io.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    io.add_argument("--run-test-eval", action="store_true")
    io.add_argument("--plot-var-indices", type=parse_csv_ints, default=[0, 10, 19])

    return parser


# -----------------------------------------------------------------------------
# Variable resolution
# -----------------------------------------------------------------------------


def resolve_variable_names(
    zarr_path: str,
    requested_pressure_vars: Sequence[str],
    requested_surface_vars: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    store = zarr.open_group(zarr_path, mode="r")
    available = set(store.keys())
    notes: List[str] = []

    def resolve_one(name: str, aliases: Dict[str, str], kind: str) -> str:
        key = name.strip()
        key_lower = key.lower()

        if key in available:
            return key

        mapped = aliases.get(key_lower, key)
        if mapped in available:
            if mapped != key:
                notes.append(f"{kind}: mapped '{key}' -> '{mapped}'")
            return mapped

        if key_lower in ("mslp", "mean_sea_level_pressure") and "surface_pressure" in available:
            notes.append(f"{kind}: '{key}' missing, using fallback 'surface_pressure'")
            return "surface_pressure"

        raise KeyError(f"Could not resolve {kind} variable '{key}'. Available: {sorted(available)}")

    resolved_pressure = [resolve_one(v, PRESSURE_VAR_ALIASES, "pressure") for v in requested_pressure_vars]
    resolved_surface = [resolve_one(v, SURFACE_VAR_ALIASES, "surface") for v in requested_surface_vars]
    return resolved_pressure, resolved_surface, notes


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class FuXiZarrDataset(Dataset):
    """Lazy WeatherBench2 Zarr dataset for one-step pretraining."""

    def __init__(
        self,
        zarr_path: str,
        pressure_vars: Sequence[str],
        surface_vars: Sequence[str],
        pressure_levels: Sequence[int],
        history_steps: int,
        time_start: str,
        time_end: str,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        stats_samples: int = 256,
    ):
        super().__init__()

        if history_steps != 2:
            raise ValueError("FuXi model expects exactly 2 history steps.")

        self.zarr_path = zarr_path
        self.history_steps = history_steps
        self.pressure_vars = list(pressure_vars)
        self.surface_vars = list(surface_vars)
        self.pressure_levels = [int(v) for v in pressure_levels]

        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        self.store = zarr.open_group(zarr_path, mode="r")
        self._check_vars_exist()

        self.latitudes = self._get_coord("latitude", "lat")
        self.longitudes = self._get_coord("longitude", "lon")
        self.spatial_shape = (int(self.latitudes.shape[0]), int(self.longitudes.shape[0]))

        self._decode_time(time_start, time_end)
        self._resolve_level_indices()
        self._bind_arrays()
        self._infer_spatial_order()

        self.var_names = [
            f"{var}_plev{level}" for var in self.pressure_vars for level in self.pressure_levels
        ] + list(self.surface_vars)
        self.channels = len(self.var_names)

        if self.n_times <= self.history_steps:
            raise ValueError(
                f"Not enough timesteps ({self.n_times}) for history_steps={self.history_steps}"
            )

        if mean is None or std is None:
            self.mean, self.std = self._compute_stats(stats_samples)
        else:
            self.mean = mean.clone()
            self.std = std.clone()

    def _check_vars_exist(self) -> None:
        keys = set(self.store.keys())
        required = set(self.pressure_vars) | set(self.surface_vars) | {"time", "level"}
        missing = sorted(required - keys)
        if missing:
            raise KeyError(f"Missing zarr keys: {missing}; available: {sorted(keys)}")

    def _get_coord(self, primary: str, secondary: str) -> np.ndarray:
        if primary in self.store:
            return np.asarray(self.store[primary][:])
        if secondary in self.store:
            return np.asarray(self.store[secondary][:])
        raise KeyError(f"Neither '{primary}' nor '{secondary}' found in zarr")

    def _decode_time(self, time_start: str, time_end: str) -> None:
        raw_time = np.asarray(self.store["time"][:])
        attrs = dict(self.store["time"].attrs)
        units = attrs.get("units", "hours since 1959-01-01")

        if " since " not in units:
            raise ValueError(f"Unsupported time units: {units}")

        delta_unit_raw, base_date_raw = units.split(" since ", 1)
        delta_unit = delta_unit_raw.strip().lower().rstrip("s")
        unit_map = {"hour": "h", "minute": "m", "second": "s", "day": "D"}
        if delta_unit not in unit_map:
            raise ValueError(f"Unsupported delta unit '{delta_unit}' in '{units}'")

        base_date = np.datetime64(base_date_raw.strip())
        all_times = base_date + raw_time.astype(np.int64).astype(f"timedelta64[{unit_map[delta_unit]}]")
        mask = (all_times >= np.datetime64(time_start)) & (all_times <= np.datetime64(time_end))

        self._time_indices = np.where(mask)[0]
        self.n_times = int(self._time_indices.shape[0])
        if self.n_times == 0:
            raise ValueError(f"No timesteps found in range [{time_start}, {time_end}]")

    def _resolve_level_indices(self) -> None:
        available = np.asarray(self.store["level"][:]).astype(int)
        index_by_level = {int(level): idx for idx, level in enumerate(available.tolist())}
        missing = [level for level in self.pressure_levels if level not in index_by_level]
        if missing:
            raise ValueError(
                f"Requested pressure levels missing: {missing}; available: {available.tolist()}"
            )
        self._level_indices = [int(index_by_level[level]) for level in self.pressure_levels]

    def _bind_arrays(self) -> None:
        self._pressure_arrays = [self.store[name] for name in self.pressure_vars]
        self._surface_arrays = [self.store[name] for name in self.surface_vars]

    def _infer_spatial_order(self) -> None:
        lat_n, lon_n = self.spatial_shape
        pshape = self._pressure_arrays[0].shape
        if pshape[2] == lat_n and pshape[3] == lon_n:
            self._transpose_spatial = False
        elif pshape[2] == lon_n and pshape[3] == lat_n:
            self._transpose_spatial = True
        else:
            raise ValueError(
                f"Cannot infer spatial layout from pressure shape={pshape}, lat/lon={self.spatial_shape}"
            )

    def _load_frame_raw(self, index_in_slice: int) -> np.ndarray:
        t_idx = int(self._time_indices[index_in_slice])
        parts: List[np.ndarray] = []

        for arr in self._pressure_arrays:
            block = np.asarray(arr[t_idx, self._level_indices, :, :], dtype=np.float32)
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block)

        for arr in self._surface_arrays:
            block = np.asarray(arr[t_idx, :, :], dtype=np.float32)
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block[np.newaxis, ...])

        return np.concatenate(parts, axis=0)

    def _compute_stats(self, stats_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(1, min(stats_samples, self.n_times))
        sample_positions = np.linspace(0, self.n_times - 1, num=n, dtype=np.int64)

        ch_sum = np.zeros((len(self.var_names),), dtype=np.float64)
        ch_sum_sq = np.zeros((len(self.var_names),), dtype=np.float64)
        pixels = 0

        for pos in sample_positions.tolist():
            frame = self._load_frame_raw(int(pos))
            flat = frame.reshape(frame.shape[0], -1).astype(np.float64)
            ch_sum += flat.sum(axis=1)
            ch_sum_sq += np.square(flat).sum(axis=1)
            pixels += flat.shape[1]

        mean = ch_sum / max(pixels, 1)
        var = ch_sum_sq / max(pixels, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean_t = torch.from_numpy(mean.astype(np.float32)).view(-1, 1, 1)
        std_t = torch.from_numpy(std.astype(np.float32)).view(-1, 1, 1)
        return mean_t, std_t

    def __len__(self) -> int:
        return self.n_times - self.history_steps

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history_frames: List[torch.Tensor] = []
        for t in range(self.history_steps):
            frame = self._load_frame_raw(index + t)
            history_frames.append((torch.from_numpy(frame) - self.mean) / self.std)
        history = torch.stack(history_frames, dim=1)  # (C, 2, H, W)

        target = (torch.from_numpy(self._load_frame_raw(index + self.history_steps)) - self.mean) / self.std
        return history, target


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class LatitudeWeightedRMSELoss(nn.Module):
    """Area-weighted RMSE (cos(latitude) weighting)."""

    def __init__(self, num_lat: int, lat_range: Tuple[float, float]):
        super().__init__()
        latitudes = torch.linspace(lat_range[0], lat_range[1], num_lat)
        weights = torch.cos(torch.deg2rad(latitudes))
        weights = weights / weights.mean().clamp(min=1e-8)
        self.register_buffer("weights", weights.view(1, 1, -1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weighted_mse = torch.square(pred - target) * self.weights
        return torch.sqrt(weighted_mse.mean().clamp(min=1e-12))


def select_loss(loss_name: str, spatial_h: int, latitudes: np.ndarray, device: torch.device) -> nn.Module:
    lat_range = (float(np.min(latitudes)), float(np.max(latitudes)))
    if loss_name == "rmse":
        return LatitudeWeightedRMSELoss(num_lat=spatial_h, lat_range=lat_range).to(device)
    return LatitudeWeightedL1Loss(num_lat=spatial_h, lat_range=lat_range).to(device)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    accelerator: Accelerator,
    grad_clip: float,
    max_iters: Optional[int],
    global_step: int,
) -> Tuple[float, float, int]:
    model.train()
    device = accelerator.device

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    mae_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_sum = torch.zeros((), device=device, dtype=torch.float64)

    optimizer.zero_grad(set_to_none=True)
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bs = float(history.shape[0])

        with accelerator.accumulate(model):
            with accelerator.autocast():
                pred = model(history)
                loss = criterion(pred, target)
                mae = torch.abs(pred.float() - target.float()).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients and grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs_t = torch.tensor(bs, device=device, dtype=torch.float64)
        loss_sum += loss.detach().double() * bs_t
        mae_sum += mae.detach().double() * bs_t
        n_sum += bs_t

        global_step += 1
        if max_iters is not None and global_step >= max_iters:
            break

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    mae_sum = accelerator.reduce(mae_sum, reduction="sum")
    n_sum = accelerator.reduce(n_sum, reduction="sum")

    avg_loss = float((loss_sum / n_sum.clamp(min=1.0)).item())
    avg_mae = float((mae_sum / n_sum.clamp(min=1.0)).item())
    return avg_loss, avg_mae, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
) -> Tuple[float, float]:
    model.eval()
    device = accelerator.device

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    mae_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_sum = torch.zeros((), device=device, dtype=torch.float64)

    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bs = float(history.shape[0])

        with accelerator.autocast():
            pred = model(history)
            loss = criterion(pred, target)
            mae = torch.abs(pred.float() - target.float()).mean()

        bs_t = torch.tensor(bs, device=device, dtype=torch.float64)
        loss_sum += loss.detach().double() * bs_t
        mae_sum += mae.detach().double() * bs_t
        n_sum += bs_t

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    mae_sum = accelerator.reduce(mae_sum, reduction="sum")
    n_sum = accelerator.reduce(n_sum, reduction="sum")

    avg_loss = float((loss_sum / n_sum.clamp(min=1.0)).item())
    avg_mae = float((mae_sum / n_sum.clamp(min=1.0)).item())
    return avg_loss, avg_mae


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------


def save_history_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    fields = ["epoch", "global_step", "train_loss", "train_mae", "val_loss", "val_mae", "lr", "epoch_sec", "eta_hr"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def save_loss_curve(rows: List[dict], out_path: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FuXi Pretraining Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@torch.no_grad()
def save_prediction_maps(
    model: nn.Module,
    dataset: FuXiZarrDataset,
    accelerator: Accelerator,
    var_indices: Sequence[int],
    out_path: str,
) -> None:
    if len(dataset) == 0:
        return

    history, target = dataset[0]
    history = history.unsqueeze(0).to(accelerator.device)

    model.eval()
    with accelerator.autocast():
        pred = model(history).float().cpu()

    target = target.unsqueeze(0).float()
    mean = dataset.mean.unsqueeze(0).float()
    std = dataset.std.unsqueeze(0).float()
    pred_denorm = pred * std + mean
    target_denorm = target * std + mean

    valid = [i for i in var_indices if 0 <= i < dataset.channels]
    if not valid:
        valid = [0, dataset.channels // 2, dataset.channels - 1]

    lat = dataset.latitudes
    lon = dataset.longitudes
    n_rows = len(valid)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for ridx, vidx in enumerate(valid):
        name = dataset.var_names[vidx]
        tgt = target_denorm[0, vidx].numpy()
        prd = pred_denorm[0, vidx].numpy()
        vmin = float(min(tgt.min(), prd.min()))
        vmax = float(max(tgt.max(), prd.max()))

        for cidx, (arr, title) in enumerate(((tgt, "Ground Truth"), (prd, "Prediction"))):
            ax = axes[ridx, cidx]
            im = ax.imshow(
                arr,
                origin="lower",
                extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{title}: {name}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_accelerator(args: argparse.Namespace) -> Tuple[Accelerator, str]:
    if args.device == "cpu":
        accelerator = Accelerator(
            cpu=True,
            mixed_precision="no",
            gradient_accumulation_steps=args.accum_steps,
        )
        return accelerator, "none"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable.")

    mixed = "no" if args.amp == "none" else args.amp
    accelerator = Accelerator(
        mixed_precision=mixed,
        gradient_accumulation_steps=args.accum_steps,
    )
    if args.device == "cuda" and accelerator.device.type != "cuda":
        raise RuntimeError(
            f"Requested --device cuda but accelerate selected {accelerator.device}."
        )
    return accelerator, ("none" if mixed == "no" else mixed)


def main() -> None:
    args = build_parser().parse_args()
    accelerator, amp_mode = build_accelerator(args)
    hf_set_seed(args.seed, device_specific=True)

    if accelerator.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pressure_vars, surface_vars, resolution_notes = resolve_variable_names(
        args.zarr_store,
        args.pressure_vars,
        args.surface_vars,
    )

    if accelerator.is_main_process:
        print("=" * 88)
        print("FuXi Pretraining (One-Step)")
        print("=" * 88)
        print(f"Host                 : {os.uname().nodename}")
        print(f"Processes            : {accelerator.num_processes}")
        print(f"Device               : {accelerator.device}")
        print(f"Mixed precision      : {amp_mode}")
        print(f"Train split          : {args.train_start} -> {args.train_end}")
        print(f"Val split            : {args.val_start} -> {args.val_end}")
        print(f"Test split           : {args.test_start} -> {args.test_end}")
        print(f"Surface vars         : {surface_vars}")
        print(f"Pressure vars        : {pressure_vars}")
        print(f"Pressure levels      : {args.pressure_levels}")
        print(f"Batch size / process : {args.batch_size}")
        print(
            "Effective batch size : "
            f"{args.batch_size * args.accum_steps * accelerator.num_processes}"
        )
        if resolution_notes:
            print("Variable mapping notes:")
            for note in resolution_notes:
                print(f"  - {note}")
        print("=" * 88)

    t0 = time.time()
    train_set = FuXiZarrDataset(
        zarr_path=args.zarr_store,
        pressure_vars=pressure_vars,
        surface_vars=surface_vars,
        pressure_levels=args.pressure_levels,
        history_steps=args.history_steps,
        time_start=args.train_start,
        time_end=args.train_end,
        stats_samples=args.stats_samples,
    )
    val_set = FuXiZarrDataset(
        zarr_path=args.zarr_store,
        pressure_vars=pressure_vars,
        surface_vars=surface_vars,
        pressure_levels=args.pressure_levels,
        history_steps=args.history_steps,
        time_start=args.val_start,
        time_end=args.val_end,
        mean=train_set.mean,
        std=train_set.std,
    )
    test_set = FuXiZarrDataset(
        zarr_path=args.zarr_store,
        pressure_vars=pressure_vars,
        surface_vars=surface_vars,
        pressure_levels=args.pressure_levels,
        history_steps=args.history_steps,
        time_start=args.test_start,
        time_end=args.test_end,
        mean=train_set.mean,
        std=train_set.std,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = FuXi(
        num_variables=train_set.channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
        depth_pre=args.depth_pre,
        depth_mid=args.depth_mid,
        depth_post=args.depth_post,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        input_height=train_set.spatial_shape[0],
        input_width=train_set.spatial_shape[1],
        use_checkpoint=args.use_checkpoint,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.max_epochs),
        eta_min=args.lr * 0.01,
    )

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    criterion = select_loss(
        loss_name=args.loss,
        spatial_h=train_set.spatial_shape[0],
        latitudes=train_set.latitudes,
        device=accelerator.device,
    )
    raw_model = accelerator.unwrap_model(model)

    exp_name = args.exp_name or datetime.now().strftime("pretrain_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, exp_name)
    plots_dir = os.path.join(run_dir, "plots")
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Loaded datasets in {time.time() - t0:.1f}s")
        print(
            f"Samples (train/val/test): {len(train_set):,} / {len(val_set):,} / {len(test_set):,}"
        )
        print(f"Channels             : {train_set.channels}")
        print(f"Spatial shape        : {train_set.spatial_shape}")
        print(f"Model parameters     : {raw_model.count_parameters():,}")
        print(f"Run directory        : {run_dir}")

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history_rows: List[dict] = []
    epoch_secs: List[float] = []

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if accelerator.is_main_process:
            print(f"Resumed from         : {args.resume}")
            print(f"Start epoch          : {start_epoch}")
            print(f"Global step          : {global_step}")
            print(f"Best val so far      : {best_val_loss:.6f}")

    if accelerator.is_main_process:
        config = vars(args).copy()
        config.update(
            {
                "resolved_pressure_vars": pressure_vars,
                "resolved_surface_vars": surface_vars,
                "channels": train_set.channels,
                "spatial_shape": list(train_set.spatial_shape),
                "num_parameters": raw_model.count_parameters(),
                "world_size": accelerator.num_processes,
                "effective_batch_size": args.batch_size * args.accum_steps * accelerator.num_processes,
            }
        )
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    no_improve_epochs = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_mae, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            accelerator=accelerator,
            grad_clip=args.grad_clip,
            max_iters=args.max_iters,
            global_step=global_step,
        )

        val_loss, val_mae = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            accelerator=accelerator,
        )

        scheduler.step()
        epoch_sec = time.time() - epoch_start
        epoch_secs.append(epoch_sec)
        avg_epoch_sec = float(np.mean(epoch_secs))
        eta_hr = avg_epoch_sec * max(0, args.max_epochs - epoch) / 3600.0
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if accelerator.is_main_process:
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "lr": lr_now,
                "epoch_sec": epoch_sec,
                "eta_hr": eta_hr,
            }
            history_rows.append(row)

            print(
                f"Epoch {epoch:03d}/{args.max_epochs} | "
                f"train={train_loss:.6f} mae={train_mae:.6f} | "
                f"val={val_loss:.6f} mae={val_mae:.6f} | "
                f"lr={lr_now:.3e} | step={global_step} | "
                f"time={epoch_sec:.1f}s | eta={eta_hr:.2f}h"
            )

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "model_state": accelerator.get_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history_rows,
                "config": vars(args),
            }
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if improved:
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  * new best val loss: {best_val_loss:.6f}")

        accelerator.wait_for_everyone()

        if args.max_iters is not None and global_step >= args.max_iters:
            if accelerator.is_main_process:
                print(f"Reached max iterations ({args.max_iters}).")
            break

        if no_improve_epochs >= args.patience:
            if accelerator.is_main_process:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    accelerator.wait_for_everyone()

    best_ckpt = os.path.join(run_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])

    metrics = {
        "best_val_loss": float(best_val_loss),
        "global_step": int(global_step),
        "epochs_ran": len(history_rows),
        "num_parameters": raw_model.count_parameters(),
        "epoch_time_sec_mean": float(np.mean(epoch_secs)) if epoch_secs else None,
        "epoch_time_sec_last": float(epoch_secs[-1]) if epoch_secs else None,
    }

    if args.run_test_eval:
        test_loss, test_mae = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            accelerator=accelerator,
        )
        metrics["test_loss"] = float(test_loss)
        metrics["test_mae"] = float(test_mae)

    if accelerator.is_main_process:
        save_history_csv(history_rows, os.path.join(run_dir, "history.csv"))
        save_loss_curve(history_rows, os.path.join(plots_dir, "loss_curve.png"))
        save_prediction_maps(
            model=raw_model,
            dataset=test_set,
            accelerator=accelerator,
            var_indices=args.plot_var_indices,
            out_path=os.path.join(plots_dir, "prediction_maps.png"),
        )
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("=" * 88)
        print("Pretraining complete")
        print(f"Checkpoints: {run_dir}/best.pt and {run_dir}/last.pt")
        print("=" * 88)


if __name__ == "__main__":
    main()
