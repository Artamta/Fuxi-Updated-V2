#!/usr/bin/env python3
"""
Evaluate FuXi checkpoints with autoregressive rollout metrics.

Outputs (publication-oriented):
- Per-variable per-lead-step RMSE and ACC CSV
- Per-variable per-day RMSE and ACC CSV
- Summary JSON
- Heatmaps (RMSE/ACC) + selected-variable lead-time curves
- Optional prediction sample NPZ files for inspection

Metric conventions:
- RMSE is latitude-area weighted (cos(latitude) weighting)
- ACC uses anomaly correlation against a climatology field
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import zarr

try:
    from ..models.fuxi_model import FuXi
except ImportError:
    from src.models.fuxi_model import FuXi


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


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FuXi checkpoint with autoregressive rollout metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt/last.pt")
    parser.add_argument("--config", type=str, default=None, help="Optional config.json override")

    parser.add_argument("--zarr-store", type=str, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)

    parser.add_argument("--pressure-vars", type=parse_csv_strings, default=None)
    parser.add_argument("--surface-vars", type=parse_csv_strings, default=None)
    parser.add_argument("--pressure-levels", type=parse_csv_ints, default=None)

    parser.add_argument("--climo-start", type=str, default=None)
    parser.add_argument("--climo-end", type=str, default=None)
    parser.add_argument("--climo-cache", type=str, default=None)

    parser.add_argument("--rollout-steps", type=int, default=60, help="60 steps = 15 days at 6-hourly")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for faster checks")
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--history-steps", type=int, default=2)

    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-n-samples", type=int, default=3)
    parser.add_argument(
        "--plot-vars",
        type=parse_csv_strings,
        default=None,
        help="Comma-separated variable names (exact) for lead-time curves",
    )
    return parser


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


@dataclass
class DataSpec:
    zarr_store: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    pressure_vars: List[str]
    surface_vars: List[str]
    pressure_levels: List[int]
    history_steps: int
    embed_dim: int
    num_heads: int
    window_size: int
    depth_pre: int
    depth_mid: int
    depth_post: int
    mlp_ratio: float
    drop_path_rate: float


class WB2Accessor:
    def __init__(self, zarr_path: str, pressure_vars: Sequence[str], surface_vars: Sequence[str], pressure_levels: Sequence[int]):
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        self.store = zarr.open_group(zarr_path, mode="r")
        self.pressure_vars = list(pressure_vars)
        self.surface_vars = list(surface_vars)
        self.pressure_levels = [int(v) for v in pressure_levels]

        self.latitudes = self._get_coord("latitude", "lat")
        self.longitudes = self._get_coord("longitude", "lon")
        self.spatial_shape = (int(self.latitudes.shape[0]), int(self.longitudes.shape[0]))
        self.time_values = self._decode_time()

        self._resolve_level_indices()
        self._bind_arrays()
        self._infer_spatial_order()

        self.var_names = [
            f"{v}_plev{p}" for v in self.pressure_vars for p in self.pressure_levels
        ] + list(self.surface_vars)
        self.channels = len(self.var_names)

    def _get_coord(self, primary: str, secondary: str) -> np.ndarray:
        if primary in self.store:
            return np.asarray(self.store[primary][:])
        if secondary in self.store:
            return np.asarray(self.store[secondary][:])
        raise KeyError(f"Neither '{primary}' nor '{secondary}' found in zarr")

    def _decode_time(self) -> np.ndarray:
        raw_time = np.asarray(self.store["time"][:])
        attrs = dict(self.store["time"].attrs)
        units = attrs.get("units", "hours since 1959-01-01")
        if " since " not in units:
            raise ValueError(f"Unsupported time units: {units}")
        delta_unit_raw, base_date_raw = units.split(" since ", 1)
        delta_unit = delta_unit_raw.strip().lower().rstrip("s")
        unit_map = {"hour": "h", "minute": "m", "second": "s", "day": "D"}
        if delta_unit not in unit_map:
            raise ValueError(f"Unsupported delta unit '{delta_unit}'")
        base_date = np.datetime64(base_date_raw.strip())
        return base_date + raw_time.astype(np.int64).astype(f"timedelta64[{unit_map[delta_unit]}]")

    def time_indices_between(self, start_date: str, end_date: str) -> np.ndarray:
        mask = (self.time_values >= np.datetime64(start_date)) & (self.time_values <= np.datetime64(end_date))
        return np.where(mask)[0]

    def _resolve_level_indices(self) -> None:
        levels = np.asarray(self.store["level"][:]).astype(int)
        idx_by_level = {int(v): i for i, v in enumerate(levels.tolist())}
        missing = [v for v in self.pressure_levels if v not in idx_by_level]
        if missing:
            raise ValueError(f"Missing requested levels {missing}; available {levels.tolist()}")
        self._level_indices = [int(idx_by_level[v]) for v in self.pressure_levels]

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
            raise ValueError(f"Cannot infer spatial layout from shape={pshape}, lat/lon={self.spatial_shape}")

    def load_frame(self, abs_time_idx: int) -> np.ndarray:
        parts: List[np.ndarray] = []
        for arr in self._pressure_arrays:
            block = np.asarray(arr[abs_time_idx, self._level_indices, :, :], dtype=np.float32)
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block)
        for arr in self._surface_arrays:
            block = np.asarray(arr[abs_time_idx, :, :], dtype=np.float32)
            if self._transpose_spatial:
                block = np.swapaxes(block, -2, -1)
            parts.append(block[np.newaxis, ...])
        return np.concatenate(parts, axis=0)


def compute_channel_stats(accessor: WB2Accessor, time_indices: np.ndarray, stats_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = max(1, min(int(stats_samples), int(time_indices.shape[0])))
    sampled_pos = np.linspace(0, time_indices.shape[0] - 1, num=n, dtype=np.int64)

    ch_sum = np.zeros((accessor.channels,), dtype=np.float64)
    ch_sq = np.zeros((accessor.channels,), dtype=np.float64)
    pixels = 0
    for pos in sampled_pos.tolist():
        frame = accessor.load_frame(int(time_indices[pos]))
        flat = frame.reshape(frame.shape[0], -1).astype(np.float64)
        ch_sum += flat.sum(axis=1)
        ch_sq += np.square(flat).sum(axis=1)
        pixels += flat.shape[1]

    mean = ch_sum / max(pixels, 1)
    var = ch_sq / max(pixels, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    mean_t = torch.from_numpy(mean.astype(np.float32)).view(-1, 1, 1)
    std_t = torch.from_numpy(std.astype(np.float32)).view(-1, 1, 1)
    return mean_t, std_t


def compute_climatology(
    accessor: WB2Accessor,
    climo_indices: np.ndarray,
    cache_path: Optional[Path],
) -> np.ndarray:
    if cache_path is not None and cache_path.is_file():
        data = np.load(cache_path)
        if "climatology" in data:
            clim = data["climatology"]
            if clim.shape[0] == accessor.channels:
                return clim.astype(np.float32)

    ch_sum = np.zeros((accessor.channels, accessor.spatial_shape[0], accessor.spatial_shape[1]), dtype=np.float64)
    n = 0
    for i, t_idx in enumerate(climo_indices.tolist(), start=1):
        frame = accessor.load_frame(int(t_idx)).astype(np.float64)
        ch_sum += frame
        n += 1
        if i % 2000 == 0:
            print(f"  climatology progress: {i}/{len(climo_indices)}")

    clim = (ch_sum / max(n, 1)).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, climatology=clim)
    return clim


class RolloutDataset(Dataset):
    def __init__(
        self,
        accessor: WB2Accessor,
        eval_time_indices: np.ndarray,
        rollout_steps: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        history_steps: int = 2,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        if history_steps != 2:
            raise ValueError("FuXi expects history_steps=2")
        self.accessor = accessor
        self.eval_time_indices = eval_time_indices
        self.rollout_steps = int(rollout_steps)
        self.mean = mean
        self.std = std
        self.history_steps = history_steps

        n = eval_time_indices.shape[0]
        usable = n - history_steps - self.rollout_steps + 1
        if usable <= 0:
            raise ValueError(
                f"Not enough eval timesteps ({n}) for history={history_steps} and rollout={rollout_steps}"
            )
        self.starts = np.arange(usable, dtype=np.int64)
        if max_samples is not None:
            self.starts = self.starts[: int(max_samples)]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        abs_idx = self.eval_time_indices

        f0 = self.accessor.load_frame(int(abs_idx[s + 0]))
        f1 = self.accessor.load_frame(int(abs_idx[s + 1]))
        h0 = (torch.from_numpy(f0) - self.mean) / self.std
        h1 = (torch.from_numpy(f1) - self.mean) / self.std
        history = torch.stack([h0, h1], dim=1)  # (C,2,H,W)

        future_list: List[torch.Tensor] = []
        for k in range(self.rollout_steps):
            fr = self.accessor.load_frame(int(abs_idx[s + self.history_steps + k]))
            fr_t = (torch.from_numpy(fr) - self.mean) / self.std
            future_list.append(fr_t)
        future = torch.stack(future_list, dim=0)  # (S,C,H,W)

        init_time = str(self.accessor.time_values[int(abs_idx[s + 1])])
        return history, future, init_time


def choose_device(device_mode: str) -> torch.device:
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model(spec: DataSpec, channels: int, spatial_shape: Tuple[int, int], checkpoint_state: Dict[str, torch.Tensor]) -> FuXi:
    model = FuXi(
        num_variables=channels,
        embed_dim=int(spec.embed_dim),
        num_heads=int(spec.num_heads),
        window_size=int(spec.window_size),
        depth_pre=int(spec.depth_pre),
        depth_mid=int(spec.depth_mid),
        depth_post=int(spec.depth_post),
        mlp_ratio=float(spec.mlp_ratio),
        drop_path_rate=float(spec.drop_path_rate),
        input_height=int(spatial_shape[0]),
        input_width=int(spatial_shape[1]),
        use_checkpoint=False,
    )
    model.load_state_dict(checkpoint_state, strict=True)
    model.eval()
    return model


def build_lat_weights(latitudes: np.ndarray, device: torch.device) -> torch.Tensor:
    lat = torch.tensor(latitudes, dtype=torch.float32, device=device)
    w = torch.cos(torch.deg2rad(lat))
    w = w / w.mean().clamp(min=1e-8)
    return w.view(1, 1, -1, 1)


@torch.no_grad()
def evaluate_rollout(
    model: FuXi,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    climatology: torch.Tensor,
    lat_w: torch.Tensor,
    rollout_steps: int,
    device: torch.device,
    amp: str,
) -> Tuple[np.ndarray, np.ndarray]:
    channels = int(mean.shape[0])
    rmse_mse_sum = torch.zeros((rollout_steps, channels), dtype=torch.float64, device=device)
    acc_sum = torch.zeros((rollout_steps, channels), dtype=torch.float64, device=device)
    count = torch.zeros((rollout_steps,), dtype=torch.float64, device=device)

    mean = mean.to(device)
    std = std.to(device)
    clim = climatology.to(device)

    total_batches = len(loader)
    for batch_idx, (history, future, _init_time) in enumerate(loader, start=1):
        history = history.to(device, non_blocking=True)  # (B,C,2,H,W)
        future = future.to(device, non_blocking=True)    # (B,S,C,H,W)
        batch_size = history.shape[0]
        hist = history

        for s in range(rollout_steps):
            with autocast_ctx(device, amp):
                pred_n = model(hist)  # (B,C,H,W) normalized

            tgt_n = future[:, s]
            pred = pred_n.float() * std + mean
            tgt = tgt_n.float() * std + mean

            err = pred - tgt
            mse = (err.square() * lat_w).mean(dim=(2, 3))  # (B,C)
            rmse_mse_sum[s] += mse.double().sum(dim=0)

            pred_anom = pred - clim
            tgt_anom = tgt - clim
            num = (pred_anom * tgt_anom * lat_w).sum(dim=(2, 3))  # (B,C)
            den = torch.sqrt(
                (pred_anom.square() * lat_w).sum(dim=(2, 3))
                * (tgt_anom.square() * lat_w).sum(dim=(2, 3))
                + 1e-12
            )
            acc = num / den
            acc_sum[s] += acc.double().sum(dim=0)

            count[s] += float(batch_size)

            # Next history uses predicted normalized frame
            hist = torch.stack([hist[:, :, 1], pred_n], dim=2)

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            done = int(count[0].item())
            avg_rmse_step1 = float(torch.sqrt((rmse_mse_sum[0] / count[0].clamp(min=1.0)).clamp(min=1e-12)).mean().item())
            avg_acc_step1 = float((acc_sum[0] / count[0].clamp(min=1.0)).mean().item())
            print(
                f"[rollout] batch {batch_idx}/{total_batches} | samples={done} | "
                f"step1_mean_rmse={avg_rmse_step1:.4f} | step1_mean_acc={avg_acc_step1:.4f}",
                flush=True,
            )

    denom = count.clamp(min=1.0).unsqueeze(1)
    rmse = torch.sqrt((rmse_mse_sum / denom).clamp(min=1e-12)).cpu().numpy()
    acc = (acc_sum / denom).cpu().numpy()
    return rmse, acc


@torch.no_grad()
def save_prediction_samples(
    model: FuXi,
    dataset: RolloutDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    amp: str,
    output_dir: Path,
    n_samples: int,
    var_names: Sequence[str],
) -> None:
    n = min(int(n_samples), len(dataset))
    if n <= 0:
        return

    sample_dir = output_dir / "prediction_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    mean_d = mean.to(device)
    std_d = std.to(device)

    for i in range(n):
        history, future, init_time = dataset[i]
        history = history.unsqueeze(0).to(device)
        future = future.unsqueeze(0).to(device)

        preds = []
        hist = history
        rollout_steps = future.shape[1]
        for s in range(rollout_steps):
            with autocast_ctx(device, amp):
                pred_n = model(hist)
            preds.append(pred_n)
            hist = torch.stack([hist[:, :, 1], pred_n], dim=2)
        pred_n = torch.stack(preds, dim=1)  # (1,S,C,H,W)

        pred = (pred_n.float() * std_d + mean_d).cpu().numpy()[0]
        truth = (future.float() * std_d + mean_d).cpu().numpy()[0]

        np.savez_compressed(
            sample_dir / f"sample_{i:03d}.npz",
            init_time=np.array(init_time),
            pred=pred,
            truth=truth,
            var_names=np.array(var_names, dtype=object),
        )


def write_metric_csvs(
    out_dir: Path,
    var_names: Sequence[str],
    rmse: np.ndarray,
    acc: np.ndarray,
) -> None:
    rollout_steps = rmse.shape[0]
    lead_steps = np.arange(1, rollout_steps + 1, dtype=np.int64)
    lead_hours = lead_steps * 6
    lead_days = lead_hours / 24.0

    per_lead_path = out_dir / "metrics_per_lead.csv"
    with per_lead_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "lead_step", "lead_hour", "lead_day", "rmse", "acc"])
        for vi, vname in enumerate(var_names):
            for si in range(rollout_steps):
                writer.writerow([vname, int(lead_steps[si]), int(lead_hours[si]), float(lead_days[si]), float(rmse[si, vi]), float(acc[si, vi])])

    # Per-day aggregation (4 steps/day)
    n_days = rollout_steps // 4
    per_day_path = out_dir / "metrics_per_day.csv"
    with per_day_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "day", "rmse_mean", "acc_mean"])
        for vi, vname in enumerate(var_names):
            for d in range(n_days):
                s0 = d * 4
                s1 = s0 + 4
                writer.writerow([vname, d + 1, float(np.mean(rmse[s0:s1, vi])), float(np.mean(acc[s0:s1, vi]))])


def plot_heatmap(values: np.ndarray, var_names: Sequence[str], title: str, cmap: str, out_path: Path, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(14, max(6, len(var_names) * 0.35)))
    im = plt.imshow(values.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    x_ticks = np.arange(values.shape[0])
    x_labels = [f"{(i + 1) * 6 / 24:.2f}" for i in x_ticks]
    plt.xticks(x_ticks[::4], x_labels[::4], rotation=0)
    plt.yticks(np.arange(len(var_names)), var_names)
    plt.xlabel("Lead time (days)")
    plt.ylabel("Variable")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_selected_curves(
    rmse: np.ndarray,
    acc: np.ndarray,
    var_names: Sequence[str],
    selected_indices: Sequence[int],
    out_dir: Path,
) -> None:
    lead_days = np.arange(1, rmse.shape[0] + 1) * 6 / 24.0

    plt.figure(figsize=(10, 6))
    for i in selected_indices:
        plt.plot(lead_days, rmse[:, i], label=var_names[i], linewidth=1.8)
    plt.xlabel("Lead time (days)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs lead time (selected variables)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_rmse_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in selected_indices:
        plt.plot(lead_days, acc[:, i], label=var_names[i], linewidth=1.8)
    plt.xlabel("Lead time (days)")
    plt.ylabel("ACC")
    plt.title("ACC vs lead time (selected variables)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_acc_curves.png", dpi=180)
    plt.close()


def resolve_spec_from_checkpoint(args: argparse.Namespace) -> Tuple[DataSpec, Dict]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg_file = json.load(f)
        cfg.update(cfg_file)

    def pick(name: str, default):
        val = getattr(args, name)
        if val is not None:
            return val
        return cfg.get(name, default)

    spec = DataSpec(
        zarr_store=pick("zarr_store", DEFAULT_ZARR_STORE),
        train_start=pick("train_start", "1979-01-01"),
        train_end=pick("train_end", "2018-12-31"),
        val_start=pick("val_start", "2019-01-01"),
        val_end=pick("val_end", "2020-12-31"),
        test_start=pick("test_start", "2021-01-01"),
        test_end=pick("test_end", "2022-12-31"),
        pressure_vars=pick("pressure_vars", list(DEFAULT_PRESSURE_VARS)),
        surface_vars=pick("surface_vars", list(DEFAULT_SURFACE_VARS)),
        pressure_levels=pick("pressure_levels", list(DEFAULT_PRESSURE_LEVELS)),
        history_steps=int(pick("history_steps", 2)),
        embed_dim=int(cfg.get("embed_dim", 1536)),
        num_heads=int(cfg.get("num_heads", 8)),
        window_size=int(cfg.get("window_size", 8)),
        depth_pre=int(cfg.get("depth_pre", 2)),
        depth_mid=int(cfg.get("depth_mid", 44)),
        depth_post=int(cfg.get("depth_post", 2)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.2)),
    )
    return spec, ckpt


def main() -> None:
    args = build_parser().parse_args()

    spec, ckpt = resolve_spec_from_checkpoint(args)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    pressure_vars, surface_vars, notes = resolve_variable_names(
        spec.zarr_store,
        spec.pressure_vars,
        spec.surface_vars,
    )

    print("=" * 96)
    print("FuXi Checkpoint Evaluation")
    print("=" * 96)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Zarr             : {spec.zarr_store}")
    print(f"Device           : {device}")
    print(f"AMP              : {args.amp}")
    print(f"Rollout steps    : {args.rollout_steps} ({args.rollout_steps * 6 / 24:.1f} days)")
    print(f"Eval split       : {spec.test_start} -> {spec.test_end}")
    print(f"Climo period     : {args.climo_start or spec.train_start} -> {args.climo_end or spec.train_end}")
    if notes:
        print("Variable mapping notes:")
        for n in notes:
            print(f"  - {n}")
    print("=" * 96)

    accessor = WB2Accessor(
        zarr_path=spec.zarr_store,
        pressure_vars=pressure_vars,
        surface_vars=surface_vars,
        pressure_levels=spec.pressure_levels,
    )

    train_idx = accessor.time_indices_between(spec.train_start, spec.train_end)
    eval_idx = accessor.time_indices_between(spec.test_start, spec.test_end)
    climo_start = args.climo_start or spec.train_start
    climo_end = args.climo_end or spec.train_end
    climo_idx = accessor.time_indices_between(climo_start, climo_end)

    mean, std = compute_channel_stats(accessor, train_idx, stats_samples=args.stats_samples)

    if args.climo_cache is not None:
        climo_cache = Path(args.climo_cache)
    else:
        stem = f"climo_{climo_start}_{climo_end}".replace(":", "_")
        climo_cache = Path(args.checkpoint).parent / f"{stem}.npz"
    climatology_np = compute_climatology(accessor, climo_idx, climo_cache)
    climatology = torch.from_numpy(climatology_np)

    dataset = RolloutDataset(
        accessor=accessor,
        eval_time_indices=eval_idx,
        rollout_steps=args.rollout_steps,
        mean=mean,
        std=std,
        history_steps=spec.history_steps,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(
        spec=spec,
        channels=accessor.channels,
        spatial_shape=accessor.spatial_shape,
        checkpoint_state=ckpt["model_state"],
    ).to(device)

    lat_w = build_lat_weights(accessor.latitudes, device)
    rmse, acc = evaluate_rollout(
        model=model,
        loader=loader,
        mean=mean,
        std=std,
        climatology=climatology,
        lat_w=lat_w,
        rollout_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
    )

    out_dir = Path(args.output_dir) if args.output_dir else (Path(args.checkpoint).parent / "evaluation_rollout")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metric_csvs(out_dir, accessor.var_names, rmse, acc)

    plot_heatmap(
        values=rmse,
        var_names=accessor.var_names,
        title="Area-weighted RMSE by variable and lead time",
        cmap="viridis",
        out_path=out_dir / "rmse_heatmap.png",
    )
    plot_heatmap(
        values=acc,
        var_names=accessor.var_names,
        title="ACC by variable and lead time",
        cmap="coolwarm",
        out_path=out_dir / "acc_heatmap.png",
        vmin=0.0,
        vmax=1.0,
    )

    if args.plot_vars is not None and len(args.plot_vars) > 0:
        selected = [i for i, n in enumerate(accessor.var_names) if n in set(args.plot_vars)]
    else:
        # Default selected variables for compact lead-time plots.
        default_names = [
            "geopotential_plev500",
            "temperature_plev850",
            "u_component_of_wind_plev850",
            "v_component_of_wind_plev850",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ]
        selected = [i for i, n in enumerate(accessor.var_names) if n in default_names]
        if not selected:
            selected = list(range(min(8, accessor.channels)))

    plot_selected_curves(
        rmse=rmse,
        acc=acc,
        var_names=accessor.var_names,
        selected_indices=selected,
        out_dir=out_dir,
    )

    save_prediction_samples(
        model=model,
        dataset=dataset,
        mean=mean,
        std=std,
        device=device,
        amp=args.amp,
        output_dir=out_dir,
        n_samples=args.save_n_samples,
        var_names=accessor.var_names,
    )

    lead_days = np.arange(1, args.rollout_steps + 1) * 6 / 24.0
    summary = {
        "checkpoint": args.checkpoint,
        "eval_samples": len(dataset),
        "rollout_steps": args.rollout_steps,
        "rollout_days": float(args.rollout_steps * 6 / 24.0),
        "variables": accessor.var_names,
        "best_acc_overall": float(np.max(acc)),
        "mean_acc_overall": float(np.mean(acc)),
        "mean_rmse_overall": float(np.mean(rmse)),
        "acc_lead_day_5_mean": float(np.mean(acc[int(5 * 4) - 1, :] if args.rollout_steps >= 20 else acc[-1, :])),
        "rmse_lead_day_5_mean": float(np.mean(rmse[int(5 * 4) - 1, :] if args.rollout_steps >= 20 else rmse[-1, :])),
        "acc_lead_day_10_mean": float(np.mean(acc[int(10 * 4) - 1, :] if args.rollout_steps >= 40 else acc[-1, :])),
        "rmse_lead_day_10_mean": float(np.mean(rmse[int(10 * 4) - 1, :] if args.rollout_steps >= 40 else rmse[-1, :])),
        "acc_lead_day_15_mean": float(np.mean(acc[int(15 * 4) - 1, :] if args.rollout_steps >= 60 else acc[-1, :])),
        "rmse_lead_day_15_mean": float(np.mean(rmse[int(15 * 4) - 1, :] if args.rollout_steps >= 60 else rmse[-1, :])),
        "lead_days": lead_days.tolist(),
        "output_dir": str(out_dir),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 96)
    print("Evaluation complete")
    print(f"Output directory : {out_dir}")
    print(f"Mean RMSE        : {summary['mean_rmse_overall']:.6f}")
    print(f"Mean ACC         : {summary['mean_acc_overall']:.6f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
