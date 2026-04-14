#!/usr/bin/env python3
"""
Generate autoregressive FuXi forecasts from saved checkpoints and write NetCDF outputs.

For each checkpoint, this script writes:
- results_new/checkpoint_<name>/forecast/forecast.nc
- results_new/checkpoint_<name>/forecast/day1_forecast.png
- results_new/checkpoint_<name>/forecast/forecast_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr

try:
    from .common import (
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        parse_checkpoint_specs,
        parse_csv_ints,
        parse_csv_strings,
    )
except ImportError:
    from src.checkpoint_study.common import (
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        parse_checkpoint_specs,
        parse_csv_ints,
        parse_csv_strings,
    )

try:
    from ..evaluation.evaluate_checkpoint import (
        DEFAULT_PRESSURE_LEVELS,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        DEFAULT_ZARR_STORE,
        DataSpec,
        WB2Accessor,
        autocast_ctx,
        build_model,
        choose_device,
        compute_channel_stats,
        resolve_variable_names,
    )
except ImportError:
    from src.evaluation.evaluate_checkpoint import (
        DEFAULT_PRESSURE_LEVELS,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        DEFAULT_ZARR_STORE,
        DataSpec,
        WB2Accessor,
        autocast_ctx,
        build_model,
        choose_device,
        compute_channel_stats,
        resolve_variable_names,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate autoregressive forecast NetCDF files from one or more checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specs: name=/path/to/best.pt OR /path/to/best.pt",
    )
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--forecast-file-name", type=str, default="forecast.nc")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override")
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
    parser.add_argument("--history-steps", type=int, default=2)

    parser.add_argument(
        "--init-times",
        type=parse_csv_strings,
        default=None,
        help="Comma-separated init timestamps from test split. Example: 2021-01-01T00,2021-01-03T12",
    )
    parser.add_argument("--init-start", type=str, default=None)
    parser.add_argument("--init-end", type=str, default=None)
    parser.add_argument("--init-stride", type=int, default=1)
    parser.add_argument("--max-inits", type=int, default=1)

    parser.add_argument("--rollout-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stats-samples", type=int, default=256)

    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--day1-plot-vars",
        type=parse_csv_strings,
        default=None,
        help="Variables to include in day-1 plot",
    )
    parser.add_argument(
        "--max-output-gb",
        type=float,
        default=8.0,
        help="Safety limit for in-memory output arrays (forecast + truth)",
    )
    return parser


def _cfg_num(cfg: Dict, name: str, default, cast):
    val = cfg.get(name, default)
    if val is None:
        val = default
    return cast(val)


def resolve_spec_for_checkpoint(checkpoint_path: str, args: argparse.Namespace) -> Tuple[DataSpec, Dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}))

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg_override = json.load(f)
        cfg.update(cfg_override)

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
        embed_dim=_cfg_num(cfg, "embed_dim", 1536, int),
        num_heads=_cfg_num(cfg, "num_heads", 8, int),
        window_size=_cfg_num(cfg, "window_size", 8, int),
        depth_pre=_cfg_num(cfg, "depth_pre", 2, int),
        depth_mid=_cfg_num(cfg, "depth_mid", 44, int),
        depth_post=_cfg_num(cfg, "depth_post", 2, int),
        mlp_ratio=_cfg_num(cfg, "mlp_ratio", 4.0, float),
        drop_path_rate=_cfg_num(cfg, "drop_path_rate", 0.2, float),
    )
    return spec, ckpt


def _match_requested_times(init_times_ns: np.ndarray, requested: Sequence[str]) -> np.ndarray:
    mask = np.zeros(init_times_ns.shape[0], dtype=bool)
    for token in requested:
        t = token.strip()
        if not t:
            continue
        # Date-only token (YYYY-MM-DD) selects all init times in that UTC day.
        if "T" not in t and " " not in t:
            day = np.datetime64(t, "D")
            mask |= init_times_ns.astype("datetime64[D]") == day
        else:
            ts = np.datetime64(t).astype("datetime64[ns]")
            mask |= init_times_ns == ts
    return mask


def select_start_positions(
    accessor: WB2Accessor,
    eval_time_indices: np.ndarray,
    rollout_steps: int,
    history_steps: int,
    init_times: Optional[Sequence[str]],
    init_start: Optional[str],
    init_end: Optional[str],
    init_stride: int,
    max_inits: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    n_eval = int(eval_time_indices.shape[0])
    usable = n_eval - history_steps - rollout_steps + 1
    if usable <= 0:
        raise ValueError(
            f"Not enough test timesteps ({n_eval}) for history={history_steps}, rollout_steps={rollout_steps}"
        )

    starts = np.arange(usable, dtype=np.int64)
    init_positions = starts + history_steps - 1
    init_times_ns = accessor.time_values[eval_time_indices[init_positions]].astype("datetime64[ns]")

    if init_times is not None and len(init_times) > 0:
        mask = _match_requested_times(init_times_ns, init_times)
    else:
        mask = np.ones(starts.shape[0], dtype=bool)
        if init_start is not None:
            mask &= init_times_ns >= np.datetime64(init_start).astype("datetime64[ns]")
        if init_end is not None:
            mask &= init_times_ns <= np.datetime64(init_end).astype("datetime64[ns]")

    starts = starts[mask]
    init_times_ns = init_times_ns[mask]

    stride = max(1, int(init_stride))
    starts = starts[::stride]
    init_times_ns = init_times_ns[::stride]

    if max_inits is not None:
        starts = starts[: int(max_inits)]
        init_times_ns = init_times_ns[: int(max_inits)]

    if starts.size == 0:
        raise ValueError("No valid init times selected. Adjust --init-* options.")

    return starts, init_times_ns


class SelectedRolloutDataset(Dataset):
    def __init__(
        self,
        accessor: WB2Accessor,
        eval_time_indices: np.ndarray,
        start_positions: np.ndarray,
        rollout_steps: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        history_steps: int = 2,
    ):
        super().__init__()
        if history_steps != 2:
            raise ValueError("FuXi expects history_steps=2")

        self.accessor = accessor
        self.eval_time_indices = eval_time_indices
        self.start_positions = start_positions
        self.rollout_steps = int(rollout_steps)
        self.mean = mean
        self.std = std
        self.history_steps = int(history_steps)

    def __len__(self) -> int:
        return int(self.start_positions.shape[0])

    def __getitem__(self, idx: int):
        s = int(self.start_positions[idx])
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

        init_time = str(self.accessor.time_values[int(abs_idx[s + self.history_steps - 1])])
        return history, future, init_time


def estimate_output_gb(n_init: int, rollout_steps: int, channels: int, h: int, w: int) -> float:
    elements = n_init * rollout_steps * channels * h * w
    total_bytes = elements * 4 * 2  # float32 forecast + float32 truth
    return float(total_bytes) / (1024.0 ** 3)


@torch.no_grad()
def run_forecast_rollout(
    model,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    rollout_steps: int,
    device: torch.device,
    amp: str,
    channels: int,
    spatial_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(loader.dataset)
    h, w = spatial_shape
    forecast = np.empty((n_samples, rollout_steps, channels, h, w), dtype=np.float32)
    truth = np.empty_like(forecast)
    init_times = np.empty((n_samples,), dtype="datetime64[ns]")

    mean_d = mean.to(device)
    std_d = std.to(device)

    cursor = 0
    total_batches = len(loader)
    for batch_idx, (history, future, init_time_batch) in enumerate(loader, start=1):
        history = history.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        batch_size = int(history.shape[0])

        hist = history
        pred_steps = []
        for s in range(rollout_steps):
            with autocast_ctx(device, amp):
                pred_n = model(hist)
            pred_steps.append(pred_n)
            hist = torch.stack([hist[:, :, 1], pred_n], dim=2)

        pred_n = torch.stack(pred_steps, dim=1)  # (B,S,C,H,W)
        pred = (pred_n.float() * std_d + mean_d).cpu().numpy().astype(np.float32, copy=False)
        tgt = (future.float() * std_d + mean_d).cpu().numpy().astype(np.float32, copy=False)

        forecast[cursor : cursor + batch_size] = pred
        truth[cursor : cursor + batch_size] = tgt
        for j, init_time in enumerate(init_time_batch):
            init_times[cursor + j] = np.datetime64(str(init_time)).astype("datetime64[ns]")
        cursor += batch_size

        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            print(
                f"[forecast] batch {batch_idx}/{total_batches} | generated={cursor}/{n_samples}",
                flush=True,
            )

    return forecast, truth, init_times


def write_forecast_netcdf(
    out_path: Path,
    forecast: np.ndarray,
    truth: np.ndarray,
    init_times: np.ndarray,
    lead_steps: np.ndarray,
    var_names: Sequence[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    checkpoint_path: Path,
    spec: DataSpec,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "forecast": (("init_time", "lead_step", "channel", "lat", "lon"), forecast),
            "truth": (("init_time", "lead_step", "channel", "lat", "lon"), truth),
        },
        coords={
            "init_time": init_times,
            "lead_step": lead_steps.astype(np.int32),
            "lead_hour": ("lead_step", (lead_steps * 6).astype(np.int32)),
            "channel": np.arange(len(var_names), dtype=np.int32),
            "channel_name": ("channel", np.asarray(var_names, dtype=str)),
            "lat": latitudes.astype(np.float32),
            "lon": longitudes.astype(np.float32),
        },
        attrs={
            "checkpoint": str(checkpoint_path),
            "zarr_store": spec.zarr_store,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "history_steps": int(spec.history_steps),
            "rollout_steps": int(lead_steps.shape[0]),
            "lead_interval_hours": 6,
        },
    )

    encoding = {
        "forecast": {"zlib": True, "complevel": 1, "dtype": "float32"},
        "truth": {"zlib": True, "complevel": 1, "dtype": "float32"},
    }
    try:
        ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    except Exception as exc:
        print(f"[warn] netcdf4 engine unavailable ({exc}); writing without explicit engine/compression")
        ds.to_netcdf(out_path)
    ds.close()


def plot_day1_forecast(
    out_path: Path,
    forecast: np.ndarray,
    truth: np.ndarray,
    var_names: Sequence[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    requested_vars: Optional[Sequence[str]],
) -> None:
    if forecast.shape[0] == 0 or forecast.shape[1] == 0:
        return

    if requested_vars is not None and len(requested_vars) > 0:
        wanted = set(requested_vars)
        selected = [i for i, v in enumerate(var_names) if v in wanted]
    else:
        defaults = [
            "geopotential_plev500",
            "temperature_plev850",
            "2m_temperature",
        ]
        selected = [i for i, v in enumerate(var_names) if v in defaults]

    if not selected:
        selected = list(range(min(3, len(var_names))))

    nrows = len(selected)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(12.5, 3.7 * nrows), squeeze=False)

    lon_min, lon_max = float(np.min(longitudes)), float(np.max(longitudes))
    lat_min, lat_max = float(np.min(latitudes)), float(np.max(latitudes))
    extent = [lon_min, lon_max, lat_min, lat_max]

    for r, idx in enumerate(selected):
        fc = forecast[0, 0, idx]
        tr = truth[0, 0, idx]
        er = fc - tr
        err_lim = float(np.max(np.abs(er)))
        err_lim = max(err_lim, 1e-6)

        panels = [
            (fc, f"Forecast day1 - {var_names[idx]}", "viridis", None, None),
            (tr, f"Truth day1 - {var_names[idx]}", "viridis", None, None),
            (er, f"Error day1 - {var_names[idx]}", "RdBu_r", -err_lim, err_lim),
        ]

        for c, (arr, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = parse_checkpoint_specs(args.checkpoints)
    device = choose_device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    print("=" * 96)
    print("FuXi Forecast to NetCDF")
    print("=" * 96)
    print(f"Checkpoints   : {len(checkpoints)}")
    print(f"Results root  : {Path(args.results_root).expanduser().resolve()}")
    print(f"Device        : {device}")
    print(f"AMP           : {args.amp}")
    print(f"Rollout steps : {args.rollout_steps}")
    print("=" * 96)

    for ck in checkpoints:
        checkpoint_dir, forecast_dir, _metrics_dir = build_checkpoint_dirs(Path(args.results_root), ck.name)
        forecast_path = forecast_dir / args.forecast_file_name

        if forecast_path.exists() and not args.overwrite:
            print(f"[skip] {ck.name}: {forecast_path} already exists. Use --overwrite to regenerate.")
            continue

        print(f"\n[run] checkpoint={ck.name} path={ck.path}")

        spec, ckpt = resolve_spec_for_checkpoint(str(ck.path), args)
        if int(spec.history_steps) != 2:
            raise ValueError("This script currently supports history_steps=2 only.")

        pressure_vars, surface_vars, notes = resolve_variable_names(
            spec.zarr_store,
            spec.pressure_vars,
            spec.surface_vars,
        )
        for note in notes:
            print(f"  [var-map] {note}")

        accessor = WB2Accessor(
            zarr_path=spec.zarr_store,
            pressure_vars=pressure_vars,
            surface_vars=surface_vars,
            pressure_levels=spec.pressure_levels,
        )
        train_idx = accessor.time_indices_between(spec.train_start, spec.train_end)
        test_idx = accessor.time_indices_between(spec.test_start, spec.test_end)
        mean, std = compute_channel_stats(accessor, train_idx, stats_samples=args.stats_samples)

        starts, selected_init_times = select_start_positions(
            accessor=accessor,
            eval_time_indices=test_idx,
            rollout_steps=args.rollout_steps,
            history_steps=spec.history_steps,
            init_times=args.init_times,
            init_start=args.init_start,
            init_end=args.init_end,
            init_stride=args.init_stride,
            max_inits=args.max_inits,
        )

        est_gb = estimate_output_gb(
            n_init=int(starts.shape[0]),
            rollout_steps=int(args.rollout_steps),
            channels=int(accessor.channels),
            h=int(accessor.spatial_shape[0]),
            w=int(accessor.spatial_shape[1]),
        )
        print(f"  selected_inits={starts.shape[0]} | estimated_output={est_gb:.2f} GiB")
        if est_gb > float(args.max_output_gb):
            raise RuntimeError(
                f"Estimated output size {est_gb:.2f} GiB exceeds --max-output-gb={args.max_output_gb}. "
                "Lower --max-inits or --rollout-steps."
            )

        dataset = SelectedRolloutDataset(
            accessor=accessor,
            eval_time_indices=test_idx,
            start_positions=starts,
            rollout_steps=args.rollout_steps,
            mean=mean,
            std=std,
            history_steps=spec.history_steps,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        checkpoint_state = ckpt.get("model_state", ckpt)
        model = build_model(
            spec=spec,
            channels=accessor.channels,
            spatial_shape=accessor.spatial_shape,
            checkpoint_state=checkpoint_state,
        ).to(device)

        forecast, truth, init_times = run_forecast_rollout(
            model=model,
            loader=loader,
            mean=mean,
            std=std,
            rollout_steps=args.rollout_steps,
            device=device,
            amp=args.amp,
            channels=accessor.channels,
            spatial_shape=accessor.spatial_shape,
        )

        # Preserve selection order from input filters.
        if selected_init_times.shape[0] == init_times.shape[0]:
            init_times = selected_init_times

        lead_steps = np.arange(1, args.rollout_steps + 1, dtype=np.int32)
        write_forecast_netcdf(
            out_path=forecast_path,
            forecast=forecast,
            truth=truth,
            init_times=init_times,
            lead_steps=lead_steps,
            var_names=accessor.var_names,
            latitudes=accessor.latitudes,
            longitudes=accessor.longitudes,
            checkpoint_path=ck.path,
            spec=spec,
        )

        plot_day1_forecast(
            out_path=forecast_dir / "day1_forecast.png",
            forecast=forecast,
            truth=truth,
            var_names=accessor.var_names,
            latitudes=accessor.latitudes,
            longitudes=accessor.longitudes,
            requested_vars=args.day1_plot_vars,
        )

        metadata = {
            "checkpoint_name": ck.name,
            "checkpoint_path": str(ck.path),
            "forecast_file": str(forecast_path),
            "zarr_store": spec.zarr_store,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "n_init_times": int(init_times.shape[0]),
            "init_times": [
                np.datetime_as_string(t.astype("datetime64[s]"), unit="s")
                for t in init_times.astype("datetime64[ns]")
            ],
            "rollout_steps": int(args.rollout_steps),
            "lead_hours": (lead_steps * 6).tolist(),
            "channels": accessor.var_names,
            "spatial_shape": [int(accessor.spatial_shape[0]), int(accessor.spatial_shape[1])],
        }
        with (forecast_dir / "forecast_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  saved: {forecast_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
