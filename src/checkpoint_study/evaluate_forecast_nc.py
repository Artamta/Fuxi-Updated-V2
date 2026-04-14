#!/usr/bin/env python3
"""
Evaluate forecast NetCDF files with RMSE, latitude-weighted RMSE, and ACC.

Expected forecast file format:
- dims: init_time, lead_step, channel, lat, lon
- data vars: forecast, truth
- coord: channel_name
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

try:
    from .common import (
        DEFAULT_CLIMATOLOGY_STORE,
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        discover_forecasts,
        parse_csv_strings,
        parse_forecast_specs,
    )
except ImportError:
    from src.checkpoint_study.common import (
        DEFAULT_CLIMATOLOGY_STORE,
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        discover_forecasts,
        parse_csv_strings,
        parse_forecast_specs,
    )

try:
    from ..evaluation.evaluate_checkpoint import PRESSURE_VAR_ALIASES, SURFACE_VAR_ALIASES
except ImportError:
    from src.evaluation.evaluate_checkpoint import PRESSURE_VAR_ALIASES, SURFACE_VAR_ALIASES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate forecast NetCDFs using ACC, RMSE, and latitude-weighted RMSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--forecast-files",
        nargs="*",
        default=None,
        help="Forecast specs: name=/path/to/forecast.nc OR /path/to/forecast.nc. If omitted, auto-discover under results root.",
    )
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--climatology-store",
        type=str,
        default=str(DEFAULT_CLIMATOLOGY_STORE),
        help="Climatology zarr store with dayofyear/hour dimensions",
    )
    parser.add_argument(
        "--plot-vars",
        type=parse_csv_strings,
        default=None,
        help="Selected variables for lead-time curve figures",
    )
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


@dataclass(frozen=True)
class ChannelClimoRef:
    variable: str
    level_index: Optional[int]


class HourlyClimatologyAccessor:
    def __init__(
        self,
        climo_store: str,
        channel_names: Sequence[str],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ):
        path = Path(climo_store).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Climatology store not found: {path}")

        self.store = zarr.open_group(str(path), mode="r")
        self.latitudes = latitudes.astype(np.float32)
        self.longitudes = longitudes.astype(np.float32)

        self.day_values = np.asarray(self.store["dayofyear"][:]).astype(np.int64)
        self.hour_values = np.asarray(self.store["hour"][:]).astype(np.int64)
        self.day_to_idx = {int(v): i for i, v in enumerate(self.day_values.tolist())}

        self.level_values = None
        self.level_to_idx: Dict[int, int] = {}
        if "level" in self.store:
            self.level_values = np.asarray(self.store["level"][:]).astype(np.int64)
            self.level_to_idx = {int(v): i for i, v in enumerate(self.level_values.tolist())}

        self.lat_name = "latitude" if "latitude" in self.store else "lat"
        self.lon_name = "longitude" if "longitude" in self.store else "lon"
        self.store_lat = np.asarray(self.store[self.lat_name][:]).astype(np.float32)
        self.store_lon = np.asarray(self.store[self.lon_name][:]).astype(np.float32)

        if self.store_lat.shape[0] != self.latitudes.shape[0] or self.store_lon.shape[0] != self.longitudes.shape[0]:
            raise ValueError(
                "Climatology and forecast grids are incompatible: "
                f"climo=({self.store_lat.shape[0]},{self.store_lon.shape[0]}) vs "
                f"forecast=({self.latitudes.shape[0]},{self.longitudes.shape[0]})"
            )

        self.channel_refs = [self._parse_channel(name) for name in channel_names]
        self.transpose_by_var: Dict[str, bool] = {}
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _resolve_var(self, raw_name: str) -> str:
        key = raw_name.strip()
        if key in self.store:
            return key

        key_lower = key.lower()
        if key_lower in PRESSURE_VAR_ALIASES:
            mapped = PRESSURE_VAR_ALIASES[key_lower]
            if mapped in self.store:
                return mapped
        if key_lower in SURFACE_VAR_ALIASES:
            mapped = SURFACE_VAR_ALIASES[key_lower]
            if mapped in self.store:
                return mapped

        if key_lower in ("mslp", "mean_sea_level_pressure") and "surface_pressure" in self.store:
            return "surface_pressure"

        raise KeyError(f"Variable '{raw_name}' was not found in climatology store")

    def _parse_channel(self, channel_name: str) -> ChannelClimoRef:
        name = str(channel_name)
        if "_plev" in name:
            base, level_txt = name.rsplit("_plev", 1)
            level = int(level_txt)
            var_name = self._resolve_var(base)
            if level not in self.level_to_idx:
                raise ValueError(f"Pressure level {level} missing in climatology levels={sorted(self.level_to_idx.keys())}")
            return ChannelClimoRef(variable=var_name, level_index=self.level_to_idx[level])

        var_name = self._resolve_var(name)
        return ChannelClimoRef(variable=var_name, level_index=None)

    def _hour_index(self, hour: int) -> int:
        # Nearest available hour in climatology set (typically 0,6,12,18).
        deltas = np.abs(self.hour_values - int(hour))
        return int(np.argmin(deltas))

    def _day_index(self, dayofyear: int) -> int:
        day = int(dayofyear)
        if day in self.day_to_idx:
            return self.day_to_idx[day]
        deltas = np.abs(self.day_values - day)
        return int(np.argmin(deltas))

    def _needs_transpose(self, variable: str, level_index: Optional[int]) -> bool:
        if variable in self.transpose_by_var:
            return self.transpose_by_var[variable]

        arr = self.store[variable]
        if level_index is None:
            sample = np.asarray(arr[0, 0, :, :])
        else:
            sample = np.asarray(arr[0, 0, level_index, :, :])

        lat_n = self.latitudes.shape[0]
        lon_n = self.longitudes.shape[0]
        if sample.shape[0] == lat_n and sample.shape[1] == lon_n:
            self.transpose_by_var[variable] = False
        elif sample.shape[0] == lon_n and sample.shape[1] == lat_n:
            self.transpose_by_var[variable] = True
        else:
            raise ValueError(
                f"Cannot infer spatial order for variable={variable}, sample_shape={sample.shape}, "
                f"expected ({lat_n},{lon_n}) or ({lon_n},{lat_n})"
            )
        return self.transpose_by_var[variable]

    def get_channel_stack(self, dayofyear: int, hour: int) -> np.ndarray:
        key = (int(dayofyear), int(hour))
        if key in self.cache:
            return self.cache[key]

        di = self._day_index(dayofyear)
        hi = self._hour_index(hour)

        out = np.empty((len(self.channel_refs), self.latitudes.shape[0], self.longitudes.shape[0]), dtype=np.float32)
        for ci, ref in enumerate(self.channel_refs):
            arr = self.store[ref.variable]
            if ref.level_index is None:
                field = np.asarray(arr[hi, di, :, :], dtype=np.float32)
            else:
                field = np.asarray(arr[hi, di, ref.level_index, :, :], dtype=np.float32)

            if self._needs_transpose(ref.variable, ref.level_index):
                field = np.swapaxes(field, -2, -1)
            out[ci] = field

        if len(self.cache) >= 1024:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out


def build_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(latitudes.astype(np.float64)))
    w = w / max(float(np.mean(w)), 1e-8)
    return w.reshape(1, -1, 1)


def _to_datetime(value) -> np.datetime64:
    return np.datetime64(str(value)).astype("datetime64[ns]")


def open_dataset_with_fallback(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception as exc:
        print(f"[warn] netcdf4 engine unavailable ({exc}); falling back to default xarray engine")
        return xr.open_dataset(path)


def evaluate_one_file(
    forecast_path: Path,
    climo_accessor: HourlyClimatologyAccessor,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = open_dataset_with_fallback(forecast_path)
    try:
        forecast = ds["forecast"].values.astype(np.float64)
        truth = ds["truth"].values.astype(np.float64)
        if forecast.shape != truth.shape:
            raise ValueError(f"forecast/truth shape mismatch: {forecast.shape} vs {truth.shape}")

        channel_names = [str(v) for v in ds["channel_name"].values.tolist()]
        lead_steps = ds["lead_step"].values.astype(np.int64)
        if "lead_hour" in ds:
            lead_hours = ds["lead_hour"].values.astype(np.int64)
        else:
            lead_hours = lead_steps * 6
        init_times = np.asarray([_to_datetime(v) for v in ds["init_time"].values])
        latitudes = ds["lat"].values.astype(np.float32)
    finally:
        ds.close()

    n_init, n_steps, n_channels, _, _ = forecast.shape
    lat_w = build_latitude_weights(latitudes)

    rmse_sum = np.zeros((n_steps, n_channels), dtype=np.float64)
    wrmse_sum = np.zeros((n_steps, n_channels), dtype=np.float64)
    acc_sum = np.zeros((n_steps, n_channels), dtype=np.float64)
    count = np.zeros((n_steps,), dtype=np.float64)

    for i in range(n_init):
        init_time = init_times[i]
        for s in range(n_steps):
            valid_time = init_time + np.timedelta64(int(lead_hours[s]), "h")
            ts = pd.Timestamp(valid_time)
            clim = climo_accessor.get_channel_stack(dayofyear=int(ts.dayofyear), hour=int(ts.hour)).astype(np.float64)

            pred = forecast[i, s]
            tgt = truth[i, s]

            err = pred - tgt
            mse = np.mean(err * err, axis=(1, 2))
            wmse = np.mean(err * err * lat_w, axis=(1, 2))

            rmse = np.sqrt(np.maximum(mse, 1e-12))
            wrmse = np.sqrt(np.maximum(wmse, 1e-12))

            pred_anom = pred - clim
            tgt_anom = tgt - clim
            num = np.sum(pred_anom * tgt_anom * lat_w, axis=(1, 2))
            den = np.sqrt(
                np.sum(pred_anom * pred_anom * lat_w, axis=(1, 2))
                * np.sum(tgt_anom * tgt_anom * lat_w, axis=(1, 2))
                + 1e-12
            )
            acc = np.clip(num / den, -1.0, 1.0)

            rmse_sum[s] += rmse
            wrmse_sum[s] += wrmse
            acc_sum[s] += acc
            count[s] += 1.0

        if i == 0 or (i + 1) % 8 == 0 or (i + 1) == n_init:
            step1 = 0
            denom = max(count[step1], 1.0)
            print(
                f"[eval] init {i+1}/{n_init} | "
                f"step1_rmse={float(rmse_sum[step1].mean()/denom):.4f} | "
                f"step1_wrmse={float(wrmse_sum[step1].mean()/denom):.4f} | "
                f"step1_acc={float(acc_sum[step1].mean()/denom):.4f}",
                flush=True,
            )

    denom = np.clip(count[:, None], 1.0, None)
    rmse_mean = rmse_sum / denom
    wrmse_mean = wrmse_sum / denom
    acc_mean = acc_sum / denom

    return channel_names, lead_steps, lead_hours, rmse_mean, wrmse_mean, acc_mean


def write_metric_csvs(
    out_dir: Path,
    channel_names: Sequence[str],
    lead_steps: np.ndarray,
    lead_hours: np.ndarray,
    rmse: np.ndarray,
    wrmse: np.ndarray,
    acc: np.ndarray,
) -> None:
    lead_days = lead_hours.astype(np.float64) / 24.0

    per_lead = out_dir / "metrics_per_lead.csv"
    with per_lead.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "lead_step", "lead_hour", "lead_day", "rmse", "rmse_lat_weighted", "acc"])
        for vi, var in enumerate(channel_names):
            for si in range(lead_steps.shape[0]):
                writer.writerow(
                    [
                        var,
                        int(lead_steps[si]),
                        int(lead_hours[si]),
                        float(lead_days[si]),
                        float(rmse[si, vi]),
                        float(wrmse[si, vi]),
                        float(acc[si, vi]),
                    ]
                )

    # 4 lead-steps/day for 6-hourly data.
    per_day = out_dir / "metrics_per_day.csv"
    n_days = int(lead_steps.shape[0] // 4)
    with per_day.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "day", "rmse_mean", "rmse_lat_weighted_mean", "acc_mean"])
        for vi, var in enumerate(channel_names):
            for d in range(n_days):
                s0 = d * 4
                s1 = s0 + 4
                writer.writerow(
                    [
                        var,
                        d + 1,
                        float(np.mean(rmse[s0:s1, vi])),
                        float(np.mean(wrmse[s0:s1, vi])),
                        float(np.mean(acc[s0:s1, vi])),
                    ]
                )


def plot_heatmap(values: np.ndarray, var_names: Sequence[str], title: str, cmap: str, out_path: Path, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(14, max(6, len(var_names) * 0.35)))
    im = plt.imshow(values.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    xticks = np.arange(values.shape[0])
    xlabels = [f"{(i + 1) * 6 / 24:.2f}" for i in xticks]
    plt.xticks(xticks[::4], xlabels[::4], rotation=0)
    plt.yticks(np.arange(len(var_names)), var_names)
    plt.xlabel("Lead time (days)")
    plt.ylabel("Variable")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_selected_curves(
    values: np.ndarray,
    var_names: Sequence[str],
    selected_indices: Sequence[int],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    lead_days = np.arange(1, values.shape[0] + 1) * 6 / 24.0
    plt.figure(figsize=(10, 6))
    for idx in selected_indices:
        plt.plot(lead_days, values[:, idx], label=var_names[idx], linewidth=1.8)
    plt.xlabel("Lead time (days)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def select_plot_indices(var_names: Sequence[str], requested: Optional[Sequence[str]]) -> List[int]:
    if requested is not None and len(requested) > 0:
        req = set(requested)
        selected = [i for i, n in enumerate(var_names) if n in req]
        if selected:
            return selected

    defaults = [
        "geopotential_plev500",
        "temperature_plev850",
        "u_component_of_wind_plev850",
        "v_component_of_wind_plev850",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
    ]
    selected = [i for i, n in enumerate(var_names) if n in defaults]
    if not selected:
        selected = list(range(min(8, len(var_names))))
    return selected


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()

    if args.forecast_files is not None and len(args.forecast_files) > 0:
        forecasts = parse_forecast_specs(args.forecast_files)
    else:
        forecasts = discover_forecasts(results_root)
    if not forecasts:
        raise RuntimeError(
            "No forecast files found. Provide --forecast-files or generate forecasts under results_new first."
        )

    print("=" * 96)
    print("Evaluate Forecast NetCDF")
    print("=" * 96)
    print(f"Forecast files    : {len(forecasts)}")
    print(f"Results root      : {results_root}")
    print(f"Climatology store : {Path(args.climatology_store).expanduser().resolve()}")
    print("=" * 96)

    for fc in forecasts:
        checkpoint_dir, _forecast_dir, metrics_dir = build_checkpoint_dirs(results_root, fc.name)
        summary_path = metrics_dir / "summary.json"

        if summary_path.exists() and not args.overwrite:
            print(f"[skip] {fc.name}: {summary_path} exists. Use --overwrite to recompute.")
            continue

        print(f"\n[run] {fc.name} -> {fc.path}")

        ds_meta = open_dataset_with_fallback(fc.path)
        try:
            channel_names = [str(v) for v in ds_meta["channel_name"].values.tolist()]
            latitudes = ds_meta["lat"].values.astype(np.float32)
            longitudes = ds_meta["lon"].values.astype(np.float32)
        finally:
            ds_meta.close()

        climo_accessor = HourlyClimatologyAccessor(
            climo_store=args.climatology_store,
            channel_names=channel_names,
            latitudes=latitudes,
            longitudes=longitudes,
        )

        channel_names, lead_steps, lead_hours, rmse, wrmse, acc = evaluate_one_file(fc.path, climo_accessor)

        metrics_dir.mkdir(parents=True, exist_ok=True)
        write_metric_csvs(
            out_dir=metrics_dir,
            channel_names=channel_names,
            lead_steps=lead_steps,
            lead_hours=lead_hours,
            rmse=rmse,
            wrmse=wrmse,
            acc=acc,
        )

        if not args.no_heatmaps:
            plot_heatmap(
                values=rmse,
                var_names=channel_names,
                title="RMSE by variable and lead time",
                cmap="viridis",
                out_path=metrics_dir / "rmse_heatmap.png",
            )
            plot_heatmap(
                values=wrmse,
                var_names=channel_names,
                title="Latitude-weighted RMSE by variable and lead time",
                cmap="magma",
                out_path=metrics_dir / "weighted_rmse_heatmap.png",
            )
            plot_heatmap(
                values=acc,
                var_names=channel_names,
                title="ACC by variable and lead time",
                cmap="coolwarm",
                out_path=metrics_dir / "acc_heatmap.png",
                vmin=0.0,
                vmax=1.0,
            )

        selected = select_plot_indices(channel_names, args.plot_vars)
        plot_selected_curves(
            values=rmse,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="RMSE",
            title="RMSE vs lead time (selected variables)",
            out_path=metrics_dir / "selected_rmse_curves.png",
        )
        plot_selected_curves(
            values=wrmse,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="Latitude-weighted RMSE",
            title="Latitude-weighted RMSE vs lead time (selected variables)",
            out_path=metrics_dir / "selected_weighted_rmse_curves.png",
        )
        plot_selected_curves(
            values=acc,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="ACC",
            title="ACC vs lead time (selected variables)",
            out_path=metrics_dir / "selected_acc_curves.png",
        )

        summary = {
            "forecast_file": str(fc.path),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_dir": str(metrics_dir),
            "variables": channel_names,
            "rollout_steps": int(lead_steps.shape[0]),
            "lead_hours": lead_hours.astype(int).tolist(),
            "mean_rmse": float(np.mean(rmse)),
            "mean_rmse_lat_weighted": float(np.mean(wrmse)),
            "mean_acc": float(np.mean(acc)),
            "best_acc": float(np.max(acc)),
            "rmse_day_5_mean": float(np.mean(rmse[19, :])) if lead_steps.shape[0] >= 20 else float(np.mean(rmse[-1, :])),
            "wrmse_day_5_mean": float(np.mean(wrmse[19, :])) if lead_steps.shape[0] >= 20 else float(np.mean(wrmse[-1, :])),
            "acc_day_5_mean": float(np.mean(acc[19, :])) if lead_steps.shape[0] >= 20 else float(np.mean(acc[-1, :])),
            "rmse_day_10_mean": float(np.mean(rmse[39, :])) if lead_steps.shape[0] >= 40 else float(np.mean(rmse[-1, :])),
            "wrmse_day_10_mean": float(np.mean(wrmse[39, :])) if lead_steps.shape[0] >= 40 else float(np.mean(wrmse[-1, :])),
            "acc_day_10_mean": float(np.mean(acc[39, :])) if lead_steps.shape[0] >= 40 else float(np.mean(acc[-1, :])),
            "rmse_day_15_mean": float(np.mean(rmse[59, :])) if lead_steps.shape[0] >= 60 else float(np.mean(rmse[-1, :])),
            "wrmse_day_15_mean": float(np.mean(wrmse[59, :])) if lead_steps.shape[0] >= 60 else float(np.mean(wrmse[-1, :])),
            "acc_day_15_mean": float(np.mean(acc[59, :])) if lead_steps.shape[0] >= 60 else float(np.mean(acc[-1, :])),
            "climatology_store": str(Path(args.climatology_store).expanduser().resolve()),
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"  mean_rmse={summary['mean_rmse']:.6f} | "
            f"mean_wrmse={summary['mean_rmse_lat_weighted']:.6f} | "
            f"mean_acc={summary['mean_acc']:.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
