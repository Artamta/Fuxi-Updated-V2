#!/usr/bin/env python3
"""
Evaluate forecast NetCDF files with latitude-weighted RMSE and ACC.

Expected forecast file format:
- dims: init_time, lead_step, channel, lat, lon
- data vars: forecast, truth
- coord: channel_name

Metric formulas implemented (matching FuXi paper equations):
RMSE(c, tau) = (1 / |D|) * sum_{t0 in D} sqrt((1 / (H * W)) * sum_{i,j} a_i * (Xhat - X)^2)
ACC(c, tau)  = (1 / |D|) * sum_{t0 in D}
              [sum_{i,j} a_i * (Xhat - M) * (X - M)] /
              sqrt(sum_{i,j} a_i * (Xhat - M)^2 * sum_{i,j} a_i * (X - M)^2)
where a_i = cos(lat_i) normalized to mean(a_i)=1.
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
        parse_csv_ints,
        parse_csv_strings,
        parse_forecast_specs,
    )
except ImportError:
    from src.checkpoint_study.common import (
        DEFAULT_CLIMATOLOGY_STORE,
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        discover_forecasts,
        parse_csv_ints,
        parse_csv_strings,
        parse_forecast_specs,
    )

try:
    from ..evaluation.evaluate_checkpoint import PRESSURE_VAR_ALIASES, SURFACE_VAR_ALIASES
except ImportError:
    from src.evaluation.evaluate_checkpoint import PRESSURE_VAR_ALIASES, SURFACE_VAR_ALIASES


SURFACE_NAMES = {
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "-",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate forecast NetCDFs using FuXi-style latitude-weighted RMSE and ACC.",
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
    parser.add_argument(
        "--horizon-days",
        type=parse_csv_ints,
        default=[5, 10, 15],
        help="Horizon windows (days) for summary metrics, e.g. 5,10,15",
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
                raise ValueError(
                    f"Pressure level {level} missing in climatology levels={sorted(self.level_to_idx.keys())}"
                )
            return ChannelClimoRef(variable=var_name, level_index=self.level_to_idx[level])

        var_name = self._resolve_var(name)
        return ChannelClimoRef(variable=var_name, level_index=None)

    def _hour_index(self, hour: int) -> int:
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
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    n_init, n_steps, n_channels, h, w = forecast.shape
    lat_w = build_latitude_weights(latitudes)

    rmse_weighted_sum = np.zeros((n_steps, n_channels), dtype=np.float64)
    rmse_unweighted_sum = np.zeros((n_steps, n_channels), dtype=np.float64)
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
            err_sq = err * err

            weighted_mse = np.sum(err_sq * lat_w, axis=(1, 2)) / float(h * w)
            unweighted_mse = np.mean(err_sq, axis=(1, 2))

            rmse_weighted = np.sqrt(np.maximum(weighted_mse, 1e-12))
            rmse_unweighted = np.sqrt(np.maximum(unweighted_mse, 1e-12))

            pred_anom = pred - clim
            tgt_anom = tgt - clim
            num = np.sum(pred_anom * tgt_anom * lat_w, axis=(1, 2))
            den = np.sqrt(
                np.sum(pred_anom * pred_anom * lat_w, axis=(1, 2))
                * np.sum(tgt_anom * tgt_anom * lat_w, axis=(1, 2))
                + 1e-12
            )
            acc = np.clip(num / den, -1.0, 1.0)

            rmse_weighted_sum[s] += rmse_weighted
            rmse_unweighted_sum[s] += rmse_unweighted
            acc_sum[s] += acc
            count[s] += 1.0

        if i == 0 or (i + 1) % 8 == 0 or (i + 1) == n_init:
            step1 = 0
            denom = max(count[step1], 1.0)
            print(
                f"[eval] init {i+1}/{n_init} | "
                f"step1_rmse={float(rmse_weighted_sum[step1].mean()/denom):.4f} | "
                f"step1_acc={float(acc_sum[step1].mean()/denom):.4f}",
                flush=True,
            )

    denom = np.clip(count[:, None], 1.0, None)
    rmse_weighted_mean = rmse_weighted_sum / denom
    rmse_unweighted_mean = rmse_unweighted_sum / denom
    acc_mean = acc_sum / denom

    return (
        channel_names,
        lead_steps,
        lead_hours,
        rmse_weighted_mean,
        rmse_unweighted_mean,
        acc_mean,
    )


def write_metric_csvs(
    out_dir: Path,
    channel_names: Sequence[str],
    lead_steps: np.ndarray,
    lead_hours: np.ndarray,
    rmse_weighted: np.ndarray,
    rmse_unweighted: np.ndarray,
    acc: np.ndarray,
) -> None:
    lead_days = lead_hours.astype(np.float64) / 24.0

    per_lead = out_dir / "metrics_per_lead.csv"
    with per_lead.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variable",
                "lead_step",
                "lead_hour",
                "lead_day",
                "rmse",
                "rmse_unweighted",
                "rmse_lat_weighted",
                "acc",
            ]
        )
        for vi, var in enumerate(channel_names):
            for si in range(lead_steps.shape[0]):
                writer.writerow(
                    [
                        var,
                        int(lead_steps[si]),
                        int(lead_hours[si]),
                        float(lead_days[si]),
                        float(rmse_weighted[si, vi]),
                        float(rmse_unweighted[si, vi]),
                        float(rmse_weighted[si, vi]),
                        float(acc[si, vi]),
                    ]
                )

    per_day = out_dir / "metrics_per_day.csv"
    n_days = int(lead_steps.shape[0] // 4)
    with per_day.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "day", "rmse_mean", "rmse_unweighted_mean", "acc_mean"])
        for vi, var in enumerate(channel_names):
            for d in range(n_days):
                s0 = d * 4
                s1 = s0 + 4
                writer.writerow(
                    [
                        var,
                        d + 1,
                        float(np.mean(rmse_weighted[s0:s1, vi])),
                        float(np.mean(rmse_unweighted[s0:s1, vi])),
                        float(np.mean(acc[s0:s1, vi])),
                    ]
                )


def write_mean_metric_csv(
    out_dir: Path,
    lead_steps: np.ndarray,
    lead_hours: np.ndarray,
    rmse_weighted: np.ndarray,
    rmse_unweighted: np.ndarray,
    acc: np.ndarray,
) -> None:
    per_lead = out_dir / "mean_metrics_per_lead.csv"
    lead_days = lead_hours.astype(np.float64) / 24.0

    rmse_mean = np.mean(rmse_weighted, axis=1)
    rmse_std = np.std(rmse_weighted, axis=1)
    rmse_unw_mean = np.mean(rmse_unweighted, axis=1)
    acc_mean = np.mean(acc, axis=1)
    acc_std = np.std(acc, axis=1)

    with per_lead.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lead_step",
                "lead_hour",
                "lead_day",
                "rmse_mean_over_variables",
                "rmse_std_over_variables",
                "rmse_unweighted_mean_over_variables",
                "acc_mean_over_variables",
                "acc_std_over_variables",
            ]
        )
        for i in range(lead_steps.shape[0]):
            writer.writerow(
                [
                    int(lead_steps[i]),
                    int(lead_hours[i]),
                    float(lead_days[i]),
                    float(rmse_mean[i]),
                    float(rmse_std[i]),
                    float(rmse_unw_mean[i]),
                    float(acc_mean[i]),
                    float(acc_std[i]),
                ]
            )


def write_horizon_summary(
    out_dir: Path,
    horizon_days: Sequence[int],
    rmse_weighted: np.ndarray,
    rmse_unweighted: np.ndarray,
    acc: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    rows = []
    summary: Dict[str, Dict[str, float]] = {}

    for day in sorted(set(int(d) for d in horizon_days if int(d) > 0)):
        max_step = min(rmse_weighted.shape[0], day * 4)
        if max_step <= 0:
            continue
        sl = slice(0, max_step)
        row = {
            "horizon_day": float(day),
            "max_lead_step": float(max_step),
            "rmse_mean": float(np.mean(rmse_weighted[sl, :])),
            "rmse_unweighted_mean": float(np.mean(rmse_unweighted[sl, :])),
            "acc_mean": float(np.mean(acc[sl, :])),
        }
        rows.append(row)
        summary[str(day)] = {
            "rmse_mean": row["rmse_mean"],
            "rmse_unweighted_mean": row["rmse_unweighted_mean"],
            "acc_mean": row["acc_mean"],
            "max_lead_step": row["max_lead_step"],
        }

    out_csv = out_dir / "horizon_window_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "horizon_day",
                "max_lead_step",
                "rmse_mean",
                "rmse_unweighted_mean",
                "acc_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary


def write_formula_note(out_dir: Path) -> None:
    text = []
    text.append("FuXi-style evaluation formulas used in this run")
    text.append("")
    text.append("Latitude weight:")
    text.append("  a_i = cos(lat_i) / mean_i(cos(lat_i))")
    text.append("")
    text.append("RMSE(c, tau):")
    text.append("  RMSE(c, tau) = (1/|D|) * sum_{t0 in D} sqrt((1/(H*W)) * sum_{i,j} a_i * (Xhat - X)^2)")
    text.append("")
    text.append("ACC(c, tau):")
    text.append("  ACC(c, tau) = (1/|D|) * sum_{t0 in D} [num / den]")
    text.append("  num = sum_{i,j} a_i * (Xhat - M) * (X - M)")
    text.append("  den = sqrt(sum_{i,j} a_i * (Xhat - M)^2 * sum_{i,j} a_i * (X - M)^2)")
    text.append("")
    text.append("Implementation detail:")
    text.append("  - M is time-dependent climatology map indexed by valid time (dayofyear, hour).")
    text.append("  - Mean over D is done after computing per-init-time RMSE/ACC for each lead.")
    (out_dir / "metrics_formula.txt").write_text("\n".join(text) + "\n")


def plot_heatmap(values: np.ndarray, var_names: Sequence[str], title: str, cmap: str, out_path: Path, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(15, max(6.2, len(var_names) * 0.35)))
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
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_overall_rollout(
    out_dir: Path,
    lead_days: np.ndarray,
    rmse_weighted: np.ndarray,
    rmse_unweighted: np.ndarray,
    acc: np.ndarray,
) -> None:
    rmse_mean = np.mean(rmse_weighted, axis=1)
    rmse_std = np.std(rmse_weighted, axis=1)
    rmse_unw_mean = np.mean(rmse_unweighted, axis=1)
    acc_mean = np.mean(acc, axis=1)
    acc_std = np.std(acc, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)

    axes[0].plot(lead_days, rmse_mean, color="#1f77b4", linewidth=2.4, label="RMSE mean")
    axes[0].fill_between(lead_days, rmse_mean - rmse_std, rmse_mean + rmse_std, color="#1f77b4", alpha=0.18)
    axes[0].set_title("Mean RMSE over variables")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)

    axes[1].plot(lead_days, rmse_unw_mean, color="#2ca02c", linewidth=2.3, label="RMSE unweighted mean")
    axes[1].set_title("Mean RMSE unweighted")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(frameon=False)

    axes[2].plot(lead_days, acc_mean, color="#d62728", linewidth=2.4, label="ACC mean")
    axes[2].fill_between(lead_days, acc_mean - acc_std, acc_mean + acc_std, color="#d62728", alpha=0.18)
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[2].set_title("Mean ACC over variables")
    axes[2].set_xlabel("Lead time (days)")
    axes[2].set_ylabel("ACC")
    axes[2].legend(frameon=False)

    fig.savefig(out_dir / "overall_mean_rollout_metrics.png", dpi=240)
    plt.close(fig)


def _valid_horizon_days(horizon_days: Sequence[int]) -> List[int]:
    return sorted(set(int(d) for d in horizon_days if int(d) > 0))


def _add_horizon_guides(ax: plt.Axes, horizon_days: Sequence[int]) -> None:
    for day in _valid_horizon_days(horizon_days):
        ax.axvline(float(day), color="#9e9e9e", linestyle="--", linewidth=0.9, alpha=0.55)


def plot_average_metrics(
    out_dir: Path,
    lead_days: np.ndarray,
    rmse_weighted: np.ndarray,
    rmse_unweighted: np.ndarray,
    acc: np.ndarray,
    horizon_days: Sequence[int],
) -> None:
    rmse_w_mean = np.mean(rmse_weighted, axis=1)

    rmse_u_mean = np.mean(rmse_unweighted, axis=1)

    acc_mean = np.mean(acc, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.0), constrained_layout=True)

    axes[0].plot(lead_days, rmse_w_mean, color="#1f77b4", linewidth=2.5, label="Weighted RMSE mean")
    axes[0].plot(
        lead_days,
        rmse_u_mean,
        color="#2ca02c",
        linewidth=2.2,
        linestyle="--",
        label="Unweighted RMSE mean",
    )
    _add_horizon_guides(axes[0], horizon_days)
    axes[0].set_title("Average RMSE over all variables")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)

    axes[1].plot(lead_days, acc_mean, color="#d62728", linewidth=2.5, label="ACC mean")
    _add_horizon_guides(axes[1], horizon_days)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("Average ACC over all variables")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(min(-0.1, float(np.nanmin(acc_mean) - 0.05)), 1.02)
    axes[1].legend(frameon=False)

    fig.savefig(out_dir / "average_rmse_acc_rollout.png", dpi=240)
    plt.close(fig)


def _is_surface_name(name: str) -> bool:
    return name in SURFACE_NAMES


def _split_variable_groups(var_names: Sequence[str]) -> Tuple[List[int], List[int]]:
    surface_idx = [i for i, n in enumerate(var_names) if _is_surface_name(n)]
    pressure_idx = [i for i, n in enumerate(var_names) if "_plev" in n]
    if not pressure_idx:
        pressure_idx = [i for i, n in enumerate(var_names) if i not in surface_idx]
    return surface_idx, pressure_idx


def plot_average_metrics_by_group(
    out_dir: Path,
    lead_days: np.ndarray,
    rmse_weighted: np.ndarray,
    acc: np.ndarray,
    var_names: Sequence[str],
    horizon_days: Sequence[int],
) -> None:
    surface_idx, pressure_idx = _split_variable_groups(var_names)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)

    axes[0].plot(lead_days, np.mean(rmse_weighted, axis=1), color="#111111", linewidth=2.4, label="All variables")
    if surface_idx:
        axes[0].plot(
            lead_days,
            np.mean(rmse_weighted[:, surface_idx], axis=1),
            color="#1f77b4",
            linewidth=2.2,
            label="Surface mean",
        )
    if pressure_idx:
        axes[0].plot(
            lead_days,
            np.mean(rmse_weighted[:, pressure_idx], axis=1),
            color="#ff7f0e",
            linewidth=2.2,
            label="Pressure mean",
        )
    _add_horizon_guides(axes[0], horizon_days)
    axes[0].set_title("Average latitude-weighted RMSE by group")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("Weighted RMSE")
    axes[0].legend(frameon=False)

    axes[1].plot(lead_days, np.mean(acc, axis=1), color="#111111", linewidth=2.4, label="All variables")
    if surface_idx:
        axes[1].plot(
            lead_days,
            np.mean(acc[:, surface_idx], axis=1),
            color="#1f77b4",
            linewidth=2.2,
            label="Surface mean",
        )
    if pressure_idx:
        axes[1].plot(
            lead_days,
            np.mean(acc[:, pressure_idx], axis=1),
            color="#ff7f0e",
            linewidth=2.2,
            label="Pressure mean",
        )
    _add_horizon_guides(axes[1], horizon_days)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("Average ACC by group")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(min(-0.1, float(np.nanmin(acc) - 0.05)), 1.02)
    axes[1].legend(frameon=False)

    fig.savefig(out_dir / "average_rmse_acc_by_group.png", dpi=240)
    plt.close(fig)


def _auto_scale_rmse_axis(
    ax: plt.Axes,
    values: np.ndarray,
    log_ratio_threshold: float = 40.0,
) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False

    positive = finite[finite > 0.0]
    if positive.size == 0:
        return False

    vmin = float(np.nanmin(positive))
    vmax = float(np.nanmax(positive))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0.0:
        return False

    ratio = vmax / max(vmin, 1e-12)
    if ratio >= float(log_ratio_threshold):
        ax.set_yscale("log")
        y0 = max(vmin * 0.8, 1e-8)
        y1 = vmax * 1.15
        if y0 < y1:
            ax.set_ylim(y0, y1)
        ax.text(
            0.99,
            0.04,
            "auto log-scale",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )
        return True

    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    if np.isfinite(ymin) and np.isfinite(ymax):
        if ymax > ymin:
            pad = 0.07 * (ymax - ymin)
        else:
            pad = 0.10 * (abs(ymin) + 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    return False


def plot_variable_group(
    out_path: Path,
    lead_days: np.ndarray,
    rmse_weighted: np.ndarray,
    acc: np.ndarray,
    var_names: Sequence[str],
    indices: Sequence[int],
    title_prefix: str,
    ncol: int,
) -> None:
    if not indices:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.2), constrained_layout=True)
    for idx in indices:
        axes[0].plot(lead_days, rmse_weighted[:, idx], linewidth=1.8, label=var_names[idx])
    used_log = _auto_scale_rmse_axis(axes[0], rmse_weighted[:, indices])
    if used_log:
        axes[0].set_title(f"{title_prefix}: RMSE (log y-scale)")
    else:
        axes[0].set_title(f"{title_prefix}: RMSE")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False, fontsize=8, ncol=ncol)

    for idx in indices:
        axes[1].plot(lead_days, acc[:, idx], linewidth=1.8, label=var_names[idx])
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_title(f"{title_prefix}: ACC")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].legend(frameon=False, fontsize=8, ncol=ncol)

    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_all_variable_metrics(
    out_path: Path,
    lead_days: np.ndarray,
    rmse_values: np.ndarray,
    acc: np.ndarray,
    var_names: Sequence[str],
    horizon_days: Sequence[int],
    rmse_label: str,
) -> None:
    if rmse_values.shape[1] != len(var_names) or acc.shape[1] != len(var_names):
        return

    fig, axes = plt.subplots(2, 1, figsize=(15.8, 10.2), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    for idx, name in enumerate(var_names):
        color = cmap(idx % 20)
        axes[0].plot(lead_days, rmse_values[:, idx], linewidth=1.45, alpha=0.95, color=color, label=name)
        axes[1].plot(lead_days, acc[:, idx], linewidth=1.45, alpha=0.95, color=color, label=name)

    used_log = _auto_scale_rmse_axis(axes[0], rmse_values)
    if used_log:
        axes[0].set_title(f"All variables: {rmse_label} (log y-scale)")
    else:
        axes[0].set_title(f"All variables: {rmse_label}")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel(rmse_label)
    _add_horizon_guides(axes[0], horizon_days)

    axes[1].set_title("All variables: ACC")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    _add_horizon_guides(axes[1], horizon_days)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axes[1].set_ylim(min(-0.1, float(np.nanmin(acc) - 0.05)), 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = 4 if len(labels) > 12 else 3
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol,
        frameon=False,
        fontsize=8,
    )

    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _choose_grid_cols(n_vars: int, max_cols: int = 5) -> int:
    if n_vars <= 1:
        return 1
    upper = min(max_cols, n_vars)
    best_cols = 1
    best_score = None
    for ncols in range(2, upper + 1):
        nrows = int(np.ceil(n_vars / ncols))
        empty = nrows * ncols - n_vars
        squareness = abs(nrows - ncols)
        score = (empty, squareness, -ncols)
        if best_score is None or score < best_score:
            best_score = score
            best_cols = ncols
    return best_cols


def plot_all_variable_rmse_overlay(
    out_path: Path,
    lead_days: np.ndarray,
    rmse_values: np.ndarray,
    var_names: Sequence[str],
    horizon_days: Sequence[int],
    title: str,
    ylabel: str = "RMSE",
) -> None:
    if rmse_values.shape[1] != len(var_names):
        return

    fig, ax = plt.subplots(1, 1, figsize=(15.5, 8.7), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    for idx, name in enumerate(var_names):
        color = cmap(idx % 20)
        ax.plot(lead_days, rmse_values[:, idx], linewidth=1.6, alpha=0.95, color=color, label=name)

    used_log = _auto_scale_rmse_axis(ax, rmse_values)
    if used_log:
        ax.set_title(f"{title} (log y-scale)")
    else:
        ax.set_title(title)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    _add_horizon_guides(ax, horizon_days)

    handles, labels = ax.get_legend_handles_labels()
    ncol = _choose_grid_cols(len(labels), max_cols=5)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        fontsize=8,
    )

    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_all_variable_rmse_panels(
    out_path: Path,
    lead_days: np.ndarray,
    rmse_values: np.ndarray,
    var_names: Sequence[str],
    title: str,
    ylabel: str = "RMSE",
) -> None:
    if rmse_values.shape[1] != len(var_names):
        return

    n_vars = len(var_names)
    ncols = _choose_grid_cols(n_vars, max_cols=5)
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n_vars:]:
        ax.axis("off")

    for idx, name in enumerate(var_names):
        ax = axes_flat[idx]
        y = rmse_values[:, idx]
        ax.plot(lead_days, y, color="#1f77b4", linewidth=1.9)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylabel)

        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymax > ymin:
                pad = 0.08 * (ymax - ymin)
            else:
                pad = 0.10 * (abs(ymin) + 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


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


def plot_selected_curves(
    values: np.ndarray,
    var_names: Sequence[str],
    selected_indices: Sequence[int],
    ylabel: str,
    title: str,
    out_path: Path,
    lead_days: Optional[np.ndarray] = None,
    auto_log_rmse: bool = False,
) -> None:
    if lead_days is None or lead_days.shape[0] != values.shape[0]:
        lead_days = np.arange(1, values.shape[0] + 1, dtype=np.float64) * 6.0 / 24.0

    plt.figure(figsize=(10.5, 6.0))
    for idx in selected_indices:
        plt.plot(lead_days, values[:, idx], label=var_names[idx], linewidth=2.0)

    used_log = False
    if auto_log_rmse and selected_indices:
        used_log = _auto_scale_rmse_axis(plt.gca(), values[:, selected_indices])

    plt.xlabel("Lead time (days)")
    plt.ylabel(ylabel)
    if used_log:
        plt.title(f"{title} (log y-scale)")
    else:
        plt.title(title)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_selected_small_multiples(
    values: np.ndarray,
    var_names: Sequence[str],
    selected_indices: Sequence[int],
    ylabel: str,
    title: str,
    out_path: Path,
    lead_days: np.ndarray,
) -> None:
    if not selected_indices:
        return

    n_vars = len(selected_indices)
    ncols = _choose_grid_cols(n_vars, max_cols=4)
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n_vars:]:
        ax.axis("off")

    for p, idx in enumerate(selected_indices):
        ax = axes_flat[p]
        y = values[:, idx]
        ax.plot(lead_days, y, color="#1f77b4", linewidth=2.0)
        ax.set_title(var_names[idx], fontsize=10)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylabel)

        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymax > ymin:
                pad = 0.08 * (ymax - ymin)
            else:
                pad = 0.10 * (abs(ymin) + 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    set_plot_style()

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

        (
            channel_names,
            lead_steps,
            lead_hours,
            rmse_weighted,
            rmse_unweighted,
            acc,
        ) = evaluate_one_file(fc.path, climo_accessor)

        lead_days = lead_hours.astype(np.float64) / 24.0
        metrics_dir.mkdir(parents=True, exist_ok=True)

        write_metric_csvs(
            out_dir=metrics_dir,
            channel_names=channel_names,
            lead_steps=lead_steps,
            lead_hours=lead_hours,
            rmse_weighted=rmse_weighted,
            rmse_unweighted=rmse_unweighted,
            acc=acc,
        )
        write_mean_metric_csv(
            out_dir=metrics_dir,
            lead_steps=lead_steps,
            lead_hours=lead_hours,
            rmse_weighted=rmse_weighted,
            rmse_unweighted=rmse_unweighted,
            acc=acc,
        )
        horizon_summary = write_horizon_summary(
            out_dir=metrics_dir,
            horizon_days=args.horizon_days,
            rmse_weighted=rmse_weighted,
            rmse_unweighted=rmse_unweighted,
            acc=acc,
        )
        write_formula_note(metrics_dir)

        if not args.no_heatmaps:
            plot_heatmap(
                values=rmse_weighted,
                var_names=channel_names,
                title="RMSE (latitude-weighted) by variable and lead time",
                cmap="viridis",
                out_path=metrics_dir / "rmse_heatmap.png",
            )
            plot_heatmap(
                values=rmse_unweighted,
                var_names=channel_names,
                title="RMSE (unweighted) by variable and lead time",
                cmap="plasma",
                out_path=metrics_dir / "rmse_unweighted_heatmap.png",
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

        plot_overall_rollout(
            out_dir=metrics_dir,
            lead_days=lead_days,
            rmse_weighted=rmse_weighted,
            rmse_unweighted=rmse_unweighted,
            acc=acc,
        )
        plot_average_metrics(
            out_dir=metrics_dir,
            lead_days=lead_days,
            rmse_weighted=rmse_weighted,
            rmse_unweighted=rmse_unweighted,
            acc=acc,
            horizon_days=args.horizon_days,
        )
        plot_average_metrics_by_group(
            out_dir=metrics_dir,
            lead_days=lead_days,
            rmse_weighted=rmse_weighted,
            acc=acc,
            var_names=channel_names,
            horizon_days=args.horizon_days,
        )
        plot_all_variable_metrics(
            out_path=metrics_dir / "all_variable_rmse_acc.png",
            lead_days=lead_days,
            rmse_values=rmse_unweighted,
            acc=acc,
            var_names=channel_names,
            horizon_days=args.horizon_days,
            rmse_label="RMSE",
        )
        plot_all_variable_metrics(
            out_path=metrics_dir / "per_variable_rmse_acc.png",
            lead_days=lead_days,
            rmse_values=rmse_unweighted,
            acc=acc,
            var_names=channel_names,
            horizon_days=args.horizon_days,
            rmse_label="RMSE",
        )
        plot_all_variable_metrics(
            out_path=metrics_dir / "all_variable_weighted_rmse_acc.png",
            lead_days=lead_days,
            rmse_values=rmse_weighted,
            acc=acc,
            var_names=channel_names,
            horizon_days=args.horizon_days,
            rmse_label="Latitude-weighted RMSE",
        )
        plot_all_variable_metrics(
            out_path=metrics_dir / "per_variable_weighted_rmse_acc.png",
            lead_days=lead_days,
            rmse_values=rmse_weighted,
            acc=acc,
            var_names=channel_names,
            horizon_days=args.horizon_days,
            rmse_label="Latitude-weighted RMSE",
        )
        plot_all_variable_rmse_overlay(
            out_path=metrics_dir / "poster_all20_rmse_overlay.png",
            lead_days=lead_days,
            rmse_values=rmse_unweighted,
            var_names=channel_names,
            horizon_days=args.horizon_days,
            title="RMSE (all 20 variables)",
            ylabel="RMSE",
        )
        plot_all_variable_rmse_panels(
            out_path=metrics_dir / "poster_all20_rmse_per_variable.png",
            lead_days=lead_days,
            rmse_values=rmse_unweighted,
            var_names=channel_names,
            title="RMSE (unweighted): all 20 variable panels",
            ylabel="RMSE",
        )

        surface_idx, pressure_idx = _split_variable_groups(channel_names)
        plot_variable_group(
            out_path=metrics_dir / "surface_variable_metrics.png",
            lead_days=lead_days,
            rmse_weighted=rmse_weighted,
            acc=acc,
            var_names=channel_names,
            indices=surface_idx,
            title_prefix="Surface variables",
            ncol=2,
        )
        plot_variable_group(
            out_path=metrics_dir / "pressure_15_variable_metrics.png",
            lead_days=lead_days,
            rmse_weighted=rmse_weighted,
            acc=acc,
            var_names=channel_names,
            indices=pressure_idx,
            title_prefix="Pressure variables (15 channels)",
            ncol=3,
        )

        selected = select_plot_indices(channel_names, args.plot_vars)
        plot_selected_curves(
            values=rmse_unweighted,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="RMSE",
            title="RMSE (unweighted) vs lead time (selected variables)",
            out_path=metrics_dir / "selected_rmse_curves.png",
            lead_days=lead_days,
            auto_log_rmse=True,
        )
        plot_selected_curves(
            values=rmse_weighted,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="Latitude-weighted RMSE",
            title="Latitude-weighted RMSE vs lead time (selected variables)",
            out_path=metrics_dir / "selected_weighted_rmse_curves.png",
            lead_days=lead_days,
            auto_log_rmse=True,
        )
        plot_selected_curves(
            values=acc,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="ACC",
            title="ACC vs lead time (selected variables)",
            out_path=metrics_dir / "selected_acc_curves.png",
            lead_days=lead_days,
        )

        plot_selected_small_multiples(
            values=rmse_unweighted,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="RMSE",
            title="RMSE (unweighted): per-variable panels",
            out_path=metrics_dir / "selected_rmse_small_multiples.png",
            lead_days=lead_days,
        )
        plot_selected_small_multiples(
            values=rmse_weighted,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="Latitude-weighted RMSE",
            title="Latitude-weighted RMSE: per-variable panels",
            out_path=metrics_dir / "selected_weighted_rmse_small_multiples.png",
            lead_days=lead_days,
        )
        plot_selected_small_multiples(
            values=acc,
            var_names=channel_names,
            selected_indices=selected,
            ylabel="ACC",
            title="ACC: per-variable panels",
            out_path=metrics_dir / "selected_acc_small_multiples.png",
            lead_days=lead_days,
        )

        summary = {
            "forecast_file": str(fc.path),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_dir": str(metrics_dir),
            "variables": channel_names,
            "rollout_steps": int(lead_steps.shape[0]),
            "lead_hours": lead_hours.astype(int).tolist(),
            "mean_rmse": float(np.mean(rmse_weighted)),
            "mean_rmse_unweighted": float(np.mean(rmse_unweighted)),
            "mean_acc": float(np.mean(acc)),
            "best_acc": float(np.max(acc)),
            "horizon_summary": horizon_summary,
            "climatology_store": str(Path(args.climatology_store).expanduser().resolve()),
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"  mean_rmse={summary['mean_rmse']:.6f} | "
            f"mean_rmse_unweighted={summary['mean_rmse_unweighted']:.6f} | "
            f"mean_acc={summary['mean_acc']:.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
