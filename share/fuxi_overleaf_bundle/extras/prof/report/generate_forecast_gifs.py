#!/usr/bin/env python3
"""
Generate report-ready forecast GIFs from checkpoint-study artifacts.

Produces six GIFs by default:
- t2m rollout to 5, 10, and 15 days
- richer-variable rollout to 5, 10, and 15 days

Each GIF includes:
- init date/time and valid date/time in frame titles
- truth / prediction / error maps
- ACC evolution panel from metrics_per_lead.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr  # type: ignore[import-not-found]
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required for GIF export. Install with: pip install Pillow"
    ) from exc


DEFAULT_RICH_VARS = [
    "geopotential_plev500",
    "total_column_water_vapour",
    "u_component_of_wind_plev850",
]

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class GifRecord:
    kind: str
    horizon_days: int
    init_index: int
    init_time: str
    output_file: str
    variables: List[str]
    frames: int


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _to_np_datetime(value) -> np.datetime64:
    return np.datetime64(str(value)).astype("datetime64[ns]")


def _pretty_dt(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[m]"), unit="m")


def _file_dt_tag(value: np.datetime64) -> str:
    text = _pretty_dt(value)
    return text.replace("-", "").replace(":", "").replace("T", "T")


def _open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return xr.open_dataset(path)


def _robust_limits(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    vmin = float(np.nanpercentile(arr, lo))
    vmax = float(np.nanpercentile(arr, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return -1.0, 1.0
    return vmin, vmax


def _figure_to_pil(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(rgba[:, :, :3])


def _draw_map_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    title: str,
    extent: Sequence[float],
    cmap: str,
    vmin: float,
    vmax: float,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude" if show_xlabel else "")
    ax.set_ylabel("Latitude" if show_ylabel else "")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2)


def _save_gif(frames: List[Image.Image], out_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {out_path}")
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 0.1))))
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _read_acc_lookup(metrics_csv: Path, n_steps: int) -> Dict[str, np.ndarray]:
    if not metrics_csv.is_file():
        return {}

    df = pd.read_csv(metrics_csv)
    required = {"variable", "lead_step", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metrics_csv} missing columns: {sorted(missing)}")

    out: Dict[str, np.ndarray] = {}
    for var_name, grp in df.groupby("variable"):
        arr = np.full((n_steps,), np.nan, dtype=np.float64)
        lead_steps = grp["lead_step"].astype(int).to_numpy()
        values = grp["acc"].astype(float).to_numpy()
        for step, val in zip(lead_steps, values):
            idx = int(step) - 1
            if 0 <= idx < n_steps:
                arr[idx] = float(val)
        out[str(var_name)] = arr
    return out


def _resolve_horizon_steps(lead_days: np.ndarray, horizon_day: int) -> int:
    max_step = int(np.searchsorted(lead_days, float(horizon_day), side="right"))
    return max(1, min(max_step, int(lead_days.shape[0])))


def _resolve_init_indices(raw_indices: Sequence[int], n_horizons: int, n_inits: int) -> List[int]:
    if not raw_indices:
        raw_indices = [0]

    if len(raw_indices) == 1 and n_horizons > 1:
        raw_indices = [int(raw_indices[0])] * n_horizons
    if len(raw_indices) < n_horizons:
        raise ValueError(
            f"Need at least {n_horizons} init indices for horizons, got {len(raw_indices)}"
        )

    out = [int(v) for v in raw_indices[:n_horizons]]
    bad = [idx for idx in out if idx < 0 or idx >= n_inits]
    if bad:
        raise ValueError(f"Init indices out of range for n_init={n_inits}: {bad}")
    return out


def _read_var_block(
    ds: xr.Dataset,
    init_idx: int,
    step_count: int,
    channel_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = ds["forecast"].isel(init_time=init_idx, lead_step=slice(0, step_count), channel=channel_idx).values
    truth = ds["truth"].isel(init_time=init_idx, lead_step=slice(0, step_count), channel=channel_idx).values
    return pred.astype(np.float32), truth.astype(np.float32)


def _build_t2m_frames(
    pred: np.ndarray,
    truth: np.ndarray,
    init_time: np.datetime64,
    lead_hours: np.ndarray,
    lead_days: np.ndarray,
    horizon_day: int,
    acc_series: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    fps: float,
    frame_step: int,
    variable_name: str,
) -> List[Image.Image]:
    del fps

    n_steps = pred.shape[0]
    indices = list(range(0, n_steps, max(1, int(frame_step))))
    if indices[-1] != (n_steps - 1):
        indices.append(n_steps - 1)

    all_vals = np.concatenate([pred.reshape(-1), truth.reshape(-1)])
    val_min, val_max = _robust_limits(all_vals, lo=2.0, hi=98.0)
    err_lim = float(np.nanpercentile(np.abs(pred - truth), 99.0))
    err_lim = max(err_lim, 1e-6)

    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    extent = [lon_min, lon_max, lat_min, lat_max]

    frames: List[Image.Image] = []
    for step_idx in indices:
        p = pred[step_idx]
        t = truth[step_idx]
        e = p - t

        valid_time = init_time + np.timedelta64(int(lead_hours[step_idx]), "h")
        current_acc = float(acc_series[step_idx]) if step_idx < acc_series.shape[0] else float("nan")
        running_acc = float(np.nanmean(acc_series[: step_idx + 1]))

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)

        panels = [
            (t, "Target", "coolwarm", val_min, val_max),
            (p, "Prediction", "coolwarm", val_min, val_max),
            (e, "Error (pred - target)", "RdBu_r", -err_lim, err_lim),
        ]

        for i, (ax, (arr, title, cmap, vmin, vmax)) in enumerate(
            zip([axes[0, 0], axes[0, 1], axes[1, 0]], panels)
        ):
            _draw_map_panel(
                fig,
                ax,
                arr,
                title=f"{variable_name} | {title}",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                show_xlabel=(i == 2),
                show_ylabel=(i in (0, 2)),
            )

        ax_acc = axes[1, 1]
        x = lead_days[:n_steps]
        y = acc_series[:n_steps]
        ax_acc.plot(x, y, color="#d62728", linewidth=2.3, label="ACC")
        ax_acc.scatter([lead_days[step_idx]], [y[step_idx]], s=40, color="black", zorder=5)
        ax_acc.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax_acc.axvline(float(horizon_day), color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_acc.set_xlim(0.0, float(horizon_day) + 0.1)

        finite = y[np.isfinite(y)]
        if finite.size > 0:
            low = min(-0.1, float(np.nanmin(finite) - 0.05))
            high = max(1.0, float(np.nanmax(finite) + 0.05))
            ax_acc.set_ylim(low, min(high, 1.02))

        ax_acc.set_title("ACC progression")
        ax_acc.set_xlabel("Lead time (days)")
        ax_acc.set_ylabel("ACC")
        ax_acc.legend(frameon=False, loc="lower left")
        text = (
            f"Init: {_pretty_dt(init_time)}\n"
            f"Valid: {_pretty_dt(valid_time)}\n"
            f"Lead: {lead_days[step_idx]:.2f} d ({int(lead_hours[step_idx])} h)\n"
            f"ACC now: {current_acc:.4f}\n"
            f"ACC running mean: {running_acc:.4f}"
        )
        ax_acc.text(
            0.98,
            0.98,
            text,
            transform=ax_acc.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95},
        )

        fig.suptitle(
            f"T2M forecast rollout to day {horizon_day} | frame {step_idx + 1}/{n_steps}",
            fontsize=13,
            fontweight="bold",
        )

        frames.append(_figure_to_pil(fig))
        plt.close(fig)

    return frames


def _build_multi_var_frames(
    pred_by_var: Dict[str, np.ndarray],
    truth_by_var: Dict[str, np.ndarray],
    init_time: np.datetime64,
    lead_hours: np.ndarray,
    lead_days: np.ndarray,
    horizon_day: int,
    acc_by_var: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    fps: float,
    frame_step: int,
    variable_names: Sequence[str],
    vars_per_page: int,
    title_prefix: str,
) -> List[Image.Image]:
    del fps

    n_steps = next(iter(pred_by_var.values())).shape[0]
    indices = list(range(0, n_steps, max(1, int(frame_step))))
    if indices[-1] != (n_steps - 1):
        indices.append(n_steps - 1)

    limits: Dict[str, Tuple[float, float]] = {}
    err_lims: Dict[str, float] = {}
    for name in variable_names:
        all_vals = np.concatenate([pred_by_var[name].reshape(-1), truth_by_var[name].reshape(-1)])
        limits[name] = _robust_limits(all_vals, lo=2.0, hi=98.0)
        err_lim = float(np.nanpercentile(np.abs(pred_by_var[name] - truth_by_var[name]), 99.0))
        err_lims[name] = max(err_lim, 1e-6)

    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    extent = [lon_min, lon_max, lat_min, lat_max]

    vars_per_page = max(1, int(vars_per_page))
    pages: List[List[str]] = [
        list(variable_names[i : i + vars_per_page])
        for i in range(0, len(variable_names), vars_per_page)
    ]

    all_curves: List[np.ndarray] = [
        acc_by_var[name][:n_steps]
        for name in variable_names
        if name in acc_by_var
    ]
    mean_curve_all = np.nanmean(np.vstack(all_curves), axis=0) if all_curves else None

    frames: List[Image.Image] = []
    for step_idx in indices:
        valid_time = init_time + np.timedelta64(int(lead_hours[step_idx]), "h")

        for page_index, page_vars in enumerate(pages, start=1):
            n_rows = len(page_vars) + 1
            fig_height = max(8.4, 2.55 * n_rows + 0.8)
            fig, axes = plt.subplots(n_rows, 3, figsize=(15.6, fig_height), constrained_layout=True)

            for row, name in enumerate(page_vars):
                p = pred_by_var[name][step_idx]
                t = truth_by_var[name][step_idx]
                e = p - t
                pmin, pmax = limits[name]
                el = err_lims[name]

                _draw_map_panel(
                    fig,
                    axes[row, 0],
                    t,
                    title=f"{name} | Target",
                    extent=extent,
                    cmap="cividis",
                    vmin=pmin,
                    vmax=pmax,
                    show_xlabel=(row == len(page_vars) - 1),
                    show_ylabel=True,
                )
                _draw_map_panel(
                    fig,
                    axes[row, 1],
                    p,
                    title=f"{name} | Prediction",
                    extent=extent,
                    cmap="cividis",
                    vmin=pmin,
                    vmax=pmax,
                    show_xlabel=(row == len(page_vars) - 1),
                    show_ylabel=False,
                )
                _draw_map_panel(
                    fig,
                    axes[row, 2],
                    e,
                    title=f"{name} | Error",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-el,
                    vmax=el,
                    show_xlabel=(row == len(page_vars) - 1),
                    show_ylabel=False,
                )

            acc_ax = axes[-1, 0]
            mean_ax = axes[-1, 1]
            text_ax = axes[-1, 2]

            means_for_step: List[float] = []
            page_curves: List[np.ndarray] = []
            for name in page_vars:
                series = acc_by_var.get(name)
                if series is None:
                    continue
                y = series[:n_steps]
                acc_ax.plot(lead_days[:n_steps], y, linewidth=1.9, label=name)
                acc_ax.scatter([lead_days[step_idx]], [y[step_idx]], s=24)
                means_for_step.append(float(y[step_idx]))
                page_curves.append(y)

            acc_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            acc_ax.axvline(float(horizon_day), color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
            acc_ax.set_xlim(0.0, float(horizon_day) + 0.1)
            acc_ax.set_title("ACC by variable (page)")
            acc_ax.set_xlabel("Lead time (days)")
            acc_ax.set_ylabel("ACC")
            if page_curves:
                acc_ax.legend(frameon=False, fontsize=7, loc="lower left")

            if mean_curve_all is not None:
                mean_ax.plot(
                    lead_days[:n_steps],
                    mean_curve_all,
                    color="#d62728",
                    linewidth=2.4,
                    label="Mean ACC (all vars)",
                )
                mean_ax.scatter([lead_days[step_idx]], [mean_curve_all[step_idx]], s=36, color="black")
                mean_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
                mean_ax.axvline(float(horizon_day), color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
                mean_ax.set_xlim(0.0, float(horizon_day) + 0.1)
                mean_ax.set_title("Mean ACC (all vars)")
                mean_ax.set_xlabel("Lead time (days)")
                mean_ax.set_ylabel("ACC")
                mean_ax.legend(frameon=False, loc="lower left", fontsize=8)

            text_ax.axis("off")
            mean_now = float(np.nanmean(means_for_step)) if means_for_step else float("nan")
            text = (
                f"Init: {_pretty_dt(init_time)}\n"
                f"Valid: {_pretty_dt(valid_time)}\n"
                f"Lead: {lead_days[step_idx]:.2f} d ({int(lead_hours[step_idx])} h)\n"
                f"Mean ACC now (page): {mean_now:.4f}\n"
                f"Page {page_index}/{len(pages)} | Vars on page: {len(page_vars)}"
            )
            text_ax.text(
                0.02,
                0.98,
                text,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "#bbbbbb",
                    "alpha": 0.95,
                },
            )

            fig.suptitle(
                f"{title_prefix} to day {horizon_day} | frame {step_idx + 1}/{n_steps} | page {page_index}/{len(pages)}",
                fontsize=13,
                fontweight="bold",
            )

            frames.append(_figure_to_pil(fig))
            plt.close(fig)

    return frames


def _build_rich_frames(
    pred_by_var: Dict[str, np.ndarray],
    truth_by_var: Dict[str, np.ndarray],
    init_time: np.datetime64,
    lead_hours: np.ndarray,
    lead_days: np.ndarray,
    horizon_day: int,
    acc_by_var: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    fps: float,
    frame_step: int,
    rich_vars: Sequence[str],
) -> List[Image.Image]:
    return _build_multi_var_frames(
        pred_by_var=pred_by_var,
        truth_by_var=truth_by_var,
        init_time=init_time,
        lead_hours=lead_hours,
        lead_days=lead_days,
        horizon_day=horizon_day,
        acc_by_var=acc_by_var,
        lat=lat,
        lon=lon,
        fps=fps,
        frame_step=frame_step,
        variable_names=list(rich_vars),
        vars_per_page=max(1, len(rich_vars)),
        title_prefix="Richer-variable rollout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 6 forecast GIFs (t2m + richer vars across 5/10/15 day horizons)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--forecast-file", type=str, required=True, help="Path to forecast.nc")
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to metrics_per_lead.csv (used for ACC overlays)",
    )
    parser.add_argument("--output-dir", type=str, default="prof/report/gifs", help="GIF output directory")

    parser.add_argument("--horizon-days", type=parse_csv_ints, default=[5, 10, 15])
    parser.add_argument(
        "--horizon-init-indices",
        type=parse_csv_ints,
        default=[0, 1, 2],
        help="Init index to use for each horizon; defaults map 5d->0, 10d->1, 15d->2",
    )
    parser.add_argument("--t2m-var-name", type=str, default="2m_temperature")
    parser.add_argument("--rich-vars", type=parse_csv_strings, default=list(DEFAULT_RICH_VARS))
    parser.add_argument(
        "--make-all-vars-gif",
        action="store_true",
        help="Also generate an all-variable GIF (paged) for each horizon/init pair",
    )
    parser.add_argument(
        "--all-vars-per-page",
        type=int,
        default=6,
        help="Variables per page in all-variable GIF",
    )

    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--frame-step", type=int, default=1, help="Render every Nth step, always includes final step")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    forecast_path = Path(args.forecast_file).expanduser().resolve()
    metrics_path = Path(args.metrics_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not forecast_path.is_file():
        raise FileNotFoundError(f"Forecast file not found: {forecast_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    horizons = sorted(set(int(v) for v in args.horizon_days if int(v) > 0))
    if not horizons:
        raise ValueError("At least one positive horizon day is required")

    ds = _open_dataset(forecast_path)
    try:
        channel_names = [str(v) for v in ds["channel_name"].values.tolist()]
        n_inits = int(ds.sizes["init_time"])
        n_steps = int(ds.sizes["lead_step"])

        lead_steps = ds["lead_step"].values.astype(np.int64)
        if "lead_hour" in ds:
            lead_hours = ds["lead_hour"].values.astype(np.int64)
        else:
            lead_hours = lead_steps * 6
        lead_days = lead_hours.astype(np.float64) / 24.0

        init_times = np.asarray([_to_np_datetime(v) for v in ds["init_time"].values])
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)

        init_indices = _resolve_init_indices(args.horizon_init_indices, len(horizons), n_inits)

        if args.t2m_var_name not in channel_names:
            raise ValueError(
                f"t2m variable '{args.t2m_var_name}' missing in forecast file. Available count={len(channel_names)}"
            )
        t2m_idx = channel_names.index(args.t2m_var_name)

        rich_vars = [v for v in args.rich_vars if v in channel_names]
        if len(rich_vars) < 3:
            raise ValueError(
                "Need 3 richer variables present in forecast file. "
                f"Requested={args.rich_vars}, available={rich_vars}"
            )
        rich_vars = rich_vars[:3]

        acc_lookup = _read_acc_lookup(metrics_path, n_steps=n_steps)
        if args.t2m_var_name not in acc_lookup:
            raise ValueError(
                f"ACC series for {args.t2m_var_name} not found in {metrics_path}"
            )
        for name in rich_vars:
            if name not in acc_lookup:
                raise ValueError(f"ACC series for {name} not found in {metrics_path}")

        all_vars = [name for name in channel_names if name in set(acc_lookup.keys())]
        if args.make_all_vars_gif and not all_vars:
            raise ValueError("No variables with ACC series available for all-variable GIF")

        records: List[GifRecord] = []

        for horizon_day, init_idx in zip(horizons, init_indices):
            init_time = init_times[init_idx]
            step_count = _resolve_horizon_steps(lead_days, horizon_day)

            t2m_pred, t2m_truth = _read_var_block(
                ds=ds,
                init_idx=init_idx,
                step_count=step_count,
                channel_idx=t2m_idx,
            )
            t2m_acc = acc_lookup[args.t2m_var_name][:step_count]

            t2m_frames = _build_t2m_frames(
                pred=t2m_pred,
                truth=t2m_truth,
                init_time=init_time,
                lead_hours=lead_hours[:step_count],
                lead_days=lead_days[:step_count],
                horizon_day=horizon_day,
                acc_series=t2m_acc,
                lat=lat,
                lon=lon,
                fps=args.fps,
                frame_step=args.frame_step,
                variable_name=args.t2m_var_name,
            )
            t2m_name = f"t2m_rollout_{horizon_day:02d}d_init{init_idx}_{_file_dt_tag(init_time)}.gif"
            t2m_path = out_dir / t2m_name
            _save_gif(t2m_frames, t2m_path, fps=args.fps)
            records.append(
                GifRecord(
                    kind="t2m",
                    horizon_days=horizon_day,
                    init_index=init_idx,
                    init_time=_pretty_dt(init_time),
                    output_file=str(t2m_path),
                    variables=[args.t2m_var_name],
                    frames=len(t2m_frames),
                )
            )
            print(f"[saved] {t2m_path} ({len(t2m_frames)} frames)")

            rich_pred: Dict[str, np.ndarray] = {}
            rich_truth: Dict[str, np.ndarray] = {}
            for name in rich_vars:
                cidx = channel_names.index(name)
                p, t = _read_var_block(
                    ds=ds,
                    init_idx=init_idx,
                    step_count=step_count,
                    channel_idx=cidx,
                )
                rich_pred[name] = p
                rich_truth[name] = t

            rich_acc = {name: acc_lookup[name][:step_count] for name in rich_vars}
            rich_frames = _build_rich_frames(
                pred_by_var=rich_pred,
                truth_by_var=rich_truth,
                init_time=init_time,
                lead_hours=lead_hours[:step_count],
                lead_days=lead_days[:step_count],
                horizon_day=horizon_day,
                acc_by_var=rich_acc,
                lat=lat,
                lon=lon,
                fps=args.fps,
                frame_step=args.frame_step,
                rich_vars=rich_vars,
            )
            rich_name = f"rich_rollout_{horizon_day:02d}d_init{init_idx}_{_file_dt_tag(init_time)}.gif"
            rich_path = out_dir / rich_name
            _save_gif(rich_frames, rich_path, fps=args.fps)
            records.append(
                GifRecord(
                    kind="rich",
                    horizon_days=horizon_day,
                    init_index=init_idx,
                    init_time=_pretty_dt(init_time),
                    output_file=str(rich_path),
                    variables=list(rich_vars),
                    frames=len(rich_frames),
                )
            )
            print(f"[saved] {rich_path} ({len(rich_frames)} frames)")

            if args.make_all_vars_gif:
                all_pred: Dict[str, np.ndarray] = {}
                all_truth: Dict[str, np.ndarray] = {}
                for name in all_vars:
                    cidx = channel_names.index(name)
                    p, t = _read_var_block(
                        ds=ds,
                        init_idx=init_idx,
                        step_count=step_count,
                        channel_idx=cidx,
                    )
                    all_pred[name] = p
                    all_truth[name] = t

                all_acc = {name: acc_lookup[name][:step_count] for name in all_vars}
                all_frames = _build_multi_var_frames(
                    pred_by_var=all_pred,
                    truth_by_var=all_truth,
                    init_time=init_time,
                    lead_hours=lead_hours[:step_count],
                    lead_days=lead_days[:step_count],
                    horizon_day=horizon_day,
                    acc_by_var=all_acc,
                    lat=lat,
                    lon=lon,
                    fps=args.fps,
                    frame_step=args.frame_step,
                    variable_names=all_vars,
                    vars_per_page=max(1, int(args.all_vars_per_page)),
                    title_prefix="All-variable rollout",
                )
                all_name = f"allvars_rollout_{horizon_day:02d}d_init{init_idx}_{_file_dt_tag(init_time)}.gif"
                all_path = out_dir / all_name
                _save_gif(all_frames, all_path, fps=args.fps)
                records.append(
                    GifRecord(
                        kind="all_vars",
                        horizon_days=horizon_day,
                        init_index=init_idx,
                        init_time=_pretty_dt(init_time),
                        output_file=str(all_path),
                        variables=list(all_vars),
                        frames=len(all_frames),
                    )
                )
                print(f"[saved] {all_path} ({len(all_frames)} frames)")

        metadata = {
            "forecast_file": str(forecast_path),
            "metrics_file": str(metrics_path),
            "horizons": horizons,
            "horizon_init_indices": init_indices,
            "rich_variables": rich_vars,
            "all_variable_mode": bool(args.make_all_vars_gif),
            "all_variables_per_page": int(args.all_vars_per_page),
            "records": [r.__dict__ for r in records],
        }
        metadata_path = out_dir / "gif_manifest.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[saved] {metadata_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
