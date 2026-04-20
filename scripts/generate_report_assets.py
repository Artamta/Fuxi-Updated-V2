#!/usr/bin/env python3
"""
Generate report figures from existing forecast/metrics artifacts.

This script is designed for fast report assembly when full checkpoint evaluation
cannot be re-run in the current environment.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.facecolor": "#fbfbfc",
            "axes.facecolor": "#fbfbfc",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready plot assets from existing results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Repo root (auto-detected by default)")
    parser.add_argument("--output-dir", type=str, default="report", help="Output directory for report images")

    parser.add_argument(
        "--forecast-a",
        type=str,
        default="results_new/checkpoint_emb768/forecast/forecast.nc",
        help="Forecast NetCDF for model A",
    )
    parser.add_argument(
        "--forecast-b",
        type=str,
        default="results_new/checkpoint_emb1024/forecast/forecast.nc",
        help="Forecast NetCDF for model B",
    )
    parser.add_argument(
        "--metrics-a",
        type=str,
        default="results_new/checkpoint_emb768/metrics/metrics_per_lead.csv",
        help="metrics_per_lead.csv for model A",
    )
    parser.add_argument(
        "--metrics-b",
        type=str,
        default="results_new/checkpoint_emb1024/metrics/metrics_per_lead.csv",
        help="metrics_per_lead.csv for model B",
    )
    parser.add_argument(
        "--history-a",
        type=str,
        default="Models_paper/pretrain/embed_768_8-32-8/history.csv",
        help="Training history CSV for model A",
    )
    parser.add_argument(
        "--history-b",
        type=str,
        default="Models_paper/pretrain/embed_1024_8-32-8/history.csv",
        help="Training history CSV for model B",
    )

    parser.add_argument("--label-a", type=str, default="Embed-768")
    parser.add_argument("--label-b", type=str, default="Embed-1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scatter-points", type=int, default=180_000)
    return parser.parse_args()


def resolve_path(repo_root: Path, path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def read_forecast(path: Path) -> Dict[str, np.ndarray]:
    ds = xr.open_dataset(path)
    try:
        var_names = [str(v) for v in ds["channel_name"].values.tolist()]
        lead_hours = ds["lead_hour"].values.astype(np.int32) if "lead_hour" in ds else (ds["lead_step"].values.astype(np.int32) * 6)
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)

        pred = ds["forecast"].isel(init_time=0).values.astype(np.float32)  # (S, C, H, W)
        truth = ds["truth"].isel(init_time=0).values.astype(np.float32)    # (S, C, H, W)
    finally:
        ds.close()

    return {
        "var_names": np.array(var_names, dtype=object),
        "lead_hours": lead_hours,
        "lat": lat,
        "lon": lon,
        "pred": pred,
        "truth": truth,
    }


def read_metrics(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"variable", "lead_step", "lead_day", "rmse", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    global_df = (
        df.groupby("lead_day", as_index=False)
        .agg(rmse_mean=("rmse", "mean"), acc_mean=("acc", "mean"))
        .sort_values("lead_day")
    )
    return df, global_df


def nearest_step(lead_hours: np.ndarray, target_hours: float) -> int:
    return int(np.argmin(np.abs(lead_hours.astype(np.float64) - float(target_hours))))


def robust_limits(values: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    vmin = float(np.nanpercentile(values, lo))
    vmax = float(np.nanpercentile(values, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return -1.0, 1.0
    return vmin, vmax


def choose_level_variable(var_names: Sequence[str], level: int) -> Tuple[int, str]:
    preferred = [
        f"geopotential_plev{level}",
        f"temperature_plev{level}",
        f"u_component_of_wind_plev{level}",
        f"v_component_of_wind_plev{level}",
        f"specific_humidity_plev{level}",
    ]
    for name in preferred:
        if name in var_names:
            return var_names.index(name), name
    idxs = [i for i, n in enumerate(var_names) if n.endswith(f"_plev{level}")]
    if idxs:
        return idxs[0], var_names[idxs[0]]
    return 0, var_names[0]


def plot_target_vs_pred(out_path: Path, model: Dict[str, np.ndarray], variable: str = "geopotential_plev500", lead_hours: int = 120) -> None:
    var_names = model["var_names"].tolist()
    var_idx = var_names.index(variable) if variable in var_names else 0
    step_idx = nearest_step(model["lead_hours"], lead_hours)

    pred = model["pred"][step_idx, var_idx]
    truth = model["truth"][step_idx, var_idx]
    err = pred - truth

    vmin, vmax = robust_limits(np.concatenate([pred.ravel(), truth.ravel()]))
    err_lim = max(float(np.nanpercentile(np.abs(err), 99.0)), 1e-8)

    lon = model["lon"]
    lat = model["lat"]
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    panels = [
        (truth, "Truth", "coolwarm", vmin, vmax),
        (pred, "Prediction", "coolwarm", vmin, vmax),
        (err, "Error (pred - truth)", "RdBu_r", -err_lim, err_lim),
    ]

    for ax, (arr, title, cmap, pmin, pmax) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap=cmap, extent=extent, aspect="auto", vmin=pmin, vmax=pmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    lead_day = float(model["lead_hours"][step_idx]) / 24.0
    fig.suptitle(f"{variable} at lead day {lead_day:.2f}", fontsize=12)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_level_scatter(
    out_path: Path,
    model: Dict[str, np.ndarray],
    level: int,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    var_names = model["var_names"].tolist()
    idx, var_name = choose_level_variable(var_names, level)
    if idx is None:
        # Fallback: save an informative empty panel.
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No variables available at {level} hPa", ha="center", va="center")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    truth = model["truth"][:, idx, :, :].reshape(-1)
    pred = model["pred"][:, idx, :, :].reshape(-1)

    finite = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[finite]
    pred = pred[finite]

    if truth.size > max_points:
        choose = rng.choice(truth.size, size=max_points, replace=False)
        truth = truth[choose]
        pred = pred[choose]

    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    corr = float(np.corrcoef(pred, truth)[0, 1]) if truth.size > 1 else float("nan")
    bias = float(np.mean(pred - truth))

    low = float(np.nanpercentile(np.concatenate([truth, pred]), 0.5))
    high = float(np.nanpercentile(np.concatenate([truth, pred]), 99.5))
    pad = 0.04 * (high - low)
    low -= pad
    high += pad

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    hb = ax.hexbin(truth, pred, gridsize=110, bins="log", mincnt=1, cmap="magma")
    ax.plot([low, high], [low, high], color="red", linewidth=1.5, linestyle="--", label="1:1 line")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"{level} hPa scatter ({var_name})")
    ax.legend(frameon=False)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log10(count)")

    text = f"N={truth.size:,}\nRMSE={rmse:.3f}\nCorr={corr:.3f}\nBias={bias:.3f}"
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_temporal_metrics(out_path: Path, g_a: pd.DataFrame, g_b: pd.DataFrame, label_a: str, label_b: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    axes[0].plot(g_a["lead_day"], g_a["rmse_mean"], linewidth=2.4, marker="o", markersize=2.8, label=label_a)
    axes[0].plot(g_b["lead_day"], g_b["rmse_mean"], linewidth=2.4, marker="o", markersize=2.8, label=label_b)
    axes[0].set_title("Mean RMSE vs lead time")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(g_a["lead_day"], g_a["acc_mean"], linewidth=2.4, marker="o", markersize=2.8, label=label_a)
    axes[1].plot(g_b["lead_day"], g_b["acc_mean"], linewidth=2.4, marker="o", markersize=2.8, label=label_b)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("Mean ACC vs lead time")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    for ax in axes:
        for d in (5, 10, 15):
            ax.axvline(float(d), color="#6b7280", linestyle=":", linewidth=1.0, alpha=0.45)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_latitude_error(out_path: Path, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], label_a: str, label_b: str) -> None:
    lat = a["lat"]

    # Normalize channel-wise to avoid geopotential dominating the aggregate profile.
    ch_scale_a = np.nanstd(a["truth"], axis=(0, 2, 3), keepdims=True)
    ch_scale_b = np.nanstd(b["truth"], axis=(0, 2, 3), keepdims=True)
    ch_scale_a = np.maximum(ch_scale_a, 1e-8)
    ch_scale_b = np.maximum(ch_scale_b, 1e-8)

    err_a = np.mean(np.abs((a["pred"] - a["truth"]) / ch_scale_a), axis=(1, 3))  # (S, H)
    err_b = np.mean(np.abs((b["pred"] - b["truth"]) / ch_scale_b), axis=(1, 3))  # (S, H)

    lead_hours = a["lead_hours"].astype(np.float64)
    target_days = [1.0, 5.0, 10.0, 15.0]
    idxs = [nearest_step(lead_hours, d * 24.0) for d in target_days]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.6), constrained_layout=True)

    for d, idx in zip(target_days, idxs):
        axes[0].plot(lat, err_a[idx], linewidth=2.0, label=f"{label_a} day {d:.0f}")
    axes[0].set_title("Latitude-wise normalized MAE profile")
    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Normalized MAE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    idx15 = nearest_step(lead_hours, 15.0 * 24.0)
    axes[1].plot(lat, err_a[idx15], linewidth=2.2, label=f"{label_a} day 15")
    axes[1].plot(lat, err_b[idx15], linewidth=2.2, label=f"{label_b} day 15")
    axes[1].set_title("Day-15 latitude normalized error comparison")
    axes[1].set_xlabel("Latitude")
    axes[1].set_ylabel("Normalized MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_train_val(out_path: Path, hist_a: Path, hist_b: Path, label_a: str, label_b: str) -> None:
    df_a = pd.read_csv(hist_a)
    df_b = pd.read_csv(hist_b)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), constrained_layout=True, sharex=True)

    for df, label, color in ((df_a, label_a, "#1f77b4"), (df_b, label_b, "#ff7f0e")):
        train = df["train_loss"].astype(float)
        val = df["val_loss"].astype(float)
        # Light smoothing makes trend comparisons clearer while preserving raw traces.
        train_s = train.ewm(alpha=0.25, adjust=False).mean()
        val_s = val.ewm(alpha=0.25, adjust=False).mean()

        axes[0].plot(df["epoch"], train, linewidth=1.3, alpha=0.3, color=color)
        axes[0].plot(df["epoch"], train_s, linewidth=2.3, color=color, label=label)

        axes[1].plot(df["epoch"], val, linewidth=1.3, alpha=0.3, color=color)
        axes[1].plot(df["epoch"], val_s, linewidth=2.3, color=color, label=label)

    axes[0].set_title("Training loss (raw + EMA)")
    axes[0].set_ylabel("Train loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].set_title("Validation loss (raw + EMA)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_variable_error(out_path: Path, df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> None:
    per_a = df_a.groupby("variable", as_index=False).agg(rmse_mean=("rmse", "mean"), acc_mean=("acc", "mean"))
    per_b = df_b.groupby("variable", as_index=False).agg(rmse_mean=("rmse", "mean"), acc_mean=("acc", "mean"))

    merged = per_a.merge(per_b, on="variable", suffixes=("_a", "_b"))
    # Relative RMSE change (%) is far more interpretable across mixed-unit variables.
    merged["rmse_change_pct"] = 100.0 * (merged["rmse_mean_b"] - merged["rmse_mean_a"]) / np.maximum(np.abs(merged["rmse_mean_a"]), 1e-8)
    merged["acc_change"] = merged["acc_mean_b"] - merged["acc_mean_a"]
    merged = merged.sort_values("rmse_change_pct", ascending=False).reset_index(drop=True)

    x = np.arange(len(merged))
    w = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(15.0, 9.2), constrained_layout=True)

    colors = ["#2ca02c" if v <= 0.0 else "#d62728" for v in merged["rmse_change_pct"]]
    axes[0].bar(x, merged["rmse_change_pct"], color=colors)
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title(f"Per-variable RMSE change (%) for {label_b} vs {label_a}")
    axes[0].set_ylabel("RMSE change (%)")
    axes[0].grid(axis="y", alpha=0.25)

    colors_acc = ["#2ca02c" if v >= 0.0 else "#d62728" for v in merged["acc_change"]]
    axes[1].bar(x, merged["acc_change"], color=colors_acc)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_title(f"Per-variable ACC change for {label_b} vs {label_a}")
    axes[1].set_ylabel("ACC change")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(merged["variable"], rotation=65, ha="right", fontsize=8)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_cascade_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    ax.axis("off")

    def add_box(x0: float, y0: float, text: str, fc: str) -> None:
        rect = plt.Rectangle((x0, y0), 2.2, 0.9, facecolor=fc, edgecolor="#1f2937", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x0 + 1.1, y0 + 0.45, text, ha="center", va="center", fontsize=10, color="#111827")

    add_box(0.5, 1.6, "Initial state\n(t0)", "#e5f4ff")
    add_box(3.3, 1.6, "Short model\n(0-5 days)", "#d9f99d")
    add_box(6.1, 1.6, "Medium model\n(5-10 days)", "#fde68a")
    add_box(8.9, 1.6, "Long model\n(10-15 days)", "#fecaca")

    for x0 in [2.75, 5.55, 8.35]:
        ax.annotate("", xy=(x0 + 0.45, 2.05), xytext=(x0 - 0.25, 2.05), arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#1f2937"))

    ax.text(0.5, 0.6, "Autoregressive chaining: output of one stage is fed as input to the next stage.", fontsize=10)
    ax.text(0.5, 0.28, "Optional perturbations for ensemble members can be applied at each transition.", fontsize=9, color="#374151")

    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 3.1)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    return False


def main() -> None:
    args = parse_args()

    set_plot_style()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    out_dir = resolve_path(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_a = resolve_path(repo_root, args.forecast_a)
    forecast_b = resolve_path(repo_root, args.forecast_b)
    metrics_a = resolve_path(repo_root, args.metrics_a)
    metrics_b = resolve_path(repo_root, args.metrics_b)
    history_a = resolve_path(repo_root, args.history_a)
    history_b = resolve_path(repo_root, args.history_b)

    for p in [forecast_a, forecast_b, metrics_a, metrics_b, history_a, history_b]:
        ensure_exists(p)

    model_a = read_forecast(forecast_a)
    model_b = read_forecast(forecast_b)

    df_a, g_a = read_metrics(metrics_a)
    df_b, g_b = read_metrics(metrics_b)

    rng = np.random.default_rng(args.seed)

    # Core report placeholders.
    plot_target_vs_pred(out_dir / "target_vs_pred.png", model=model_a)
    plot_level_scatter(out_dir / "250.png", model=model_a, level=250, rng=rng, max_points=args.max_scatter_points)
    plot_level_scatter(out_dir / "500.png", model=model_a, level=500, rng=rng, max_points=args.max_scatter_points)
    plot_level_scatter(out_dir / "850.png", model=model_a, level=850, rng=rng, max_points=args.max_scatter_points)
    plot_temporal_metrics(out_dir / "temporal_rmse.png", g_a, g_b, args.label_a, args.label_b)
    plot_latitude_error(out_dir / "lat_err.png", model_a, model_b, args.label_a, args.label_b)
    plot_train_val(out_dir / "train_val.png", history_a, history_b, args.label_a, args.label_b)
    plot_per_variable_error(out_dir / "per_var_err.png", df_a, df_b, args.label_a, args.label_b)

    # Architecture and cascade assets.
    arc_src = resolve_path(repo_root, "notebooks/outputs/fuxi_architecture_paper_level.png")
    copied_arc = copy_if_exists(arc_src, out_dir / "arc.png")
    if not copied_arc:
        raise FileNotFoundError(f"Missing architecture image: {arc_src}")

    plot_cascade_diagram(out_dir / "cascade.png")

    # Optional convenience copies for extra report figures.
    copy_if_exists(resolve_path(repo_root, "results_shared/comparison/model_comparison.png"), out_dir / "model_comparison.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb768/forecast/day1_forecast.png"), out_dir / "day1_forecast_emb768.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb1024/forecast/day1_forecast.png"), out_dir / "day1_forecast_emb1024.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb768/metrics/poster_all20_rmse_per_variable.png"), out_dir / "poster_all20_rmse_emb768.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb768/metrics/poster_all20_acc_per_variable.png"), out_dir / "poster_all20_acc_emb768.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb1024/metrics/poster_all20_rmse_per_variable.png"), out_dir / "poster_all20_rmse_emb1024.png")
    copy_if_exists(resolve_path(repo_root, "results_new/checkpoint_emb1024/metrics/poster_all20_acc_per_variable.png"), out_dir / "poster_all20_acc_emb1024.png")

    print("Generated report assets in:", out_dir)
    for name in [
        "arc.png",
        "target_vs_pred.png",
        "250.png",
        "500.png",
        "850.png",
        "temporal_rmse.png",
        "lat_err.png",
        "train_val.png",
        "per_var_err.png",
        "cascade.png",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
