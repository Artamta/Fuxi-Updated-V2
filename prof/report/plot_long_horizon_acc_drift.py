#!/usr/bin/env python3
"""
Plot long-horizon autoregressive ACC drift from metrics_per_lead.csv.

Recommended use: metrics from a 90-day run (360 lead steps at 6-hour cadence).
Outputs:
- acc_drift_overview.png
- acc_drift_key_variables.png
- acc_threshold_summary.csv
- acc_drift_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE_NAMES = {
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
}

DEFAULT_KEY_VARS = [
    "2m_temperature",
    "geopotential_plev500",
    "total_column_water_vapour",
    "u_component_of_wind_plev850",
]


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_csv_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


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
            "figure.dpi": 140,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ACC drift plots/tables from metrics_per_lead.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics-file", type=str, required=True, help="Path to metrics_per_lead.csv")
    parser.add_argument("--output-dir", type=str, default="prof/report/plots", help="Output directory")
    parser.add_argument("--t2m-var-name", type=str, default="2m_temperature")
    parser.add_argument("--key-vars", type=parse_csv_strings, default=list(DEFAULT_KEY_VARS))
    parser.add_argument("--thresholds", type=parse_csv_floats, default=[0.8, 0.6, 0.4])
    parser.add_argument(
        "--smoothing-window-steps",
        type=int,
        default=1,
        help="Rolling mean smoothing in lead steps (1 disables smoothing)",
    )
    parser.add_argument(
        "--final-window-days",
        type=float,
        default=7.0,
        help="Window length near max lead for summary stats",
    )
    return parser


def _smooth(series: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return series
    s = pd.Series(series)
    return s.rolling(window=int(window), min_periods=1, center=True).mean().to_numpy(dtype=np.float64)


def _first_crossing_day(lead_days: np.ndarray, series: np.ndarray, threshold: float) -> float:
    finite = np.isfinite(series)
    if not np.any(finite):
        return float("nan")
    idx = np.where((series < float(threshold)) & finite)[0]
    if idx.size == 0:
        return float("nan")
    return float(lead_days[int(idx[0])])


def _mean_series(pivot: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
    available = [v for v in variables if v in pivot.columns]
    if not available:
        return np.full((pivot.shape[0],), np.nan, dtype=np.float64)
    return pivot[available].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)


def _series(pivot: pd.DataFrame, name: str) -> np.ndarray:
    if name in pivot.columns:
        return pivot[name].to_numpy(dtype=np.float64)
    return np.full((pivot.shape[0],), np.nan, dtype=np.float64)


def _safe_polyfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    coeff = np.polyfit(x[finite], y[finite], deg=1)
    return float(coeff[0])


def _write_threshold_table(
    out_path: Path,
    lead_days: np.ndarray,
    series_by_name: Dict[str, np.ndarray],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for name, series in series_by_name.items():
        for threshold in thresholds:
            rows.append(
                {
                    "series": name,
                    "threshold": float(threshold),
                    "first_day_below_threshold": _first_crossing_day(lead_days, series, threshold),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def _plot_overview(
    out_path: Path,
    lead_days: np.ndarray,
    all_mean: np.ndarray,
    surface_mean: np.ndarray,
    upper_mean: np.ndarray,
    t2m_series: np.ndarray,
    rich_mean: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11.8, 6.4), constrained_layout=True)

    ax.plot(lead_days, all_mean, color="#111111", linewidth=2.6, label="All-variable mean ACC")
    ax.plot(lead_days, surface_mean, color="#1f77b4", linewidth=2.2, label="Surface mean ACC")
    ax.plot(lead_days, upper_mean, color="#ff7f0e", linewidth=2.2, label="Upper-air mean ACC")
    ax.plot(lead_days, t2m_series, color="#2ca02c", linewidth=2.1, label="T2M ACC")
    ax.plot(lead_days, rich_mean, color="#d62728", linewidth=2.1, label="Richer-set mean ACC")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65, label="Climatology-skill baseline (ACC=0)")

    max_day = float(np.nanmax(lead_days))
    for marker in [5, 10, 15, 30, 60, 90]:
        if marker <= max_day + 1e-6:
            ax.axvline(float(marker), color="#777777", linestyle="--", linewidth=0.9, alpha=0.4)

    finite = np.concatenate([
        all_mean[np.isfinite(all_mean)],
        surface_mean[np.isfinite(surface_mean)],
        upper_mean[np.isfinite(upper_mean)],
        t2m_series[np.isfinite(t2m_series)],
        rich_mean[np.isfinite(rich_mean)],
    ])
    if finite.size > 0:
        ymin = min(-0.15, float(np.nanmin(finite) - 0.05))
        ymax = min(1.02, max(1.0, float(np.nanmax(finite) + 0.05)))
        ax.set_ylim(ymin, ymax)

    ax.set_title("Long-horizon ACC drift (autoregressive rollout)")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("ACC")
    ax.set_xlim(0.0, max_day)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _plot_key_vars(
    out_path: Path,
    lead_days: np.ndarray,
    series_by_name: Dict[str, np.ndarray],
    rich_mean: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11.8, 6.4), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    idx = 0
    for name, series in series_by_name.items():
        if not np.any(np.isfinite(series)):
            continue
        ax.plot(lead_days, series, linewidth=2.1, color=cmap(idx % 10), label=name)
        idx += 1

    ax.plot(lead_days, rich_mean, linewidth=2.6, color="#111111", linestyle="--", label="Richer-set mean")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)

    max_day = float(np.nanmax(lead_days))
    for marker in [5, 10, 15, 30, 60, 90]:
        if marker <= max_day + 1e-6:
            ax.axvline(float(marker), color="#777777", linestyle="--", linewidth=0.9, alpha=0.4)

    ax.set_title("Key-variable ACC trajectories")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("ACC")
    ax.set_xlim(0.0, max_day)
    ax.set_ylim(-0.2, 1.02)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    set_plot_style()

    metrics_path = Path(args.metrics_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    required = {"variable", "lead_step", "lead_day", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metrics_path} missing required columns: {sorted(missing)}")

    pivot = (
        df.pivot_table(index="lead_day", columns="variable", values="acc", aggfunc="mean")
        .sort_index()
        .astype(float)
    )
    lead_days = pivot.index.to_numpy(dtype=np.float64)

    variables = list(pivot.columns)
    surface_vars = [v for v in variables if v in SURFACE_NAMES]
    upper_vars = [v for v in variables if "_plev" in v]
    if not upper_vars:
        upper_vars = [v for v in variables if v not in surface_vars]

    key_vars = [v for v in args.key_vars if v in variables]
    if not key_vars:
        key_vars = [v for v in DEFAULT_KEY_VARS if v in variables]
    if not key_vars:
        key_vars = variables[: min(4, len(variables))]

    all_mean = _mean_series(pivot, variables)
    surface_mean = _mean_series(pivot, surface_vars)
    upper_mean = _mean_series(pivot, upper_vars)
    t2m_series = _series(pivot, args.t2m_var_name)
    rich_mean = _mean_series(pivot, key_vars)

    window = max(1, int(args.smoothing_window_steps))
    all_mean = _smooth(all_mean, window)
    surface_mean = _smooth(surface_mean, window)
    upper_mean = _smooth(upper_mean, window)
    t2m_series = _smooth(t2m_series, window)
    rich_mean = _smooth(rich_mean, window)

    key_series: Dict[str, np.ndarray] = {}
    for name in key_vars:
        key_series[name] = _smooth(_series(pivot, name), window)

    overview_png = out_dir / "acc_drift_overview.png"
    key_png = out_dir / "acc_drift_key_variables.png"
    threshold_csv = out_dir / "acc_threshold_summary.csv"
    summary_md = out_dir / "acc_drift_summary.md"

    _plot_overview(
        out_path=overview_png,
        lead_days=lead_days,
        all_mean=all_mean,
        surface_mean=surface_mean,
        upper_mean=upper_mean,
        t2m_series=t2m_series,
        rich_mean=rich_mean,
    )
    _plot_key_vars(
        out_path=key_png,
        lead_days=lead_days,
        series_by_name=key_series,
        rich_mean=rich_mean,
    )

    threshold_series: Dict[str, np.ndarray] = {
        "all_mean": all_mean,
        "surface_mean": surface_mean,
        "upper_air_mean": upper_mean,
        "t2m": t2m_series,
        "richer_mean": rich_mean,
    }
    for name, series in key_series.items():
        threshold_series[name] = series

    thresholds = [float(v) for v in args.thresholds]
    threshold_df = _write_threshold_table(
        out_path=threshold_csv,
        lead_days=lead_days,
        series_by_name=threshold_series,
        thresholds=thresholds,
    )

    max_day = float(np.nanmax(lead_days))
    final_start = max(0.0, max_day - float(args.final_window_days))
    final_mask = lead_days >= final_start
    final_mean_acc = float(np.nanmean(all_mean[final_mask])) if np.any(final_mask) else float(np.nanmean(all_mean))
    final_t2m_acc = float(np.nanmean(t2m_series[final_mask])) if np.any(final_mask) else float(np.nanmean(t2m_series))
    final_rich_acc = float(np.nanmean(rich_mean[final_mask])) if np.any(final_mask) else float(np.nanmean(rich_mean))

    tail_len = max(8, int(round(0.2 * lead_days.shape[0])))
    tail_x = lead_days[-tail_len:]
    tail_all = all_mean[-tail_len:]
    tail_t2m = t2m_series[-tail_len:]
    tail_rich = rich_mean[-tail_len:]

    slope_all = _safe_polyfit_slope(tail_x, tail_all)
    slope_t2m = _safe_polyfit_slope(tail_x, tail_t2m)
    slope_rich = _safe_polyfit_slope(tail_x, tail_rich)

    day_below_zero_all = _first_crossing_day(lead_days, all_mean, 0.0)
    day_below_zero_t2m = _first_crossing_day(lead_days, t2m_series, 0.0)
    day_below_zero_rich = _first_crossing_day(lead_days, rich_mean, 0.0)

    lines = []
    lines.append("# ACC Drift Summary")
    lines.append("")
    lines.append(f"- Metrics source: {metrics_path}")
    lines.append(f"- Max lead day in input: {max_day:.2f}")
    lines.append(f"- Smoothing window (steps): {window}")
    lines.append(f"- Final-window days: {float(args.final_window_days):.2f}")
    lines.append("")
    lines.append("## Final-window mean ACC")
    lines.append(f"- All-variable mean ACC: {final_mean_acc:.4f}")
    lines.append(f"- T2M ACC: {final_t2m_acc:.4f}")
    lines.append(f"- Richer-set mean ACC: {final_rich_acc:.4f}")
    lines.append("")
    lines.append("## Late-horizon slope (tail linear fit, ACC/day)")
    lines.append(f"- All-variable mean slope: {slope_all:.6f}")
    lines.append(f"- T2M slope: {slope_t2m:.6f}")
    lines.append(f"- Richer-set mean slope: {slope_rich:.6f}")
    lines.append("")
    lines.append("## Day of ACC crossing below climatology baseline (ACC < 0)")
    lines.append(f"- All-variable mean: {day_below_zero_all:.2f}" if np.isfinite(day_below_zero_all) else "- All-variable mean: not crossed")
    lines.append(f"- T2M: {day_below_zero_t2m:.2f}" if np.isfinite(day_below_zero_t2m) else "- T2M: not crossed")
    lines.append(f"- Richer-set mean: {day_below_zero_rich:.2f}" if np.isfinite(day_below_zero_rich) else "- Richer-set mean: not crossed")
    lines.append("")
    lines.append("## Interpretation helper")
    if np.isfinite(slope_all):
        if slope_all < -1e-4:
            lines.append("- All-variable ACC is still decreasing at late horizon.")
        elif slope_all > 1e-4:
            lines.append("- All-variable ACC has slight late-horizon recovery.")
        else:
            lines.append("- All-variable ACC is close to a late-horizon plateau.")
    lines.append("- Use acc_threshold_summary.csv to cite lead-time thresholds (ACC < 0.8/0.6/0.4, etc.).")

    summary_md.write_text("\n".join(lines) + "\n")

    print(f"[saved] {overview_png}")
    print(f"[saved] {key_png}")
    print(f"[saved] {threshold_csv} ({len(threshold_df)} rows)")
    print(f"[saved] {summary_md}")


if __name__ == "__main__":
    main()
