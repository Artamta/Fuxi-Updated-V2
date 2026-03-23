#!/usr/bin/env python3
"""
Post-process evaluate_checkpoint.py outputs.

Outputs:
- per_variable_summary.csv
- global_lead_summary.csv
- acc_over_lead.png
- rmse_over_lead.png
- acc_rolling_mean.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze evaluation metrics from evaluate_checkpoint.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory containing metrics_per_lead.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <eval-dir>/analysis)")
    parser.add_argument("--rolling-window", type=int, default=4, help="Rolling mean window in lead steps")
    return parser


def safe_day_mean(df: pd.DataFrame, metric: str, day: int) -> float:
    step = day * 4
    part = df[df["lead_step"] == step]
    if len(part) == 0:
        return float("nan")
    return float(part[metric].mean())


def main() -> None:
    args = build_parser().parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (eval_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    lead_csv = eval_dir / "metrics_per_lead.csv"
    if not lead_csv.is_file():
        raise FileNotFoundError(f"Missing metrics file: {lead_csv}")

    df = pd.read_csv(lead_csv)
    required = {"variable", "lead_step", "lead_day", "rmse", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics_per_lead.csv missing columns: {sorted(missing)}")

    # Per-variable summary across all lead steps.
    per_var = (
        df.groupby("variable", as_index=False)
        .agg(
            acc_mean_60=("acc", "mean"),
            acc_std_60=("acc", "std"),
            acc_min_60=("acc", "min"),
            acc_max_60=("acc", "max"),
            rmse_mean_60=("rmse", "mean"),
            rmse_std_60=("rmse", "std"),
            rmse_min_60=("rmse", "min"),
            rmse_max_60=("rmse", "max"),
        )
    )

    # Day-specific values per variable.
    for day in (5, 10, 15):
        step = day * 4
        day_part = (
            df[df["lead_step"] == step][["variable", "acc", "rmse"]]
            .rename(columns={"acc": f"acc_day{day}", "rmse": f"rmse_day{day}"})
        )
        per_var = per_var.merge(day_part, on="variable", how="left")

    per_var = per_var.sort_values("acc_mean_60", ascending=False).reset_index(drop=True)
    per_var.to_csv(out_dir / "per_variable_summary.csv", index=False)

    # Global lead-wise average over variables.
    global_lead = (
        df.groupby(["lead_step", "lead_day"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .sort_values("lead_step")
        .reset_index(drop=True)
    )

    w = max(1, int(args.rolling_window))
    global_lead[f"acc_rolling_mean_w{w}"] = global_lead["acc_mean"].rolling(window=w, min_periods=1).mean()
    global_lead["acc_cumulative_mean"] = global_lead["acc_mean"].expanding(min_periods=1).mean()
    global_lead.to_csv(out_dir / "global_lead_summary.csv", index=False)

    # Build compact textual summary.
    summary_lines = [
        f"eval_dir: {eval_dir}",
        f"n_variables: {df['variable'].nunique()}",
        f"n_lead_steps: {df['lead_step'].nunique()}",
        f"acc_mean_overall_60xvars: {df['acc'].mean():.6f}",
        f"rmse_mean_overall_60xvars: {df['rmse'].mean():.6f}",
        f"acc_mean_day5: {safe_day_mean(df, 'acc', 5):.6f}",
        f"acc_mean_day10: {safe_day_mean(df, 'acc', 10):.6f}",
        f"acc_mean_day15: {safe_day_mean(df, 'acc', 15):.6f}",
        f"rmse_mean_day5: {safe_day_mean(df, 'rmse', 5):.6f}",
        f"rmse_mean_day10: {safe_day_mean(df, 'rmse', 10):.6f}",
        f"rmse_mean_day15: {safe_day_mean(df, 'rmse', 15):.6f}",
        f"rolling_window_steps: {w}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    # Plot: ACC over lead
    plt.figure(figsize=(9, 5.0))
    plt.plot(global_lead["lead_day"], global_lead["acc_mean"], label="ACC mean over variables", linewidth=2.0)
    plt.fill_between(
        global_lead["lead_day"],
        global_lead["acc_mean"] - global_lead["acc_std"].fillna(0.0),
        global_lead["acc_mean"] + global_lead["acc_std"].fillna(0.0),
        alpha=0.15,
        label="±1 std",
    )
    plt.xlabel("Lead day")
    plt.ylabel("ACC")
    plt.title("ACC over lead time (mean over variables)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_over_lead.png", dpi=180)
    plt.close()

    # Plot: RMSE over lead
    plt.figure(figsize=(9, 5.0))
    plt.plot(global_lead["lead_day"], global_lead["rmse_mean"], label="RMSE mean over variables", linewidth=2.0)
    plt.fill_between(
        global_lead["lead_day"],
        global_lead["rmse_mean"] - global_lead["rmse_std"].fillna(0.0),
        global_lead["rmse_mean"] + global_lead["rmse_std"].fillna(0.0),
        alpha=0.15,
        label="±1 std",
    )
    plt.xlabel("Lead day")
    plt.ylabel("RMSE")
    plt.title("RMSE over lead time (mean over variables)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_over_lead.png", dpi=180)
    plt.close()

    # Plot: ACC rolling mean
    plt.figure(figsize=(9, 5.0))
    plt.plot(global_lead["lead_day"], global_lead["acc_mean"], alpha=0.4, label="ACC mean", linewidth=1.5)
    plt.plot(
        global_lead["lead_day"],
        global_lead[f"acc_rolling_mean_w{w}"],
        label=f"ACC rolling mean (window={w})",
        linewidth=2.3,
    )
    plt.plot(
        global_lead["lead_day"],
        global_lead["acc_cumulative_mean"],
        label="ACC cumulative mean (1..step)",
        linewidth=2.0,
    )
    plt.xlabel("Lead day")
    plt.ylabel("ACC")
    plt.title("ACC rolling/cumulative mean over 60 lead steps")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_rolling_mean.png", dpi=180)
    plt.close()

    # Plot: per-variable ACC mean (sorted)
    acc_sorted = per_var.sort_values("acc_mean_60", ascending=False)
    plt.figure(figsize=(11, 6.5))
    plt.barh(acc_sorted["variable"], acc_sorted["acc_mean_60"])
    plt.gca().invert_yaxis()
    plt.xlabel("ACC mean over 60 lead steps")
    plt.title("Per-variable ACC (higher is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_variable_acc_mean.png", dpi=180)
    plt.close()

    # Plot: per-variable RMSE mean (sorted)
    rmse_sorted = per_var.sort_values("rmse_mean_60", ascending=True)
    plt.figure(figsize=(11, 6.5))
    plt.barh(rmse_sorted["variable"], rmse_sorted["rmse_mean_60"])
    plt.xlabel("RMSE mean over 60 lead steps")
    plt.title("Per-variable RMSE (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_variable_rmse_mean.png", dpi=180)
    plt.close()

    print("Saved analysis to:", out_dir)
    print(" -", out_dir / "per_variable_summary.csv")
    print(" -", out_dir / "global_lead_summary.csv")
    print(" -", out_dir / "summary.txt")
    print(" -", out_dir / "acc_over_lead.png")
    print(" -", out_dir / "rmse_over_lead.png")
    print(" -", out_dir / "acc_rolling_mean.png")
    print(" -", out_dir / "per_variable_acc_mean.png")
    print(" -", out_dir / "per_variable_rmse_mean.png")


if __name__ == "__main__":
    main()
