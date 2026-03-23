#!/usr/bin/env python3
"""
Compare two evaluation run directories produced by evaluate_checkpoint.py.

Each run directory is expected to contain:
- metrics_per_lead.csv
- summary.json (optional but recommended)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two evaluate_checkpoint outputs (ACC/RMSE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-a-dir", type=str, required=True)
    parser.add_argument("--run-b-dir", type=str, required=True)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--out-dir", type=str, required=True)
    return parser


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, dict]:
    lead_csv = run_dir / "metrics_per_lead.csv"
    if not lead_csv.is_file():
        raise FileNotFoundError(f"Missing file: {lead_csv}")
    lead_df = pd.read_csv(lead_csv)

    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    return lead_df, summary


def compute_global_metrics(lead_df: pd.DataFrame) -> dict:
    out = {}
    out["mean_rmse_overall"] = float(lead_df["rmse"].mean())
    out["mean_acc_overall"] = float(lead_df["acc"].mean())

    for day in (5, 10, 15):
        step = day * 4
        day_df = lead_df[lead_df["lead_step"] == step]
        if len(day_df) == 0:
            out[f"rmse_day{day}"] = float("nan")
            out[f"acc_day{day}"] = float("nan")
        else:
            out[f"rmse_day{day}"] = float(day_df["rmse"].mean())
            out[f"acc_day{day}"] = float(day_df["acc"].mean())
    return out


def align_runs(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["variable", "lead_step"]
    common = a[keys].merge(b[keys], on=keys, how="inner")
    a2 = common.merge(a, on=keys, how="left")
    b2 = common.merge(b, on=keys, how="left")
    return a2, b2


def plot_global_curves(
    out_dir: Path,
    a: pd.DataFrame,
    b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    a_step = a.groupby("lead_step", as_index=False)[["rmse", "acc"]].mean().sort_values("lead_step")
    b_step = b.groupby("lead_step", as_index=False)[["rmse", "acc"]].mean().sort_values("lead_step")
    lead_days_a = a_step["lead_step"].to_numpy() * 6.0 / 24.0
    lead_days_b = b_step["lead_step"].to_numpy() * 6.0 / 24.0

    plt.figure(figsize=(9, 5.2))
    plt.plot(lead_days_a, a_step["rmse"].to_numpy(), label=label_a, linewidth=2)
    plt.plot(lead_days_b, b_step["rmse"].to_numpy(), label=label_b, linewidth=2)
    plt.xlabel("Lead day")
    plt.ylabel("Mean RMSE across variables")
    plt.title("RMSE over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_mean_rmse_over_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.2))
    plt.plot(lead_days_a, a_step["acc"].to_numpy(), label=label_a, linewidth=2)
    plt.plot(lead_days_b, b_step["acc"].to_numpy(), label=label_b, linewidth=2)
    plt.xlabel("Lead day")
    plt.ylabel("Mean ACC across variables")
    plt.title("ACC over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_mean_acc_over_time.png", dpi=180)
    plt.close()


def plot_variable_delta(
    out_dir: Path,
    per_var: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    # RMSE delta (B - A): negative is better for B
    rmse_sorted = per_var.reindex(
        per_var["rmse_diff_b_minus_a"].abs().sort_values(ascending=False).index
    ).head(20)
    plt.figure(figsize=(11, 6.2))
    plt.barh(rmse_sorted["variable"], rmse_sorted["rmse_diff_b_minus_a"])
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel(f"RMSE delta ({label_b} - {label_a})")
    plt.title("Top-20 variable RMSE deltas (absolute)")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_rmse_top20.png", dpi=180)
    plt.close()

    # ACC delta (B - A): positive is better for B
    acc_sorted = per_var.reindex(
        per_var["acc_diff_b_minus_a"].abs().sort_values(ascending=False).index
    ).head(20)
    plt.figure(figsize=(11, 6.2))
    plt.barh(acc_sorted["variable"], acc_sorted["acc_diff_b_minus_a"])
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel(f"ACC delta ({label_b} - {label_a})")
    plt.title("Top-20 variable ACC deltas (absolute)")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_acc_top20.png", dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    run_a = Path(args.run_a_dir).expanduser().resolve()
    run_b = Path(args.run_b_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a, summary_a_json = load_run(run_a)
    df_b, summary_b_json = load_run(run_b)
    df_a, df_b = align_runs(df_a, df_b)

    metrics_a = compute_global_metrics(df_a)
    metrics_b = compute_global_metrics(df_b)

    # prefer summary.json values when present
    if summary_a_json:
        metrics_a["mean_rmse_overall"] = float(summary_a_json.get("mean_rmse_overall", metrics_a["mean_rmse_overall"]))
        metrics_a["mean_acc_overall"] = float(summary_a_json.get("mean_acc_overall", metrics_a["mean_acc_overall"]))
    if summary_b_json:
        metrics_b["mean_rmse_overall"] = float(summary_b_json.get("mean_rmse_overall", metrics_b["mean_rmse_overall"]))
        metrics_b["mean_acc_overall"] = float(summary_b_json.get("mean_acc_overall", metrics_b["mean_acc_overall"]))

    summary_rows = []
    metric_order = [
        "mean_rmse_overall",
        "mean_acc_overall",
        "rmse_day5",
        "acc_day5",
        "rmse_day10",
        "acc_day10",
        "rmse_day15",
        "acc_day15",
    ]
    for m in metric_order:
        a_val = float(metrics_a.get(m, np.nan))
        b_val = float(metrics_b.get(m, np.nan))
        summary_rows.append(
            {
                "metric": m,
                args.label_a: a_val,
                args.label_b: b_val,
                "delta_b_minus_a": b_val - a_val,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_compare.csv", index=False)

    per_var_a = (
        df_a.groupby("variable", as_index=False)[["rmse", "acc"]]
        .mean()
        .rename(columns={"rmse": f"rmse_{args.label_a}", "acc": f"acc_{args.label_a}"})
    )
    per_var_b = (
        df_b.groupby("variable", as_index=False)[["rmse", "acc"]]
        .mean()
        .rename(columns={"rmse": f"rmse_{args.label_b}", "acc": f"acc_{args.label_b}"})
    )

    per_var = per_var_a.merge(per_var_b, on="variable", how="inner")
    per_var["rmse_diff_b_minus_a"] = per_var[f"rmse_{args.label_b}"] - per_var[f"rmse_{args.label_a}"]
    per_var["acc_diff_b_minus_a"] = per_var[f"acc_{args.label_b}"] - per_var[f"acc_{args.label_a}"]
    per_var.to_csv(out_dir / "per_variable_compare.csv", index=False)

    plot_global_curves(out_dir=out_dir, a=df_a, b=df_b, label_a=args.label_a, label_b=args.label_b)
    plot_variable_delta(out_dir=out_dir, per_var=per_var, label_a=args.label_a, label_b=args.label_b)

    report = {
        "run_a_dir": str(run_a),
        "run_b_dir": str(run_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "summary_compare_csv": str(out_dir / "summary_compare.csv"),
        "per_variable_compare_csv": str(out_dir / "per_variable_compare.csv"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("Saved:", out_dir)
    print(" -", out_dir / "summary_compare.csv")
    print(" -", out_dir / "per_variable_compare.csv")
    print(" -", out_dir / "compare_mean_rmse_over_time.png")
    print(" -", out_dir / "compare_mean_acc_over_time.png")
    print(" -", out_dir / "delta_rmse_top20.png")
    print(" -", out_dir / "delta_acc_top20.png")


if __name__ == "__main__":
    main()
