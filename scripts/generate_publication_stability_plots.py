#!/usr/bin/env python3
"""
Generate publication-style stability plots from a completed evaluation run.

Input folder must contain metrics_per_lead.csv produced by evaluate_checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("metrics_per_lead.csv is empty")

    variables = sorted({r["variable"] for r in rows})
    lead_steps = sorted({int(r["lead_step"]) for r in rows})

    var_to_idx = {v: i for i, v in enumerate(variables)}
    step_to_idx = {s: i for i, s in enumerate(lead_steps)}

    rmse = np.full((len(lead_steps), len(variables)), np.nan, dtype=np.float64)
    acc = np.full((len(lead_steps), len(variables)), np.nan, dtype=np.float64)

    for r in rows:
        s = step_to_idx[int(r["lead_step"])]
        v = var_to_idx[r["variable"]]
        rmse[s, v] = float(r["rmse"])
        acc[s, v] = float(r["acc"])

    lead_days = np.asarray(lead_steps, dtype=np.float64) * 6.0 / 24.0
    return variables, lead_steps, lead_days, rmse, acc


def choose_highlight_variables(variables: List[str]) -> List[str]:
    preferred = [
        "2m_temperature",
        "surface_pressure",
        "geopotential_plev500",
        "temperature_plev850",
        "u_component_of_wind_plev850",
        "v_component_of_wind_plev850",
    ]
    chosen = [v for v in preferred if v in variables]
    if len(chosen) >= 4:
        return chosen[:6]
    return variables[: min(6, len(variables))]


def plot_global_panel(out_dir: Path, lead_days: np.ndarray, rmse: np.ndarray, acc: np.ndarray) -> None:
    rmse_mean = np.nanmean(rmse, axis=1)
    rmse_q25 = np.nanpercentile(rmse, 25, axis=1)
    rmse_q75 = np.nanpercentile(rmse, 75, axis=1)

    acc_mean = np.nanmean(acc, axis=1)
    acc_q25 = np.nanpercentile(acc, 25, axis=1)
    acc_q75 = np.nanpercentile(acc, 75, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].plot(lead_days, rmse_mean, color="#1f77b4", linewidth=2.2, label="Mean RMSE")
    axes[0].fill_between(lead_days, rmse_q25, rmse_q75, color="#1f77b4", alpha=0.2, label="IQR")
    axes[0].set_title("RMSE Growth Over Lead Time")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(lead_days, acc_mean, color="#d62728", linewidth=2.2, label="Mean ACC")
    axes[1].fill_between(lead_days, acc_q25, acc_q75, color="#d62728", alpha=0.2, label="IQR")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("ACC Decay Over Lead Time")
    axes[1].set_xlabel("Lead Time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.savefig(out_dir / "publication_global_rmse_acc_panel.png", dpi=250)
    fig.savefig(out_dir / "publication_global_rmse_acc_panel.pdf")
    plt.close(fig)


def plot_normalized_skill(out_dir: Path, lead_days: np.ndarray, rmse: np.ndarray, acc: np.ndarray) -> None:
    rmse_mean = np.nanmean(rmse, axis=1)
    acc_mean = np.nanmean(acc, axis=1)

    rmse_norm = rmse_mean / max(rmse_mean[0], 1e-12)
    acc_norm = acc_mean / max(abs(acc_mean[0]), 1e-12)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(lead_days, rmse_norm, color="#1f77b4", linewidth=2.2, label="RMSE / RMSE(day0)")
    ax.plot(lead_days, acc_norm, color="#d62728", linewidth=2.2, label="ACC / ACC(day0)")
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Normalized Skill Drift")
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel("Normalized Value")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.savefig(out_dir / "publication_normalized_skill_drift.png", dpi=250)
    fig.savefig(out_dir / "publication_normalized_skill_drift.pdf")
    plt.close(fig)


def plot_variable_curves(out_dir: Path, variables: List[str], lead_days: np.ndarray, rmse: np.ndarray, acc: np.ndarray) -> None:
    highlights = choose_highlight_variables(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), constrained_layout=True)

    for v in highlights:
        i = var_to_idx[v]
        axes[0].plot(lead_days, rmse[:, i], linewidth=1.8, label=v)
    axes[0].set_title("Selected Variables: RMSE")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    for v in highlights:
        i = var_to_idx[v]
        axes[1].plot(lead_days, acc[:, i], linewidth=1.8, label=v)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("Selected Variables: ACC")
    axes[1].set_xlabel("Lead Time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(out_dir / "publication_selected_variable_curves.png", dpi=250)
    fig.savefig(out_dir / "publication_selected_variable_curves.pdf")
    plt.close(fig)


def write_short_report(out_dir: Path, lead_days: np.ndarray, rmse: np.ndarray, acc: np.ndarray) -> None:
    rmse_mean = np.nanmean(rmse, axis=1)
    acc_mean = np.nanmean(acc, axis=1)

    text = []
    text.append("Publication Stability Snapshot")
    text.append("")
    text.append(f"Lead time max (days): {lead_days[-1]:.2f}")
    text.append(f"Mean RMSE at first step: {rmse_mean[0]:.6f}")
    text.append(f"Mean RMSE at last step: {rmse_mean[-1]:.6f}")
    text.append(f"Mean RMSE growth: {(rmse_mean[-1]-rmse_mean[0]):.6f}")
    text.append(f"Mean ACC at first step: {acc_mean[0]:.6f}")
    text.append(f"Mean ACC at last step: {acc_mean[-1]:.6f}")
    text.append(f"Mean ACC drop: {(acc_mean[0]-acc_mean[-1]):.6f}")

    tail_n = min(10, len(acc_mean))
    tail_acc = float(np.mean(acc_mean[-tail_n:]))
    text.append(f"Mean ACC tail(last {tail_n}): {tail_acc:.6f}")

    if tail_acc > 0.3:
        text.append("Interpretation: Stable and skillful at long lead times.")
    elif tail_acc > 0.0:
        text.append("Interpretation: Degrading but still useful.")
    elif tail_acc > -0.2:
        text.append("Interpretation: Mostly climatology-like behavior.")
    else:
        text.append("Interpretation: Unstable or phase-flipped behavior.")

    (out_dir / "publication_stability_report.txt").write_text("\n".join(text) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication stability plots")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to results/stability_run_* folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_csv = run_dir / "metrics_per_lead.csv"

    variables, _steps, lead_days, rmse, acc = load_metrics(metrics_csv)
    plot_global_panel(run_dir, lead_days, rmse, acc)
    plot_normalized_skill(run_dir, lead_days, rmse, acc)
    plot_variable_curves(run_dir, variables, lead_days, rmse, acc)
    write_short_report(run_dir, lead_days, rmse, acc)

    print("Done. Added publication plots to:")
    print(run_dir)
    print(run_dir / "publication_global_rmse_acc_panel.png")
    print(run_dir / "publication_normalized_skill_drift.png")
    print(run_dir / "publication_selected_variable_curves.png")
    print(run_dir / "publication_stability_report.txt")


if __name__ == "__main__":
    main()
