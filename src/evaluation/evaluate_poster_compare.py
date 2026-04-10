#!/usr/bin/env python3
"""
Poster-oriented evaluation and comparison for FuXi checkpoints.

This script is designed to be simple and shareable:
1) Runs `src.evaluation.evaluate_checkpoint` for each checkpoint.
2) Computes isotropic power spectra from saved prediction samples.
3) Builds cross-checkpoint comparison tables and plots for RMSE/ACC.

Outputs under --output-root:
- checkpoint_<name>/  # full evaluate_checkpoint outputs + spectra CSV/PNG
- comparison_summary.csv
- comparison_rmse_acc.png
- comparison_rmse_<var>.png
- comparison_acc_<var>.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


@dataclass
class CheckpointRun:
    name: str
    checkpoint: Path
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run poster-friendly FuXi evaluation + checkpoint comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specs as name=/path/to/best.pt OR /path/to/best.pt",
    )
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--python", type=str, default=sys.executable)

    # Forwarded evaluate_checkpoint args
    p.add_argument("--rollout-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--stats-samples", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16")
    p.add_argument("--save-n-samples", type=int, default=24)
    p.add_argument("--zarr-store", type=str, default=None)
    p.add_argument("--climo-start", type=str, default=None)
    p.add_argument("--climo-end", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Forward --no-heatmaps to evaluate_checkpoint for cleaner outputs",
    )

    # Spectra settings
    p.add_argument(
        "--spectra-vars",
        type=parse_csv_strings,
        default=["geopotential_plev500", "temperature_plev850", "2m_temperature", "surface_pressure"],
        help="Variable names for power spectra plots",
    )
    p.add_argument(
        "--spectra-lead-steps",
        type=parse_csv_ints,
        default=[20, 40, 60],
        help="Lead steps (1-based) used for spectra extraction",
    )

    # Comparison curves
    p.add_argument(
        "--compare-vars",
        type=parse_csv_strings,
        default=["geopotential_plev500", "2m_temperature", "surface_pressure"],
        help="Variables used for across-checkpoint RMSE/ACC lead-time plots",
    )
    return p


def parse_checkpoint_specs(specs: Sequence[str], output_root: Path) -> List[CheckpointRun]:
    runs: List[CheckpointRun] = []
    for spec in specs:
        if "=" in spec:
            raw_name, raw_path = spec.split("=", 1)
            name = sanitize_name(raw_name)
            ckpt = Path(raw_path).expanduser().resolve()
        else:
            ckpt = Path(spec).expanduser().resolve()
            name = sanitize_name(ckpt.parent.name or ckpt.stem)

        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        runs.append(
            CheckpointRun(
                name=name,
                checkpoint=ckpt,
                output_dir=output_root / f"checkpoint_{name}",
            )
        )
    return runs


def build_eval_command(args: argparse.Namespace, run: CheckpointRun) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "src.evaluation.evaluate_checkpoint",
        "--checkpoint",
        str(run.checkpoint),
        "--output-dir",
        str(run.output_dir),
        "--rollout-steps",
        str(args.rollout_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--stats-samples",
        str(args.stats_samples),
        "--save-n-samples",
        str(args.save_n_samples),
        "--device",
        args.device,
        "--amp",
        args.amp,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.zarr_store:
        cmd.extend(["--zarr-store", args.zarr_store])
    if args.climo_start:
        cmd.extend(["--climo-start", args.climo_start])
    if args.climo_end:
        cmd.extend(["--climo-end", args.climo_end])
    if args.no_heatmaps:
        cmd.append("--no-heatmaps")
    return cmd


def run_or_reuse_eval(args: argparse.Namespace, run: CheckpointRun) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run.output_dir / "summary.json"

    if args.skip_existing and summary_file.is_file():
        print(f"[reuse] {run.name}: existing summary found at {summary_file}")
        return

    cmd = build_eval_command(args, run)
    log_path = run.output_dir / "run.log"

    print(f"[eval] {run.name}: {' '.join(cmd)}")
    with log_path.open("w") as logf:
        logf.write("Command:\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=os.environ.copy(),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {run.name}. See {log_path}")


def load_summary(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_metrics_per_lead(path: Path) -> Tuple[List[int], List[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    rows: List[Tuple[str, int, float, float]] = []
    var_to_idx: Dict[str, int] = {}
    step_to_idx: Dict[int, int] = {}
    vars_list: List[str] = []
    step_list: List[int] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            step = int(row["lead_step"])
            rmse = float(row["rmse"])
            acc = float(row["acc"])

            if var not in var_to_idx:
                var_to_idx[var] = len(vars_list)
                vars_list.append(var)
            if step not in step_to_idx:
                step_to_idx[step] = len(step_list)
                step_list.append(step)
            rows.append((var, step, rmse, acc))

    rmse_arr = np.full((len(step_list), len(vars_list)), np.nan, dtype=np.float64)
    acc_arr = np.full((len(step_list), len(vars_list)), np.nan, dtype=np.float64)
    for var, step, rmse, acc in rows:
        si = step_to_idx[step]
        vi = var_to_idx[var]
        rmse_arr[si, vi] = rmse
        acc_arr[si, vi] = acc

    return step_list, vars_list, rmse_arr, acc_arr


def radial_lookup(height: int, width: int) -> Tuple[np.ndarray, int]:
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.rfftfreq(width) * width
    k_mag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    bins = np.rint(k_mag).astype(np.int64)
    n_bins = int(bins.max()) + 1
    return bins.ravel(), n_bins


def isotropic_power(field: np.ndarray, flat_bins: np.ndarray, n_bins: int) -> np.ndarray:
    centered = field.astype(np.float64) - float(np.mean(field))
    fft = np.fft.rfft2(centered)
    power2d = (np.abs(fft) ** 2) / (field.shape[0] * field.shape[1])
    power_flat = power2d.ravel()
    p_sum = np.bincount(flat_bins, weights=power_flat, minlength=n_bins)
    p_cnt = np.bincount(flat_bins, minlength=n_bins)
    return p_sum / np.maximum(p_cnt, 1)


def list_sample_files(sample_dir: Path) -> List[Path]:
    if not sample_dir.is_dir():
        return []
    return sorted(sample_dir.glob("sample_*.npz"))


def build_spectra(
    sample_files: Sequence[Path],
    selected_vars: Sequence[str],
    selected_steps: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Dict[int, Dict[str, np.ndarray]]], List[str]]:
    if not sample_files:
        raise ValueError("No prediction sample files found for spectra computation")

    first = np.load(sample_files[0], allow_pickle=True)
    var_names = [str(v) for v in first["var_names"].tolist()]
    pred0 = first["pred"]  # (S,C,H,W)
    _, _, h, w = pred0.shape
    flat_bins, n_bins = radial_lookup(h, w)
    k = np.arange(n_bins, dtype=np.int64)

    missing: List[str] = []
    var_to_idx: Dict[str, int] = {}
    for name in selected_vars:
        if name in var_names:
            var_to_idx[name] = var_names.index(name)
        else:
            missing.append(name)

    steps = sorted(set(int(s) for s in selected_steps if int(s) >= 1))
    if not steps:
        raise ValueError("selected_steps is empty after filtering")

    spectra: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    counts: Dict[Tuple[str, int], int] = {}
    for var in var_to_idx:
        spectra[var] = {}
        for step in steps:
            spectra[var][step] = {
                "pred": np.zeros((n_bins,), dtype=np.float64),
                "truth": np.zeros((n_bins,), dtype=np.float64),
            }
            counts[(var, step)] = 0

    for path in sample_files:
        data = np.load(path, allow_pickle=True)
        pred = data["pred"]
        truth = data["truth"]
        total_steps = pred.shape[0]

        for step in steps:
            if step > total_steps:
                continue
            si = step - 1
            for var, vi in var_to_idx.items():
                pred_spec = isotropic_power(pred[si, vi], flat_bins, n_bins)
                truth_spec = isotropic_power(truth[si, vi], flat_bins, n_bins)
                spectra[var][step]["pred"] += pred_spec
                spectra[var][step]["truth"] += truth_spec
                counts[(var, step)] += 1

    for var in spectra:
        for step in spectra[var]:
            c = max(counts[(var, step)], 1)
            spectra[var][step]["pred"] /= c
            spectra[var][step]["truth"] /= c

    return k, spectra, missing


def write_spectra_outputs(
    out_dir: Path,
    k: np.ndarray,
    spectra: Dict[str, Dict[int, Dict[str, np.ndarray]]],
) -> None:
    spectra_dir = out_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    for var, by_step in spectra.items():
        csv_path = spectra_dir / f"power_spectra_{sanitize_name(var)}.csv"
        steps = sorted(by_step.keys())

        header = ["wavenumber"]
        for step in steps:
            header.extend([f"truth_step_{step}", f"pred_step_{step}"])

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(k)):
                row = [int(k[i])]
                for step in steps:
                    row.extend([
                        float(by_step[step]["truth"][i]),
                        float(by_step[step]["pred"][i]),
                    ])
                writer.writerow(row)

        # Plot without k=0 to avoid DC dominance.
        nonzero = np.where(k > 0)[0]
        kk = k[nonzero]

        plt.figure(figsize=(8, 5))
        for step in steps:
            truth = by_step[step]["truth"][nonzero]
            pred = by_step[step]["pred"][nonzero]
            plt.plot(kk, truth, linestyle="--", linewidth=1.5, label=f"truth step {step}")
            plt.plot(kk, pred, linestyle="-", linewidth=1.8, label=f"pred step {step}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Wavenumber")
        plt.ylabel("Power")
        plt.title(f"Power spectrum: {var}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(spectra_dir / f"power_spectra_{sanitize_name(var)}.png", dpi=180)
        plt.close()


def write_comparison_summary(output_root: Path, rows: List[dict]) -> None:
    fields = [
        "checkpoint",
        "rollout_steps",
        "eval_samples",
        "mean_rmse_overall",
        "mean_acc_overall",
        "rmse_lead_day_5_mean",
        "acc_lead_day_5_mean",
        "rmse_lead_day_10_mean",
        "acc_lead_day_10_mean",
        "rmse_lead_day_15_mean",
        "acc_lead_day_15_mean",
    ]
    out_csv = output_root / "comparison_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def plot_overall_comparison(output_root: Path, rows: List[dict]) -> None:
    names = [str(r["checkpoint"]) for r in rows]
    rmse = np.array([float(r["mean_rmse_overall"]) for r in rows], dtype=np.float64)
    acc = np.array([float(r["mean_acc_overall"]) for r in rows], dtype=np.float64)
    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    w = 0.38
    ax1.bar(x - w / 2, rmse, width=w, color="#1f77b4", label="mean RMSE")
    ax1.set_ylabel("Mean RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, acc, width=w, color="#d62728", label="mean ACC")
    ax2.set_ylabel("Mean ACC")
    ax2.set_ylim(0.0, max(1.0, float(np.nanmax(acc) * 1.1)))

    lines, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        lines.extend(h)
        labels.extend(l)
    ax1.legend(lines, labels, loc="upper right")

    fig.suptitle("Checkpoint comparison: overall RMSE and ACC")
    fig.tight_layout()
    fig.savefig(output_root / "comparison_rmse_acc.png", dpi=180)
    plt.close(fig)


def plot_variable_curves(
    output_root: Path,
    run_to_metrics: Dict[str, Tuple[List[int], List[str], np.ndarray, np.ndarray]],
    compare_vars: Sequence[str],
) -> None:
    for var_name in compare_vars:
        plt.figure(figsize=(9, 5))
        plotted = False
        for run_name, (steps, vars_list, rmse, _acc) in run_to_metrics.items():
            if var_name not in vars_list:
                continue
            vi = vars_list.index(var_name)
            lead_days = np.array(steps, dtype=np.float64) * 6.0 / 24.0
            plt.plot(lead_days, rmse[:, vi], linewidth=1.8, label=run_name)
            plotted = True
        if plotted:
            plt.xlabel("Lead time (days)")
            plt.ylabel("RMSE")
            plt.title(f"RMSE vs lead time: {var_name}")
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_root / f"comparison_rmse_{sanitize_name(var_name)}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        plotted = False
        for run_name, (steps, vars_list, _rmse, acc) in run_to_metrics.items():
            if var_name not in vars_list:
                continue
            vi = vars_list.index(var_name)
            lead_days = np.array(steps, dtype=np.float64) * 6.0 / 24.0
            plt.plot(lead_days, acc[:, vi], linewidth=1.8, label=run_name)
            plotted = True
        if plotted:
            plt.xlabel("Lead time (days)")
            plt.ylabel("ACC")
            plt.title(f"ACC vs lead time: {var_name}")
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_root / f"comparison_acc_{sanitize_name(var_name)}.png", dpi=180)
        plt.close()


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs = parse_checkpoint_specs(args.checkpoints, output_root)

    comparison_rows: List[dict] = []
    run_to_metrics: Dict[str, Tuple[List[int], List[str], np.ndarray, np.ndarray]] = {}

    for run in runs:
        run_or_reuse_eval(args, run)

        summary_path = run.output_dir / "summary.json"
        metrics_path = run.output_dir / "metrics_per_lead.csv"
        sample_dir = run.output_dir / "prediction_samples"

        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary.json for {run.name}: {summary_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics_per_lead.csv for {run.name}: {metrics_path}")

        summary = load_summary(summary_path)
        comparison_rows.append(
            {
                "checkpoint": run.name,
                "rollout_steps": summary.get("rollout_steps"),
                "eval_samples": summary.get("eval_samples"),
                "mean_rmse_overall": summary.get("mean_rmse_overall"),
                "mean_acc_overall": summary.get("mean_acc_overall"),
                "rmse_lead_day_5_mean": summary.get("rmse_lead_day_5_mean"),
                "acc_lead_day_5_mean": summary.get("acc_lead_day_5_mean"),
                "rmse_lead_day_10_mean": summary.get("rmse_lead_day_10_mean"),
                "acc_lead_day_10_mean": summary.get("acc_lead_day_10_mean"),
                "rmse_lead_day_15_mean": summary.get("rmse_lead_day_15_mean"),
                "acc_lead_day_15_mean": summary.get("acc_lead_day_15_mean"),
            }
        )

        run_to_metrics[run.name] = load_metrics_per_lead(metrics_path)

        sample_files = list_sample_files(sample_dir)
        if not sample_files:
            print(f"[warn] {run.name}: no prediction samples found under {sample_dir}. Skip spectra.")
            continue

        k, spectra, missing = build_spectra(
            sample_files=sample_files,
            selected_vars=args.spectra_vars,
            selected_steps=args.spectra_lead_steps,
        )
        if missing:
            print(f"[warn] {run.name}: missing spectra vars: {missing}")
        write_spectra_outputs(run.output_dir, k, spectra)
        print(f"[spectra] {run.name}: wrote spectra outputs under {run.output_dir / 'spectra'}")

    comparison_rows.sort(key=lambda x: str(x["checkpoint"]))
    write_comparison_summary(output_root, comparison_rows)
    plot_overall_comparison(output_root, comparison_rows)
    plot_variable_curves(output_root, run_to_metrics, args.compare_vars)

    print("=" * 88)
    print("Poster comparison complete")
    print(f"Output root: {output_root}")
    print(f"Runs       : {[r.name for r in runs]}")
    print("=" * 88)


if __name__ == "__main__":
    main()
