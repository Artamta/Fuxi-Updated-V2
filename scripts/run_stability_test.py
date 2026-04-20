#!/usr/bin/env python3
"""
One clean stability script for emb_768.

Outputs:
- summary.json
- metrics_per_lead.csv
- stability_summary.json
- mean_rmse_acc_climatology_plot.(png|pdf)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def find_768_checkpoints(root: Path) -> List[Path]:
    ckpts: List[Path] = []
    for base in (root / "checkpoints", root / "results"):
        if base.exists():
            ckpts.extend(base.rglob("best.pt"))
    return sorted([p.resolve() for p in ckpts if "emb_768" in str(p)])


def choose_checkpoint(root: Path, explicit: Optional[str], random_mode: bool) -> Path:
    if explicit:
        ckpt = Path(explicit).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    candidates = find_768_checkpoints(root)
    if not candidates:
        raise FileNotFoundError("No emb_768 best.pt checkpoint found under checkpoints/ or results/")
    if random_mode:
        return random.choice(candidates)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def choose_climo_cache(root: Path, explicit: Optional[str], use_saved: bool) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Climatology cache not found: {p}")
        return p
    if not use_saved:
        return None

    local = sorted((root / "results" / "climatology_cache" / "emb_768").glob("climo_*.npz"))
    if not local:
        return None
    return max(local, key=lambda p: p.stat().st_mtime).resolve()


def resolve_eval_climo_cache(
    root: Path,
    outdir: Path,
    selected_cache: Optional[Path],
    use_saved: bool,
) -> Path:
    if selected_cache is not None:
        return selected_cache

    if use_saved:
        cache_dir = root / "results" / "climatology_cache" / "emb_768"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "climo_auto_emb768.npz"

    # Force a writable, run-local cache path when user disables saved cache reuse.
    return outdir / "climo_cache_autogen.npz"


def run_eval(root: Path, ckpt: Path, outdir: Path, args: argparse.Namespace, climo_cache: Optional[Path]) -> None:
    device_arg = args.device
    if args.gpu_index is not None and device_arg == "auto":
        device_arg = "cuda"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluation.evaluate_checkpoint",
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(outdir),
        "--device",
        device_arg,
        "--amp",
        args.amp,
        "--rollout-steps",
        str(args.rollout_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-samples",
        str(args.max_samples),
        "--stats-samples",
        str(args.stats_samples),
        "--save-n-samples",
        str(args.save_n_samples),
        "--plot-vars",
        args.plot_vars,
    ]
    if climo_cache is not None:
        cmd.extend(["--climo-cache", str(climo_cache)])

    print("Running evaluation:")
    print(" ".join(cmd))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_index))
        print(f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} (requested gpu index)")
    subprocess.run(cmd, cwd=root, env=env, check=True)


def load_mean_curves(metrics_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")

    by_step: Dict[int, Dict[str, List[float]]] = {}
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["lead_step"])
            by_step.setdefault(step, {"rmse": [], "acc": []})
            by_step[step]["rmse"].append(float(row["rmse"]))
            by_step[step]["acc"].append(float(row["acc"]))

    steps = sorted(by_step.keys())
    lead_days = np.array(steps, dtype=np.float64) * 6.0 / 24.0
    rmse_mean = np.array([np.mean(by_step[s]["rmse"]) for s in steps], dtype=np.float64)
    acc_mean = np.array([np.mean(by_step[s]["acc"]) for s in steps], dtype=np.float64)
    return lead_days, rmse_mean, acc_mean


def climatology_rmse_baseline(sample_dir: Path, climo_cache: Optional[Path]) -> Optional[np.ndarray]:
    if climo_cache is None or not climo_cache.is_file() or not sample_dir.exists():
        return None

    with np.load(climo_cache) as cdata:
        if "climatology" not in cdata:
            return None
        clim = cdata["climatology"].astype(np.float32)

    curves: List[np.ndarray] = []
    for sample in sorted(sample_dir.glob("sample_*.npz")):
        with np.load(sample, allow_pickle=True) as d:
            truth = d["truth"].astype(np.float32)
        err = clim[None, ...] - truth
        rmse_curve = np.sqrt(np.mean(err * err, axis=(1, 2, 3)))
        curves.append(rmse_curve)

    if not curves:
        return None
    return np.mean(np.stack(curves, axis=0), axis=0)


def save_plot(outdir: Path, lead_days: np.ndarray, rmse_mean: np.ndarray, acc_mean: np.ndarray, rmse_climo: Optional[np.ndarray]) -> None:
    fig, ax1 = plt.subplots(figsize=(10.5, 5.3))

    model_rmse = ax1.plot(lead_days, rmse_mean, color="#1f77b4", linewidth=2.3, label="Mean RMSE (model)")
    if rmse_climo is not None and len(rmse_climo) == len(lead_days):
        climo_rmse = ax1.plot(
            lead_days,
            rmse_climo,
            color="#2ca02c",
            linewidth=2.0,
            linestyle="--",
            label="RMSE (climatology baseline)",
        )
    else:
        climo_rmse = []
    ax1.set_xlabel("Lead Time (days)")
    ax1.set_ylabel("RMSE", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    model_acc = ax2.plot(lead_days, acc_mean, color="#d62728", linewidth=2.3, label="Mean ACC (model)")
    ax2.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.6, alpha=0.9)
    ax2.set_ylabel("ACC", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    handles = model_rmse + climo_rmse + model_acc + [plt.Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.6)]
    labels = [h.get_label() for h in (model_rmse + climo_rmse + model_acc)] + ["ACC climatology reference (0)"]
    ax1.legend(handles, labels, loc="best", frameon=False)

    plt.title("emb_768 Stability: Mean RMSE / Mean ACC vs Lead Time")
    fig.tight_layout()
    fig.savefig(outdir / "mean_rmse_acc_climatology_plot.png", dpi=260)
    fig.savefig(outdir / "mean_rmse_acc_climatology_plot.pdf")
    plt.close(fig)


def save_summary(
    outdir: Path,
    ckpt: Path,
    climo_cache: Optional[Path],
    lead_days: np.ndarray,
    rmse_mean: np.ndarray,
    acc_mean: np.ndarray,
    rmse_climo: Optional[np.ndarray],
) -> None:
    summary = {
        "checkpoint": str(ckpt),
        "climo_cache_used": str(climo_cache) if climo_cache else None,
        "lead_days_max": float(lead_days[-1]),
        "mean_rmse_step1": float(rmse_mean[0]),
        "mean_rmse_last": float(rmse_mean[-1]),
        "mean_rmse_growth": float(rmse_mean[-1] - rmse_mean[0]),
        "mean_acc_step1": float(acc_mean[0]),
        "mean_acc_last": float(acc_mean[-1]),
        "mean_acc_drop": float(acc_mean[0] - acc_mean[-1]),
        "climatology_rmse_last": float(rmse_climo[-1]) if rmse_climo is not None else None,
    }
    (outdir / "stability_summary.json").write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple emb_768 stability run + plot")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--random", type=parse_bool, default=False, help="true/false, pick random emb_768 checkpoint")
    parser.add_argument("--use-saved-climatology", type=parse_bool, default=True, help="true/false")
    parser.add_argument("--climo-cache", type=str, default=None, help="Optional explicit climatology cache .npz")

    parser.add_argument("--rollout-steps", type=int, default=120, help="120 steps = 30 days")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--save-n-samples", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Optional physical GPU index (e.g. 2 for A100 GPU2). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--plot-vars",
        type=str,
        default="2m_temperature,geopotential_plev500,u_component_of_wind_plev250",
        help="Comma-separated variables for evaluator plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    ckpt = choose_checkpoint(root, args.checkpoint, bool(args.random))
    selected_climo_cache = choose_climo_cache(root, args.climo_cache, bool(args.use_saved_climatology))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = root / "results" / f"stability_run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    climo_cache = resolve_eval_climo_cache(
        root=root,
        outdir=outdir,
        selected_cache=selected_climo_cache,
        use_saved=bool(args.use_saved_climatology),
    )

    if args.rollout_steps >= 240 and args.batch_size > 1:
        print(
            f"Long-horizon rollout detected ({args.rollout_steps} steps); "
            f"forcing batch_size=1 for stability (requested {args.batch_size})."
        )
        args.batch_size = 1

    print(f"Repository root: {root}")
    print(f"Selected checkpoint: {ckpt}")
    print(f"Climatology cache: {climo_cache}")
    if args.gpu_index is not None:
        print(f"Requested GPU index: {args.gpu_index}")
    print(f"Output directory: {outdir}")

    run_eval(root, ckpt, outdir, args, climo_cache)

    lead_days, rmse_mean, acc_mean = load_mean_curves(outdir / "metrics_per_lead.csv")
    rmse_climo = climatology_rmse_baseline(outdir / "prediction_samples", climo_cache)

    save_plot(outdir, lead_days, rmse_mean, acc_mean, rmse_climo)
    save_summary(outdir, ckpt, climo_cache, lead_days, rmse_mean, acc_mean, rmse_climo)

    print("Done.")
    print("Main outputs:")
    print(f"  {outdir / 'summary.json'}")
    print(f"  {outdir / 'metrics_per_lead.csv'}")
    print(f"  {outdir / 'stability_summary.json'}")
    print(f"  {outdir / 'mean_rmse_acc_climatology_plot.png'}")


if __name__ == "__main__":
    main()
