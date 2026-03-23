#!/usr/bin/env python3
"""
Run a long-rollout stability test for a FuXi checkpoint.

Simple behavior:
1) Pick checkpoint:
    - default: best checkpoint for emb_768
    - random mode: pick random checkpoint if --random true
    - override: --checkpoint /path/to/file.pt
2) Reuse saved climatology:
    - tries local results/climatology_cache/emb_<dim>/climo_*.npz
    - if missing and --copy-climo-from-prev true, copies from fuxi_paper_prev
3) Run src.evaluation.evaluate_checkpoint.
4) Write compact stability outputs:
    - stability_summary.json
    - stability_rmse_acc_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PREV_REPO = Path("/home/raj.ayush/fuxi-final/fuxi_paper_prev")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_checkpoints(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for base in (root / "checkpoints", root / "results"):
        if not base.exists():
            continue
        candidates.extend(base.rglob("best.pt"))
        candidates.extend(base.rglob("last.pt"))
    # Stable ordering for reproducibility in latest/random behavior
    return sorted({p.resolve() for p in candidates})


def choose_checkpoint(mode: str, all_ckpts: List[Path], explicit: str | None) -> Path:
    if explicit:
        ckpt = Path(explicit).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if not all_ckpts:
        raise FileNotFoundError("No checkpoints found under checkpoints/ or results/")

    if mode == "latest":
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)
    if mode == "random":
        return random.choice(all_ckpts)

    raise ValueError(f"Unsupported mode: {mode}")


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def infer_embedding_dim(ckpt: Path) -> int | None:
    m = re.search(r"emb_(\d+)", str(ckpt))
    if m:
        return int(m.group(1))
    return None


def find_local_climo_files(root: Path, emb_dim: int) -> List[Path]:
    climo_dir = root / "results" / "climatology_cache" / f"emb_{emb_dim}"
    if not climo_dir.exists():
        return []
    return sorted(climo_dir.glob("climo_*.npz"))


def copy_climo_from_prev_repo(root: Path, prev_repo: Path, emb_dim: int) -> List[Path]:
    src_dir = prev_repo / "Models_paper" / "pretrain" / f"emb_{emb_dim}"
    if not src_dir.exists():
        return []

    src_files = sorted(src_dir.glob("climo_*.npz"))
    if not src_files:
        return []

    dest_dir = root / "results" / "climatology_cache" / f"emb_{emb_dim}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for src in src_files:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def choose_checkpoint_simple(
    all_ckpts: List[Path],
    explicit: str | None,
    random_mode: bool,
    preferred_embedding: int,
) -> Path:
    if explicit:
        ckpt = Path(explicit).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if not all_ckpts:
        raise FileNotFoundError("No checkpoints found under checkpoints/ or results/")

    if random_mode:
        return random.choice(all_ckpts)

    pref = [p for p in all_ckpts if p.name == "best.pt" and f"emb_{preferred_embedding}" in str(p)]
    if pref:
        return max(pref, key=lambda p: p.stat().st_mtime)

    best_any = [p for p in all_ckpts if p.name == "best.pt"]
    if best_any:
        return max(best_any, key=lambda p: p.stat().st_mtime)

    return max(all_ckpts, key=lambda p: p.stat().st_mtime)


def resolve_climo_cache(args: argparse.Namespace, root: Path, ckpt: Path, outdir: Path) -> Path | None:
    if args.climo_cache:
        p = Path(args.climo_cache).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Climatology cache not found: {p}")
        return p

    if not args.use_saved_climo:
        return None

    emb_dim = infer_embedding_dim(ckpt)
    if emb_dim is None:
        return None

    local = find_local_climo_files(root, emb_dim)
    if not local and args.copy_climo_from_prev:
        prev_repo = Path(args.prev_repo).expanduser().resolve()
        local = copy_climo_from_prev_repo(root, prev_repo, emb_dim)

    if not local:
        return None

    chosen = max(local, key=lambda p: p.stat().st_mtime)
    snapshot = outdir / chosen.name
    if not snapshot.exists():
        shutil.copy2(chosen, snapshot)
    return snapshot


def run_eval(root: Path, ckpt: Path, outdir: Path, args: argparse.Namespace, climo_cache: Path | None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate_checkpoint",
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(outdir),
        "--device",
        args.device,
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
    subprocess.run(cmd, cwd=root, check=True)


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
    rmse = np.array([np.mean(by_step[s]["rmse"]) for s in steps], dtype=np.float64)
    acc = np.array([np.mean(by_step[s]["acc"]) for s in steps], dtype=np.float64)
    lead_days = np.array(steps, dtype=np.float64) * 6.0 / 24.0
    return lead_days, rmse, acc


def diagnose_stability(acc: np.ndarray) -> str:
    tail_n = min(10, len(acc))
    tail = float(np.mean(acc[-tail_n:]))
    if tail > 0.3:
        return "stable_skillful"
    if tail > 0.0:
        return "degrading_but_reasonable"
    if tail > -0.2:
        return "climatology_like_or_weak_skill"
    return "unstable_or_phase_flipped"


def write_outputs(
    outdir: Path,
    ckpt: Path,
    lead_days: np.ndarray,
    rmse: np.ndarray,
    acc: np.ndarray,
    climo_cache: Path | None,
) -> None:
    rmse_growth = float(rmse[-1] - rmse[0])
    acc_drop = float(acc[0] - acc[-1])
    tail_n = min(10, len(acc))
    acc_tail = float(np.mean(acc[-tail_n:]))

    summary = {
        "checkpoint": str(ckpt),
        "climo_cache_used": str(climo_cache) if climo_cache is not None else None,
        "lead_days_max": float(lead_days[-1]),
        "mean_rmse_step1": float(rmse[0]),
        "mean_rmse_last": float(rmse[-1]),
        "mean_rmse_growth": rmse_growth,
        "mean_acc_step1": float(acc[0]),
        "mean_acc_last": float(acc[-1]),
        "mean_acc_drop": acc_drop,
        "mean_acc_tail_last10": acc_tail,
        "stability_label": diagnose_stability(acc),
        "note": "Tail ACC near 0 suggests climatology-like behavior; strongly negative suggests unstable forecast.",
    }
    (outdir / "stability_summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(10, 4.8))
    plt.plot(lead_days, rmse, label="Mean RMSE", linewidth=2)
    plt.plot(lead_days, acc, label="Mean ACC", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Lead time (days)")
    plt.ylabel("Metric value")
    plt.title("Long-rollout stability: RMSE and ACC over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stability_rmse_acc_curve.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long-rollout checkpoint stability test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--random", type=parse_bool, default=False, help="true/false")
    parser.add_argument("--preferred-embedding", type=int, default=768, help="Used when random=false")
    parser.add_argument("--mode", choices=["random", "latest"], default=None, help="Compatibility; random mode if set to random")

    parser.add_argument("--use-saved-climo", type=parse_bool, default=True, help="true/false")
    parser.add_argument("--copy-climo-from-prev", type=parse_bool, default=True, help="true/false")
    parser.add_argument("--prev-repo", type=str, default=str(DEFAULT_PREV_REPO))
    parser.add_argument("--climo-cache", type=str, default=None, help="Explicit climo .npz path")

    parser.add_argument("--rollout-steps", type=int, default=120, help="120 steps = 30 days")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--save-n-samples", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
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
    all_ckpts = find_checkpoints(root)

    random_mode = bool(args.random) or args.mode == "random"
    ckpt = choose_checkpoint_simple(
        all_ckpts=all_ckpts,
        explicit=args.checkpoint,
        random_mode=random_mode,
        preferred_embedding=int(args.preferred_embedding),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = root / "results" / f"stability_run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    climo_cache = resolve_climo_cache(args, root, ckpt, outdir)

    print(f"Repository root: {root}")
    print(f"Selected checkpoint: {ckpt}")
    print(f"Random mode: {random_mode}")
    print(f"Climatology cache: {climo_cache if climo_cache else 'auto-compute'}")
    print(f"Output directory: {outdir}")

    run_eval(root, ckpt, outdir, args, climo_cache)

    lead_days, rmse, acc = load_mean_curves(outdir / "metrics_per_lead.csv")
    write_outputs(outdir, ckpt, lead_days, rmse, acc, climo_cache)

    print("Done.")
    print(f"Main outputs:")
    print(f"  {outdir / 'summary.json'}")
    print(f"  {outdir / 'metrics_per_lead.csv'}")
    print(f"  {outdir / 'stability_summary.json'}")
    print(f"  {outdir / 'stability_rmse_acc_curve.png'}")


if __name__ == "__main__":
    main()
