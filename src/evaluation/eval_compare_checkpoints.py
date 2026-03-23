#!/usr/bin/env python3
"""
Compare multiple FuXi checkpoints with unified evaluation outputs.

What this script does:
1. Runs evaluate_checkpoint.py for each checkpoint.
2. Runs checkpoints in parallel across multiple GPUs (e.g., 2 GPUs).
3. Saves per-checkpoint outputs under:
   eval/checkpoint_<name>/
4. Builds cross-checkpoint tables/plots:
   - ACC mean over lead time (global)
   - RMSE mean over lead time (global)
   - RMSE growth vs lead time
   - RMSE/ACC mean per variable per checkpoint
   - Prediction maps: single-step vs multi-step + error growth map
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare evaluation for multiple FuXi checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specs: name=/path/to/best.pt OR /path/to/best.pt",
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="/home/raj.ayush/fuxi_advanced/fuxi_paper/eval",
        help="Root output directory for comparison outputs",
    )
    parser.add_argument("--eval-script", type=str, default="evaluate_checkpoint.py")
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids for parallel runs")
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")

    parser.add_argument("--rollout-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--save-n-samples", type=int, default=1)

    parser.add_argument("--climo-start", type=str, default=None)
    parser.add_argument("--climo-end", type=str, default=None)
    parser.add_argument("--climo-cache-dir", type=str, default=None)

    parser.add_argument(
        "--map-variable",
        type=str,
        default="2m_temperature",
        help="Variable name used for single-step/multi-step map visualizations",
    )
    parser.add_argument(
        "--map-lead-step",
        type=int,
        default=20,
        help="Lead step for multi-step map (e.g., 20 -> day 5)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


@dataclass
class CheckpointJob:
    name: str
    checkpoint: Path
    output_dir: Path
    assigned_gpu: Optional[str]


@dataclass
class JobResult:
    name: str
    checkpoint: Path
    output_dir: Path
    log_path: Path
    returncode: int
    assigned_gpu: Optional[str]
    duration_sec: float


@dataclass
class MetricsMatrix:
    lead_steps: List[int]
    variables: List[str]
    rmse: np.ndarray  # (S, C)
    acc: np.ndarray   # (S, C)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def parse_checkpoint_specs(specs: Sequence[str], eval_root: Path) -> List[CheckpointJob]:
    jobs: List[CheckpointJob] = []
    for spec in specs:
        if "=" in spec:
            name_raw, path_raw = spec.split("=", 1)
            name = sanitize_name(name_raw)
            ckpt = Path(path_raw).expanduser().resolve()
        else:
            ckpt = Path(spec).expanduser().resolve()
            name = sanitize_name(ckpt.parent.name or ckpt.stem)

        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        out_dir = eval_root / f"checkpoint_{name}"
        jobs.append(
            CheckpointJob(
                name=name,
                checkpoint=ckpt,
                output_dir=out_dir,
                assigned_gpu=None,
            )
        )
    return jobs


def parse_gpu_list(gpus_csv: str) -> List[str]:
    return [g.strip() for g in gpus_csv.split(",") if g.strip()]


def build_eval_command(
    args: argparse.Namespace,
    job: CheckpointJob,
    climo_cache_dir: Optional[Path],
) -> List[str]:
    cmd = [
        args.python,
        str(Path(args.eval_script)),
        "--checkpoint",
        str(job.checkpoint),
        "--output-dir",
        str(job.output_dir),
        "--rollout-steps",
        str(args.rollout_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--amp",
        args.amp,
        "--stats-samples",
        str(args.stats_samples),
        "--save-n-samples",
        str(args.save_n_samples),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.climo_start is not None:
        cmd.extend(["--climo-start", args.climo_start])
    if args.climo_end is not None:
        cmd.extend(["--climo-end", args.climo_end])
    if climo_cache_dir is not None:
        cache_path = climo_cache_dir / f"climo_{job.name}.npz"
        cmd.extend(["--climo-cache", str(cache_path)])
    return cmd


def run_single_eval(
    args: argparse.Namespace,
    job: CheckpointJob,
    gpu_id: Optional[str],
    climo_cache_dir: Optional[Path],
) -> JobResult:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "run.log"
    cmd = build_eval_command(args, job, climo_cache_dir=climo_cache_dir)

    env = os.environ.copy()
    if args.device in ("cuda", "auto") and gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    start = time.time()
    with log_path.open("w") as logf:
        logf.write("Command:\n")
        logf.write(" ".join(cmd) + "\n\n")
        if gpu_id is not None:
            logf.write(f"Assigned GPU: {gpu_id}\n\n")
        logf.flush()

        proc = subprocess.run(
            cmd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.time() - start
    return JobResult(
        name=job.name,
        checkpoint=job.checkpoint,
        output_dir=job.output_dir,
        log_path=log_path,
        returncode=proc.returncode,
        assigned_gpu=gpu_id,
        duration_sec=elapsed,
    )


def run_jobs(args: argparse.Namespace, jobs: List[CheckpointJob], gpu_ids: List[str]) -> List[JobResult]:
    climo_cache_dir = Path(args.climo_cache_dir).expanduser().resolve() if args.climo_cache_dir else None
    if climo_cache_dir is not None:
        climo_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu" or len(gpu_ids) == 0:
        gpu_assignments = [None] * len(jobs)
        workers = 1
    else:
        gpu_assignments = [gpu_ids[i % len(gpu_ids)] for i in range(len(jobs))]
        workers = min(len(gpu_ids), len(jobs))

    for i, job in enumerate(jobs):
        job.assigned_gpu = gpu_assignments[i]

    if args.dry_run:
        print("Dry run. Commands:")
        for i, job in enumerate(jobs):
            cmd = build_eval_command(args, job, climo_cache_dir=climo_cache_dir)
            gpu = gpu_assignments[i]
            prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
            print(prefix + " ".join(cmd))
        return []

    results: List[JobResult] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        future_to_job = {}
        for i, job in enumerate(jobs):
            future = ex.submit(run_single_eval, args, job, gpu_assignments[i], climo_cache_dir)
            future_to_job[future] = job

        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            status = "OK" if result.returncode == 0 else "FAILED"
            gpu_txt = f"gpu={result.assigned_gpu}" if result.assigned_gpu is not None else "cpu"
            print(
                f"[{status}] {result.name} ({gpu_txt}) "
                f"time={result.duration_sec/60.0:.1f} min log={result.log_path}"
            )

    results.sort(key=lambda x: x.name)
    return results


def read_metrics_per_lead(path: Path) -> MetricsMatrix:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    rows: List[Tuple[str, int, float, float]] = []
    var_to_idx: Dict[str, int] = {}
    lead_to_idx: Dict[int, int] = {}
    variables: List[str] = []
    lead_steps: List[int] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            step = int(row["lead_step"])
            rmse_val = float(row["rmse"])
            acc_val = float(row["acc"])

            if var not in var_to_idx:
                var_to_idx[var] = len(variables)
                variables.append(var)
            if step not in lead_to_idx:
                lead_to_idx[step] = len(lead_steps)
                lead_steps.append(step)
            rows.append((var, step, rmse_val, acc_val))

    rmse = np.full((len(lead_steps), len(variables)), np.nan, dtype=np.float64)
    acc = np.full((len(lead_steps), len(variables)), np.nan, dtype=np.float64)
    for var, step, rmse_val, acc_val in rows:
        si = lead_to_idx[step]
        vi = var_to_idx[var]
        rmse[si, vi] = rmse_val
        acc[si, vi] = acc_val

    return MetricsMatrix(
        lead_steps=lead_steps,
        variables=variables,
        rmse=rmse,
        acc=acc,
    )


def align_matrix(matrix: MetricsMatrix, ref_vars: Sequence[str], ref_steps: Sequence[int]) -> MetricsMatrix:
    ref_var_to_idx = {v: i for i, v in enumerate(ref_vars)}
    ref_step_to_idx = {s: i for i, s in enumerate(ref_steps)}

    rmse = np.full((len(ref_steps), len(ref_vars)), np.nan, dtype=np.float64)
    acc = np.full((len(ref_steps), len(ref_vars)), np.nan, dtype=np.float64)

    cur_var_to_idx = {v: i for i, v in enumerate(matrix.variables)}
    cur_step_to_idx = {s: i for i, s in enumerate(matrix.lead_steps)}

    for step in ref_steps:
        if step not in cur_step_to_idx:
            continue
        for var in ref_vars:
            if var not in cur_var_to_idx:
                continue
            si_ref = ref_step_to_idx[step]
            vi_ref = ref_var_to_idx[var]
            si_cur = cur_step_to_idx[step]
            vi_cur = cur_var_to_idx[var]
            rmse[si_ref, vi_ref] = matrix.rmse[si_cur, vi_cur]
            acc[si_ref, vi_ref] = matrix.acc[si_cur, vi_cur]

    return MetricsMatrix(
        lead_steps=list(ref_steps),
        variables=list(ref_vars),
        rmse=rmse,
        acc=acc,
    )


def load_prediction_sample(sample_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample file: {sample_path}")
    data = np.load(sample_path, allow_pickle=True)
    pred = data["pred"]    # (S, C, H, W)
    truth = data["truth"]  # (S, C, H, W)
    var_raw = data["var_names"]
    var_names = [str(v) for v in var_raw.tolist()]
    return pred, truth, var_names


def choose_var_index(var_names: Sequence[str], requested: str) -> int:
    if requested in var_names:
        return var_names.index(requested)
    # alias fallback for common choice
    aliases = {
        "t2m": "2m_temperature",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "v20": "10m_v_component_of_wind",
        "mslp": "surface_pressure",
        "tcwv": "total_column_water_vapour",
    }
    mapped = aliases.get(requested, None)
    if mapped is not None and mapped in var_names:
        return var_names.index(mapped)
    return 0


def plot_single_vs_multi_maps(
    pred: np.ndarray,
    truth: np.ndarray,
    var_name: str,
    var_idx: int,
    lead_step_multi: int,
    out_path: Path,
) -> np.ndarray:
    s_max = pred.shape[0]
    idx1 = 0
    idxm = max(0, min(lead_step_multi - 1, s_max - 1))

    t1 = truth[idx1, var_idx]
    p1 = pred[idx1, var_idx]
    e1 = np.abs(p1 - t1)

    tm = truth[idxm, var_idx]
    pm = pred[idxm, var_idx]
    em = np.abs(pm - tm)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    rows = [
        ("Single-step (lead=6h)", t1, p1, e1),
        (f"Multi-step (lead={(idxm + 1) * 6}h)", tm, pm, em),
    ]

    for r, (row_title, truth_map, pred_map, err_map) in enumerate(rows):
        vmin = float(min(truth_map.min(), pred_map.min()))
        vmax = float(max(truth_map.max(), pred_map.max()))
        emax = float(max(np.max(e1), np.max(em)))

        im0 = axes[r, 0].imshow(truth_map, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"{row_title}\nTruth")
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(pred_map, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[r, 1].set_title(f"{row_title}\nPrediction")
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

        im2 = axes[r, 2].imshow(err_map, origin="lower", cmap="magma", vmin=0.0, vmax=emax)
        axes[r, 2].set_title(f"{row_title}\nAbs Error")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{var_name}: single-step vs multi-step maps", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return em - e1


def plot_error_growth_map(growth_map: np.ndarray, var_name: str, out_path: Path) -> None:
    vmax = float(np.max(np.abs(growth_map)))
    vmax = max(vmax, 1e-8)

    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(growth_map, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(f"{var_name}: error growth map\n(abs error multi-step - abs error single-step)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        writer.writerows(rows)


def plot_global_curves(
    out_dir: Path,
    lead_steps: Sequence[int],
    checkpoint_names: Sequence[str],
    mean_rmse: Dict[str, np.ndarray],
    mean_acc: Dict[str, np.ndarray],
) -> None:
    lead_days = np.asarray(lead_steps, dtype=np.float64) * 6.0 / 24.0

    plt.figure(figsize=(9, 5.5))
    for name in checkpoint_names:
        plt.plot(lead_days, mean_rmse[name], label=name, linewidth=2.0)
    plt.xlabel("Lead time (days)")
    plt.ylabel("Mean RMSE (across variables)")
    plt.title("Checkpoint comparison: RMSE over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_mean_rmse_over_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for name in checkpoint_names:
        plt.plot(lead_days, mean_acc[name], label=name, linewidth=2.0)
    plt.xlabel("Lead time (days)")
    plt.ylabel("Mean ACC (across variables)")
    plt.title("Checkpoint comparison: ACC over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_mean_acc_over_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for name in checkpoint_names:
        growth = mean_rmse[name] - mean_rmse[name][0]
        plt.plot(lead_days, growth, label=name, linewidth=2.0)
    plt.xlabel("Lead time (days)")
    plt.ylabel("RMSE increase from first step")
    plt.title("Checkpoint comparison: error growth over lead time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_error_growth_over_time.png", dpi=180)
    plt.close()


def plot_variable_heatmap(
    out_path: Path,
    matrix: np.ndarray,
    variables: Sequence[str],
    checkpoints: Sequence[str],
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(max(8, len(checkpoints) * 1.2), max(6, len(variables) * 0.35)))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.xticks(np.arange(len(checkpoints)), checkpoints, rotation=0)
    plt.yticks(np.arange(len(variables)), variables)
    plt.title(title)
    plt.xlabel("Checkpoint")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_growth_compare_panel(
    out_path: Path,
    growth_maps: Dict[str, np.ndarray],
    var_name: str,
) -> None:
    if len(growth_maps) == 0:
        return

    names = sorted(growth_maps.keys())
    n = len(names)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    vmax = max(float(np.max(np.abs(growth_maps[nm]))) for nm in names)
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx >= n:
                ax.axis("off")
                continue
            name = names[idx]
            im = ax.imshow(growth_maps[name], origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(name)
            idx += 1
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{var_name}: error growth map comparison", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    eval_root.mkdir(parents=True, exist_ok=True)

    eval_script = Path(args.eval_script)
    if not eval_script.is_file():
        eval_script = Path(__file__).resolve().parent / args.eval_script
    if not eval_script.is_file():
        raise FileNotFoundError(f"evaluate script not found: {args.eval_script}")
    args.eval_script = str(eval_script.resolve())

    jobs = parse_checkpoint_specs(args.checkpoints, eval_root=eval_root)
    gpu_ids = parse_gpu_list(args.gpus)

    print("=" * 100)
    print("Running checkpoint evaluations")
    print("=" * 100)
    print(f"Eval root   : {eval_root}")
    print(f"Eval script : {args.eval_script}")
    print(f"Device      : {args.device}")
    print(f"GPUs        : {gpu_ids if len(gpu_ids) > 0 else 'N/A'}")
    print("Checkpoints :")
    for job in jobs:
        print(f"  - {job.name}: {job.checkpoint}")
    print("=" * 100)

    results = run_jobs(args, jobs, gpu_ids)
    if args.dry_run:
        return

    failed = [r for r in results if r.returncode != 0]
    if failed:
        print("\nSome evaluations failed:")
        for r in failed:
            print(f"  - {r.name}: returncode={r.returncode}, log={r.log_path}")
        raise RuntimeError("One or more checkpoint evaluations failed. See logs above.")

    print("\nAll checkpoint evaluations finished. Building comparison artifacts...")

    matrices: Dict[str, MetricsMatrix] = {}
    summaries: Dict[str, dict] = {}
    for result in results:
        metrics_csv = result.output_dir / "metrics_per_lead.csv"
        summary_json = result.output_dir / "summary.json"
        matrices[result.name] = read_metrics_per_lead(metrics_csv)
        if summary_json.is_file():
            with summary_json.open("r") as f:
                summaries[result.name] = json.load(f)
        else:
            summaries[result.name] = {}

    ref_name = sorted(matrices.keys())[0]
    ref_matrix = matrices[ref_name]
    lead_steps = ref_matrix.lead_steps
    variables = ref_matrix.variables

    aligned: Dict[str, MetricsMatrix] = {}
    for name, mat in matrices.items():
        aligned[name] = align_matrix(mat, ref_vars=variables, ref_steps=lead_steps)

    checkpoint_names = sorted(aligned.keys())

    summary_rows = []
    variable_rows = []
    mean_rmse_by_ckpt: Dict[str, np.ndarray] = {}
    mean_acc_by_ckpt: Dict[str, np.ndarray] = {}

    rmse_var_matrix = np.full((len(variables), len(checkpoint_names)), np.nan, dtype=np.float64)
    acc_var_matrix = np.full((len(variables), len(checkpoint_names)), np.nan, dtype=np.float64)

    for ckpt_col, name in enumerate(checkpoint_names):
        mat = aligned[name]
        mean_rmse_lead = np.nanmean(mat.rmse, axis=1)
        mean_acc_lead = np.nanmean(mat.acc, axis=1)
        mean_rmse_by_ckpt[name] = mean_rmse_lead
        mean_acc_by_ckpt[name] = mean_acc_lead

        rmse_total = float(np.nanmean(mat.rmse))
        acc_total = float(np.nanmean(mat.acc))
        acc_avg_over_time = float(np.nanmean(mean_acc_lead))

        sumj = summaries.get(name, {})
        summary_rows.append(
            [
                name,
                str(next(r.checkpoint for r in results if r.name == name)),
                rmse_total,
                acc_total,
                acc_avg_over_time,
                sumj.get("rmse_lead_day_5_mean", np.nan),
                sumj.get("acc_lead_day_5_mean", np.nan),
                sumj.get("rmse_lead_day_10_mean", np.nan),
                sumj.get("acc_lead_day_10_mean", np.nan),
                sumj.get("rmse_lead_day_15_mean", np.nan),
                sumj.get("acc_lead_day_15_mean", np.nan),
            ]
        )

        rmse_var = np.nanmean(mat.rmse, axis=0)
        acc_var = np.nanmean(mat.acc, axis=0)
        for var_idx, var_name in enumerate(variables):
            rmse_val = float(rmse_var[var_idx])
            acc_val = float(acc_var[var_idx])
            rmse_var_matrix[var_idx, ckpt_col] = rmse_val
            acc_var_matrix[var_idx, ckpt_col] = acc_val
            variable_rows.append([name, var_name, rmse_val, acc_val])

    save_csv(
        eval_root / "comparison_summary.csv",
        headers=[
            "checkpoint_name",
            "checkpoint_path",
            "rmse_total",
            "acc_total",
            "acc_average_over_time",
            "rmse_day5",
            "acc_day5",
            "rmse_day10",
            "acc_day10",
            "rmse_day15",
            "acc_day15",
        ],
        rows=summary_rows,
    )
    save_csv(
        eval_root / "comparison_per_variable.csv",
        headers=["checkpoint_name", "variable", "rmse_mean_over_time", "acc_mean_over_time"],
        rows=variable_rows,
    )

    plot_global_curves(
        out_dir=eval_root,
        lead_steps=lead_steps,
        checkpoint_names=checkpoint_names,
        mean_rmse=mean_rmse_by_ckpt,
        mean_acc=mean_acc_by_ckpt,
    )
    plot_variable_heatmap(
        out_path=eval_root / "compare_rmse_per_variable_heatmap.png",
        matrix=rmse_var_matrix,
        variables=variables,
        checkpoints=checkpoint_names,
        title="Mean RMSE per variable (averaged over lead time)",
        cmap="viridis",
    )
    plot_variable_heatmap(
        out_path=eval_root / "compare_acc_per_variable_heatmap.png",
        matrix=acc_var_matrix,
        variables=variables,
        checkpoints=checkpoint_names,
        title="Mean ACC per variable (averaged over lead time)",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
    )

    growth_maps: Dict[str, np.ndarray] = {}
    for name in checkpoint_names:
        sample_path = eval_root / f"checkpoint_{name}" / "prediction_samples" / "sample_000.npz"
        if not sample_path.is_file():
            print(f"[WARN] Missing sample file for {name}: {sample_path}")
            continue

        pred, truth, var_names = load_prediction_sample(sample_path)
        var_idx = choose_var_index(var_names, args.map_variable)
        chosen_var = var_names[var_idx]

        map_out = eval_root / f"checkpoint_{name}" / f"maps_single_vs_multi_{sanitize_name(chosen_var)}.png"
        growth = plot_single_vs_multi_maps(
            pred=pred,
            truth=truth,
            var_name=chosen_var,
            var_idx=var_idx,
            lead_step_multi=args.map_lead_step,
            out_path=map_out,
        )
        growth_out = eval_root / f"checkpoint_{name}" / f"error_growth_map_{sanitize_name(chosen_var)}.png"
        plot_error_growth_map(growth, var_name=chosen_var, out_path=growth_out)
        growth_maps[name] = growth

    if len(growth_maps) > 0:
        any_name = sorted(growth_maps.keys())[0]
        sample_file = eval_root / f"checkpoint_{any_name}" / "prediction_samples" / "sample_000.npz"
        _, _, vars_for_name = load_prediction_sample(sample_file)
        final_idx = choose_var_index(vars_for_name, args.map_variable)
        final_var_name = vars_for_name[final_idx]
        plot_growth_compare_panel(
            out_path=eval_root / f"compare_error_growth_maps_{sanitize_name(final_var_name)}.png",
            growth_maps=growth_maps,
            var_name=final_var_name,
        )

    report = {
        "created_at_unix": time.time(),
        "eval_root": str(eval_root),
        "checkpoints": {r.name: str(r.checkpoint) for r in results},
        "durations_sec": {r.name: r.duration_sec for r in results},
        "returncodes": {r.name: r.returncode for r in results},
        "files": {
            "comparison_summary_csv": str(eval_root / "comparison_summary.csv"),
            "comparison_per_variable_csv": str(eval_root / "comparison_per_variable.csv"),
            "rmse_curve_plot": str(eval_root / "compare_mean_rmse_over_time.png"),
            "acc_curve_plot": str(eval_root / "compare_mean_acc_over_time.png"),
            "error_growth_curve_plot": str(eval_root / "compare_error_growth_over_time.png"),
            "rmse_heatmap_plot": str(eval_root / "compare_rmse_per_variable_heatmap.png"),
            "acc_heatmap_plot": str(eval_root / "compare_acc_per_variable_heatmap.png"),
        },
    }
    with (eval_root / "comparison_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Comparison outputs saved in: {eval_root}")
    print("Main files:")
    print(f"  - {eval_root / 'comparison_summary.csv'}")
    print(f"  - {eval_root / 'comparison_per_variable.csv'}")
    print(f"  - {eval_root / 'compare_mean_rmse_over_time.png'}")
    print(f"  - {eval_root / 'compare_mean_acc_over_time.png'}")
    print(f"  - {eval_root / 'compare_error_growth_over_time.png'}")
    print(f"  - {eval_root / 'compare_rmse_per_variable_heatmap.png'}")
    print(f"  - {eval_root / 'compare_acc_per_variable_heatmap.png'}")


if __name__ == "__main__":
    main()
