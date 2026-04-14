#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelRun:
    name: str
    forecast_file: Path


@dataclass
class ForecastMeta:
    channel_names: Tuple[str, ...]
    lead_steps: Tuple[int, ...]
    lat_size: int
    lon_size: int


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")


def resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def as_cli_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> None:
    cmd_text = " ".join(shlex.quote(x) for x in cmd)
    print(f"[cmd] {cmd_text}")
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, cwd=REPO_ROOT)


def load_model_runs(config: Dict[str, Any], dry_run: bool) -> List[ModelRun]:
    models = config.get("models", [])
    if not models:
        raise ValueError("config.models is empty")

    seen: set[str] = set()
    runs: List[ModelRun] = []

    for raw in models:
        raw_name = raw.get("name")
        forecast_file_raw = raw.get("forecast_file")

        if not raw_name:
            raise ValueError("Each model entry needs a non-empty 'name'")
        if not forecast_file_raw:
            raise ValueError(f"Model '{raw_name}' is missing 'forecast_file'")

        name = sanitize_name(raw_name)
        if name in seen:
            raise ValueError(f"Duplicate model name after sanitization: {name}")
        seen.add(name)

        forecast_file = resolve_path(str(forecast_file_raw))
        if not dry_run and not forecast_file.is_file():
            raise FileNotFoundError(f"Forecast file not found for {name}: {forecast_file}")

        runs.append(ModelRun(name=name, forecast_file=forecast_file))

    return runs


def inspect_forecast_schema(forecast_file: Path) -> ForecastMeta:
    required_dim_order = ("init_time", "lead_step", "channel", "lat", "lon")

    ds = xr.open_dataset(forecast_file)
    try:
        missing_vars = [v for v in ["forecast", "truth"] if v not in ds]
        if missing_vars:
            raise ValueError(f"Missing variables {missing_vars} in {forecast_file}")

        for coord_name in ["channel_name", "lead_step", "lat", "lon"]:
            if coord_name not in ds:
                raise ValueError(f"Missing coordinate '{coord_name}' in {forecast_file}")

        fc = ds["forecast"]
        tr = ds["truth"]
        if tuple(fc.dims) != required_dim_order:
            raise ValueError(
                f"'forecast' dims must be {required_dim_order}, got {tuple(fc.dims)} in {forecast_file}"
            )
        if tuple(tr.dims) != required_dim_order:
            raise ValueError(
                f"'truth' dims must be {required_dim_order}, got {tuple(tr.dims)} in {forecast_file}"
            )
        if fc.shape != tr.shape:
            raise ValueError(f"forecast/truth shape mismatch in {forecast_file}: {fc.shape} vs {tr.shape}")

        channel_names = tuple(str(v) for v in ds["channel_name"].values.tolist())
        lead_steps = tuple(int(v) for v in ds["lead_step"].values.tolist())

        if len(channel_names) != int(fc.shape[2]):
            raise ValueError(
                f"channel_name length ({len(channel_names)}) != forecast channel dim ({fc.shape[2]}) in {forecast_file}"
            )
        if len(lead_steps) != int(fc.shape[1]):
            raise ValueError(
                f"lead_step length ({len(lead_steps)}) != forecast lead dim ({fc.shape[1]}) in {forecast_file}"
            )

        return ForecastMeta(
            channel_names=channel_names,
            lead_steps=lead_steps,
            lat_size=int(ds["lat"].size),
            lon_size=int(ds["lon"].size),
        )
    finally:
        ds.close()


def validate_forecasts(model_runs: Sequence[ModelRun], strict_consistency: bool, dry_run: bool) -> None:
    if dry_run:
        print("[info] skip forecast schema checks in dry-run mode")
        return

    reference: ForecastMeta | None = None
    reference_name: str | None = None

    for model in model_runs:
        meta = inspect_forecast_schema(model.forecast_file)
        print(
            f"[ok] {model.name}: channels={len(meta.channel_names)} "
            f"leads={len(meta.lead_steps)} grid={meta.lat_size}x{meta.lon_size}"
        )

        if reference is None:
            reference = meta
            reference_name = model.name
            continue

        if not strict_consistency:
            continue

        issues: List[str] = []
        if meta.channel_names != reference.channel_names:
            issues.append("channel_name list/order differs")
        if meta.lead_steps != reference.lead_steps:
            issues.append("lead_step list differs")
        if (meta.lat_size, meta.lon_size) != (reference.lat_size, reference.lon_size):
            issues.append("lat/lon grid shape differs")

        if issues:
            issue_text = "; ".join(issues)
            raise ValueError(
                f"Consistency check failed for {model.name} vs {reference_name}: {issue_text}. "
                "Set strict_consistency=false only if you intentionally compare different schemas."
            )


def run_evaluation(
    model_runs: Sequence[ModelRun],
    config: Dict[str, Any],
    python_bin: str,
    dry_run: bool,
    skip_eval: bool,
) -> None:
    if skip_eval:
        return

    results_root = resolve_path(config.get("results_root", "results_shared"))
    horizon_days = config.get("horizon_days", [5, 10, 15])

    forecast_specs = [f"{m.name}={m.forecast_file}" for m in model_runs]
    cmd = [
        python_bin,
        "-m",
        "src.checkpoint_study.evaluate_forecast_nc",
        "--forecast-files",
        *forecast_specs,
        "--results-root",
        str(results_root),
        "--horizon-days",
        as_cli_value(horizon_days),
    ]

    if bool(config.get("eval_no_heatmaps", True)):
        cmd.append("--no-heatmaps")
    if bool(config.get("overwrite_eval", True)):
        cmd.append("--overwrite")
    if config.get("climatology_store"):
        cmd.extend(["--climatology-store", str(resolve_path(config["climatology_store"]))])

    run_cmd(cmd, dry_run=dry_run)


def compute_l1_metrics(model_runs: Sequence[ModelRun], config: Dict[str, Any], dry_run: bool) -> None:
    if not bool(config.get("compute_l1", False)):
        return
    if dry_run:
        print("[info] skip l1 computation in dry-run mode")
        return

    results_root = resolve_path(config.get("results_root", "results_shared"))

    for model in model_runs:
        ds = xr.open_dataset(model.forecast_file)
        try:
            pred = ds["forecast"].values.astype(np.float64)
            truth = ds["truth"].values.astype(np.float64)
            lead_steps = ds["lead_step"].values.astype(np.int64)
            if "lead_hour" in ds:
                lead_hours = ds["lead_hour"].values.astype(np.int64)
            else:
                lead_hours = lead_steps * 6
        finally:
            ds.close()

        l1_per_lead = np.mean(np.abs(pred - truth), axis=(0, 2, 3, 4))
        lead_days = lead_hours.astype(np.float64) / 24.0

        metrics_dir = results_root / f"checkpoint_{model.name}" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "lead_step": lead_steps,
                "lead_hour": lead_hours,
                "lead_day": lead_days,
                "l1_mean": l1_per_lead,
            }
        ).to_csv(metrics_dir / "l1_per_lead.csv", index=False)

        summary_path = metrics_dir / "summary.json"
        if summary_path.is_file():
            with summary_path.open("r") as f:
                summary = json.load(f)
        else:
            summary = {}
        summary["mean_l1"] = float(np.mean(l1_per_lead))
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)


def build_comparison_report(model_runs: Sequence[ModelRun], config: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print("[info] skip comparison report in dry-run mode")
        return

    results_root = resolve_path(config.get("results_root", "results_shared"))
    compare_dir = results_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    include_l1 = bool(config.get("compute_l1", False))
    nrows = 3 if include_l1 else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(12.8, 4.6 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    summary_rows: List[Dict[str, Any]] = []
    plotted = 0

    for idx, model in enumerate(model_runs):
        metrics_dir = results_root / f"checkpoint_{model.name}" / "metrics"
        mean_csv = metrics_dir / "mean_metrics_per_lead.csv"
        summary_json = metrics_dir / "summary.json"
        if not mean_csv.is_file() or not summary_json.is_file():
            print(f"[warn] missing metrics for {model.name}, skipping in comparison report")
            continue

        df = pd.read_csv(mean_csv)
        with summary_json.open("r") as f:
            summary = json.load(f)

        lead_days = df["lead_day"].to_numpy(dtype=float)
        rmse = df["rmse_mean_over_variables"].to_numpy(dtype=float)
        acc = df["acc_mean_over_variables"].to_numpy(dtype=float)

        color = plt.get_cmap("tab10")(idx % 10)
        axes[0].plot(lead_days, rmse, linewidth=2.2, label=model.name, color=color)
        axes[1].plot(lead_days, acc, linewidth=2.2, label=model.name, color=color)

        row: Dict[str, Any] = {
            "model": model.name,
            "mean_rmse": float(summary.get("mean_rmse", np.nan)),
            "mean_rmse_unweighted": float(summary.get("mean_rmse_unweighted", np.nan)),
            "mean_acc": float(summary.get("mean_acc", np.nan)),
        }

        if include_l1:
            l1_csv = metrics_dir / "l1_per_lead.csv"
            if l1_csv.is_file():
                l1_df = pd.read_csv(l1_csv)
                axes[2].plot(
                    l1_df["lead_day"].to_numpy(dtype=float),
                    l1_df["l1_mean"].to_numpy(dtype=float),
                    linewidth=2.2,
                    label=model.name,
                    color=color,
                )
                row["mean_l1"] = float(np.mean(l1_df["l1_mean"].to_numpy(dtype=float)))
            else:
                row["mean_l1"] = np.nan

        summary_rows.append(row)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("[warn] no comparison figure generated (no metric files found)")
        return

    axes[0].set_title("Mean RMSE over variables")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)

    axes[1].set_title("Mean ACC over variables")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(-0.1, 1.02)
    axes[1].legend(frameon=False)

    if include_l1:
        axes[2].set_title("Mean L1 over variables")
        axes[2].set_xlabel("Lead time (days)")
        axes[2].set_ylabel("L1")
        axes[2].legend(frameon=False)

    fig.savefig(compare_dir / "model_comparison.png", dpi=260)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["mean_rmse", "mean_acc"], ascending=[True, False])
    summary_df.to_csv(compare_dir / "model_summary.csv", index=False)


def write_horizon_comparison(model_runs: Sequence[ModelRun], config: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return

    results_root = resolve_path(config.get("results_root", "results_shared"))
    compare_dir = results_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for model in model_runs:
        csv_path = results_root / f"checkpoint_{model.name}" / "metrics" / "horizon_window_summary.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "model", model.name)
        frames.append(df)

    if not frames:
        return

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(compare_dir / "horizon_comparison.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast-only shared benchmark runner for weather-model comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark JSON config")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluate_forecast_nc stage")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    parser.add_argument("--compute-l1", action="store_true", help="Force-enable L1 metric")
    parser.add_argument("--no-compute-l1", action="store_true", help="Force-disable L1 metric")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config_path = resolve_path(args.config)
    with config_path.open("r") as f:
        config = json.load(f)

    if args.compute_l1 and args.no_compute_l1:
        raise ValueError("Use either --compute-l1 or --no-compute-l1, not both")
    if args.compute_l1:
        config["compute_l1"] = True
    if args.no_compute_l1:
        config["compute_l1"] = False

    strict_consistency = bool(config.get("strict_consistency", True))
    python_bin = str(Path(sys.executable).resolve())

    print("=" * 88)
    print("Shared Weather Benchmark (forecast only)")
    print("=" * 88)
    print(f"Config             : {config_path}")
    print(f"Python             : {python_bin}")
    print(f"Results root       : {resolve_path(config.get('results_root', 'results_shared'))}")
    print(f"Strict consistency : {strict_consistency}")
    print(f"Compute L1         : {bool(config.get('compute_l1', False))}")
    print("=" * 88)

    model_runs = load_model_runs(config=config, dry_run=args.dry_run)
    validate_forecasts(model_runs=model_runs, strict_consistency=strict_consistency, dry_run=args.dry_run)

    run_evaluation(
        model_runs=model_runs,
        config=config,
        python_bin=python_bin,
        dry_run=args.dry_run,
        skip_eval=args.skip_eval,
    )
    compute_l1_metrics(model_runs=model_runs, config=config, dry_run=args.dry_run)
    build_comparison_report(model_runs=model_runs, config=config, dry_run=args.dry_run)
    write_horizon_comparison(model_runs=model_runs, config=config, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
