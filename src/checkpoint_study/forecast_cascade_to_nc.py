#!/usr/bin/env python3
"""
Generate cascade forecasts using short/medium/long checkpoints in one rollout.

This script is separate from forecast_to_nc.py so cascade experiments are isolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .common import (
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        parse_csv_ints,
        parse_csv_strings,
    )
    from .forecast_to_nc import (
        SelectedRolloutDataset,
        build_inference_model,
        estimate_output_gb,
        plot_day1_forecast,
        plot_t2m_rollout_panels,
        resolve_spec_for_checkpoint,
        select_start_positions,
        set_plot_style,
        write_forecast_netcdf,
    )
except ImportError:
    from src.checkpoint_study.common import (
        DEFAULT_RESULTS_ROOT,
        build_checkpoint_dirs,
        parse_csv_ints,
        parse_csv_strings,
    )
    from src.checkpoint_study.forecast_to_nc import (
        SelectedRolloutDataset,
        build_inference_model,
        estimate_output_gb,
        plot_day1_forecast,
        plot_t2m_rollout_panels,
        resolve_spec_for_checkpoint,
        select_start_positions,
        set_plot_style,
        write_forecast_netcdf,
    )

try:
    from ..evaluation.evaluate_checkpoint import (
        WB2Accessor,
        autocast_ctx,
        choose_device,
        compute_channel_stats,
        resolve_variable_names,
    )
except ImportError:
    from src.evaluation.evaluate_checkpoint import (
        WB2Accessor,
        autocast_ctx,
        choose_device,
        compute_channel_stats,
        resolve_variable_names,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cascade forecast NetCDF by chaining short/medium/long checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--short-checkpoint", type=str, required=True)
    parser.add_argument("--medium-checkpoint", type=str, required=True)
    parser.add_argument("--long-checkpoint", type=str, required=True)
    parser.add_argument("--cascade-name", type=str, default="cascade")
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--forecast-file-name", type=str, default="forecast_cascade.nc")
    parser.add_argument("--stage-steps", type=parse_csv_ints, default=[20, 20, 20])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--zarr-store", type=str, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--pressure-vars", type=parse_csv_strings, default=None)
    parser.add_argument("--surface-vars", type=parse_csv_strings, default=None)
    parser.add_argument("--pressure-levels", type=parse_csv_ints, default=None)
    parser.add_argument("--history-steps", type=int, default=2)

    parser.add_argument("--init-times", type=parse_csv_strings, default=None)
    parser.add_argument("--init-start", type=str, default=None)
    parser.add_argument("--init-end", type=str, default=None)
    parser.add_argument("--init-stride", type=int, default=1)
    parser.add_argument("--max-inits", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")

    parser.add_argument("--day1-plot-vars", type=parse_csv_strings, default=None)
    parser.add_argument("--skip-day1-plot", action="store_true")
    parser.add_argument("--skip-t2m-plot", action="store_true")
    parser.add_argument("--t2m-var-name", type=str, default="2m_temperature")
    parser.add_argument("--t2m-plot-days", type=parse_csv_ints, default=[1, 5, 10, 15])
    parser.add_argument("--max-output-gb", type=float, default=8.0)

    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-base-checkpoint", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", type=parse_csv_strings, default=None)
    parser.add_argument("--lora-bias", type=str, default=None)
    return parser


@torch.no_grad()
def run_cascade_rollout(
    models: Sequence,
    stage_steps: Sequence[int],
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    amp: str,
    channels: int,
    spatial_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_steps = int(sum(int(s) for s in stage_steps))
    n_samples = len(loader.dataset)
    h, w = spatial_shape
    forecast = np.empty((n_samples, total_steps, channels, h, w), dtype=np.float32)
    truth = np.empty_like(forecast)
    init_times = np.empty((n_samples,), dtype="datetime64[ns]")

    mean_d = mean.to(device)
    std_d = std.to(device)
    boundaries = np.cumsum(np.asarray(stage_steps, dtype=np.int64))

    cursor = 0
    total_batches = len(loader)
    for batch_idx, (history, future, init_time_batch) in enumerate(loader, start=1):
        history = history.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        batch_size = int(history.shape[0])

        hist = history
        pred_steps = []
        for s in range(total_steps):
            stage_idx = int(np.searchsorted(boundaries, s, side="right"))
            stage_idx = min(stage_idx, len(models) - 1)
            model = models[stage_idx]
            with autocast_ctx(device, amp):
                pred_n = model(hist)
            pred_steps.append(pred_n)
            hist = torch.stack([hist[:, :, 1], pred_n], dim=2)

        pred_n = torch.stack(pred_steps, dim=1)
        pred = (pred_n.float() * std_d + mean_d).cpu().numpy().astype(np.float32, copy=False)
        tgt = (future.float() * std_d + mean_d).cpu().numpy().astype(np.float32, copy=False)

        forecast[cursor : cursor + batch_size] = pred
        truth[cursor : cursor + batch_size] = tgt
        for j, init_time in enumerate(init_time_batch):
            init_times[cursor + j] = np.datetime64(str(init_time)).astype("datetime64[ns]")
        cursor += batch_size

        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            print(f"[cascade] batch {batch_idx}/{total_batches} | generated={cursor}/{n_samples}", flush=True)

    return forecast, truth, init_times


def main() -> None:
    args = build_parser().parse_args()
    set_plot_style()

    stage_steps = [int(x) for x in args.stage_steps]
    if len(stage_steps) != 3:
        raise ValueError("--stage-steps must contain exactly 3 integers: short,medium,long")
    if any(s <= 0 for s in stage_steps):
        raise ValueError("All stage steps must be positive")

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    short_spec, short_ckpt, short_cfg = resolve_spec_for_checkpoint(args.short_checkpoint, args)
    med_spec, med_ckpt, med_cfg = resolve_spec_for_checkpoint(args.medium_checkpoint, args)
    long_spec, long_ckpt, long_cfg = resolve_spec_for_checkpoint(args.long_checkpoint, args)

    total_steps = int(sum(stage_steps))
    print("=" * 96)
    print("FuXi Cascade Forecast to NetCDF")
    print("=" * 96)
    print(f"Cascade name      : {args.cascade_name}")
    print(f"Stage steps       : {stage_steps} (total={total_steps})")
    print(f"Short checkpoint  : {args.short_checkpoint}")
    print(f"Medium checkpoint : {args.medium_checkpoint}")
    print(f"Long checkpoint   : {args.long_checkpoint}")
    print(f"Device            : {device}")
    print(f"AMP               : {args.amp}")
    print("=" * 96)

    _checkpoint_dir, forecast_dir, _metrics_dir = build_checkpoint_dirs(Path(args.results_root), args.cascade_name)
    forecast_path = forecast_dir / args.forecast_file_name
    if forecast_path.exists() and not args.overwrite:
        print(f"[skip] {forecast_path} exists. Use --overwrite to regenerate.")
        return

    pressure_vars, surface_vars, notes = resolve_variable_names(
        short_spec.zarr_store,
        short_spec.pressure_vars,
        short_spec.surface_vars,
    )
    for note in notes:
        print(f"  [var-map] {note}")

    accessor = WB2Accessor(
        zarr_path=short_spec.zarr_store,
        pressure_vars=pressure_vars,
        surface_vars=surface_vars,
        pressure_levels=short_spec.pressure_levels,
    )
    train_idx = accessor.time_indices_between(short_spec.train_start, short_spec.train_end)
    test_idx = accessor.time_indices_between(short_spec.test_start, short_spec.test_end)
    mean, std = compute_channel_stats(accessor, train_idx, stats_samples=args.stats_samples)

    starts, selected_init_times = select_start_positions(
        accessor=accessor,
        eval_time_indices=test_idx,
        rollout_steps=total_steps,
        history_steps=short_spec.history_steps,
        init_times=args.init_times,
        init_start=args.init_start,
        init_end=args.init_end,
        init_stride=args.init_stride,
        max_inits=args.max_inits,
    )

    est_gb = estimate_output_gb(
        n_init=int(starts.shape[0]),
        rollout_steps=total_steps,
        channels=int(accessor.channels),
        h=int(accessor.spatial_shape[0]),
        w=int(accessor.spatial_shape[1]),
    )
    print(f"  selected_inits={starts.shape[0]} | estimated_output={est_gb:.2f} GiB")
    if est_gb > float(args.max_output_gb):
        raise RuntimeError(
            f"Estimated output size {est_gb:.2f} GiB exceeds --max-output-gb={args.max_output_gb}."
        )

    dataset = SelectedRolloutDataset(
        accessor=accessor,
        eval_time_indices=test_idx,
        start_positions=starts,
        rollout_steps=total_steps,
        mean=mean,
        std=std,
        history_steps=short_spec.history_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    short_model, short_lora = build_inference_model(
        spec=short_spec,
        ckpt=short_ckpt,
        cfg=short_cfg,
        args=args,
        channels=accessor.channels,
        spatial_shape=accessor.spatial_shape,
        device=device,
    )
    med_model, med_lora = build_inference_model(
        spec=med_spec,
        ckpt=med_ckpt,
        cfg=med_cfg,
        args=args,
        channels=accessor.channels,
        spatial_shape=accessor.spatial_shape,
        device=device,
    )
    long_model, long_lora = build_inference_model(
        spec=long_spec,
        ckpt=long_ckpt,
        cfg=long_cfg,
        args=args,
        channels=accessor.channels,
        spatial_shape=accessor.spatial_shape,
        device=device,
    )

    forecast, truth, init_times = run_cascade_rollout(
        models=[short_model, med_model, long_model],
        stage_steps=stage_steps,
        loader=loader,
        mean=mean,
        std=std,
        device=device,
        amp=args.amp,
        channels=accessor.channels,
        spatial_shape=accessor.spatial_shape,
    )

    if selected_init_times.shape[0] == init_times.shape[0]:
        init_times = selected_init_times

    lead_steps = np.arange(1, total_steps + 1, dtype=np.int32)
    write_forecast_netcdf(
        out_path=forecast_path,
        forecast=forecast,
        truth=truth,
        init_times=init_times,
        lead_steps=lead_steps,
        var_names=accessor.var_names,
        latitudes=accessor.latitudes,
        longitudes=accessor.longitudes,
        checkpoint_path=Path(args.long_checkpoint),
        spec=short_spec,
        lora_enabled=bool(short_lora or med_lora or long_lora),
    )

    if not args.skip_day1_plot:
        plot_day1_forecast(
            out_path=forecast_dir / "day1_forecast.png",
            forecast=forecast,
            truth=truth,
            var_names=accessor.var_names,
            latitudes=accessor.latitudes,
            longitudes=accessor.longitudes,
            requested_vars=args.day1_plot_vars,
        )

    if not args.skip_t2m_plot:
        plot_t2m_rollout_panels(
            out_path=forecast_dir / "t2m_rollout_panels.png",
            forecast=forecast,
            truth=truth,
            var_names=accessor.var_names,
            lead_steps=lead_steps,
            latitudes=accessor.latitudes,
            longitudes=accessor.longitudes,
            t2m_name=args.t2m_var_name,
            day_list=args.t2m_plot_days,
        )

    metadata = {
        "mode": "cascade",
        "cascade_name": args.cascade_name,
        "forecast_file": str(forecast_path),
        "zarr_store": short_spec.zarr_store,
        "test_start": short_spec.test_start,
        "test_end": short_spec.test_end,
        "n_init_times": int(init_times.shape[0]),
        "init_times": [
            np.datetime_as_string(t.astype("datetime64[s]"), unit="s")
            for t in init_times.astype("datetime64[ns]")
        ],
        "stage_steps": stage_steps,
        "rollout_steps": total_steps,
        "lead_hours": (lead_steps * 6).tolist(),
        "short_checkpoint": str(Path(args.short_checkpoint).expanduser().resolve()),
        "medium_checkpoint": str(Path(args.medium_checkpoint).expanduser().resolve()),
        "long_checkpoint": str(Path(args.long_checkpoint).expanduser().resolve()),
        "short_lora_enabled": bool(short_lora),
        "medium_lora_enabled": bool(med_lora),
        "long_lora_enabled": bool(long_lora),
        "channels": accessor.var_names,
    }
    with (forecast_dir / "forecast_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {forecast_path}")
    print("Done.")


if __name__ == "__main__":
    main()
