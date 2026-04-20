#!/usr/bin/env python3
"""Generate publication-quality report figures from existing evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_column_water_vapour",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-level figure pack for the report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="report")

    parser.add_argument(
        "--metrics-a",
        type=str,
        default="results_new/checkpoint_emb768/metrics/metrics_per_lead.csv",
    )
    parser.add_argument(
        "--metrics-b",
        type=str,
        default="results_new/checkpoint_emb1024/metrics/metrics_per_lead.csv",
    )
    parser.add_argument("--label-a", type=str, default="Embed-768")
    parser.add_argument("--label-b", type=str, default="Embed-1024")

    parser.add_argument(
        "--history-files",
        nargs="+",
        default=[
            "Models_paper/pretrain/embed_512_8-32-8/history.csv",
            "Models_paper/pretrain/embed_768_8-32-8/history.csv",
            "Models_paper/pretrain/embed_1024_8-32-8/history.csv",
            "Models_paper/pretrain/embed_1536_8-32-8/history.csv",
            "Models_paper/pretrain/embed_2048_8-32-8/history.csv",
        ],
    )
    parser.add_argument(
        "--embed-dims",
        nargs="+",
        type=int,
        default=[512, 768, 1024, 1536, 2048],
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.facecolor": "#fbfbfc",
            "axes.facecolor": "#fbfbfc",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def variable_order(existing: Sequence[str]) -> List[str]:
    preferred = [
        "2m_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_column_water_vapour",
        "temperature_plev850",
        "temperature_plev500",
        "temperature_plev250",
        "geopotential_plev850",
        "geopotential_plev500",
        "geopotential_plev250",
        "specific_humidity_plev850",
        "specific_humidity_plev500",
        "specific_humidity_plev250",
        "u_component_of_wind_plev850",
        "u_component_of_wind_plev500",
        "u_component_of_wind_plev250",
        "v_component_of_wind_plev850",
        "v_component_of_wind_plev500",
        "v_component_of_wind_plev250",
    ]
    seen = set(existing)
    ordered = [v for v in preferred if v in seen]
    ordered.extend([v for v in existing if v not in ordered])
    return ordered


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"variable", "lead_step", "lead_day", "rmse", "acc"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    var_order = variable_order(sorted(df["variable"].unique().tolist()))
    p = (
        df.pivot_table(index="lead_step", columns="variable", values=metric, aggfunc="mean")
        .sort_index()
        .reindex(columns=var_order)
    )
    steps = p.index.to_numpy(dtype=np.int32)
    lead_days = steps.astype(np.float64) * 6.0 / 24.0
    return p.to_numpy(dtype=np.float64), list(p.columns), lead_days


def global_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["lead_step", "lead_day"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_q25=("rmse", lambda x: float(np.quantile(x, 0.25))),
            rmse_q75=("rmse", lambda x: float(np.quantile(x, 0.75))),
            acc_mean=("acc", "mean"),
            acc_q25=("acc", lambda x: float(np.quantile(x, 0.25))),
            acc_q75=("acc", lambda x: float(np.quantile(x, 0.75))),
        )
        .sort_values("lead_step")
    )


def horizon_values(df: pd.DataFrame, day: int) -> Tuple[float, float]:
    step = int(day * 4)
    sl = df[df["lead_step"] == step]
    return float(sl["rmse"].mean()), float(sl["acc"].mean())


def group_indices(variables: Sequence[str]) -> Dict[str, List[int]]:
    groups = {
        "Surface": [i for i, v in enumerate(variables) if v in SURFACE_VARS],
        "Temperature (upper-air)": [i for i, v in enumerate(variables) if v.startswith("temperature_plev")],
        "Geopotential (upper-air)": [i for i, v in enumerate(variables) if v.startswith("geopotential_plev")],
        "Humidity (upper-air)": [i for i, v in enumerate(variables) if v.startswith("specific_humidity_plev")],
        "Wind-u (upper-air)": [i for i, v in enumerate(variables) if v.startswith("u_component_of_wind_plev")],
        "Wind-v (upper-air)": [i for i, v in enumerate(variables) if v.startswith("v_component_of_wind_plev")],
    }
    return groups


def plot_global_skill_panel(
    out_path: Path,
    g_a: pd.DataFrame,
    g_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].plot(g_a["lead_day"], g_a["rmse_mean"], color="#1f77b4", linewidth=2.4, label=label_a)
    axes[0].fill_between(g_a["lead_day"], g_a["rmse_q25"], g_a["rmse_q75"], color="#1f77b4", alpha=0.18)
    axes[0].plot(g_b["lead_day"], g_b["rmse_mean"], color="#ff7f0e", linewidth=2.4, label=label_b)
    axes[0].fill_between(g_b["lead_day"], g_b["rmse_q25"], g_b["rmse_q75"], color="#ff7f0e", alpha=0.18)
    axes[0].set_title("Global RMSE progression (mean with interquartile band)")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)

    axes[1].plot(g_a["lead_day"], g_a["acc_mean"], color="#1f77b4", linewidth=2.4, label=label_a)
    axes[1].fill_between(g_a["lead_day"], g_a["acc_q25"], g_a["acc_q75"], color="#1f77b4", alpha=0.18)
    axes[1].plot(g_b["lead_day"], g_b["acc_mean"], color="#ff7f0e", linewidth=2.4, label=label_b)
    axes[1].fill_between(g_b["lead_day"], g_b["acc_q25"], g_b["acc_q75"], color="#ff7f0e", alpha=0.18)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title("Global ACC progression (mean with interquartile band)")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_horizon_bar_summary(
    out_path: Path,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    days = [5, 10, 15]
    rmse_a, acc_a, rmse_b, acc_b = [], [], [], []
    for d in days:
        ra, aa = horizon_values(df_a, d)
        rb, ab = horizon_values(df_b, d)
        rmse_a.append(ra)
        acc_a.append(aa)
        rmse_b.append(rb)
        acc_b.append(ab)

    x = np.arange(len(days), dtype=np.float64)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    axes[0].bar(x - w / 2, rmse_a, width=w, label=label_a, color="#1f77b4")
    axes[0].bar(x + w / 2, rmse_b, width=w, label=label_b, color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Day {d}" for d in days])
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Horizon RMSE summary")
    axes[0].legend(frameon=False)

    axes[1].bar(x - w / 2, acc_a, width=w, label=label_a, color="#1f77b4")
    axes[1].bar(x + w / 2, acc_b, width=w, label=label_b, color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Day {d}" for d in days])
    axes[1].set_ylabel("ACC")
    axes[1].set_title("Horizon ACC summary")
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_variable_delta(
    out_path: Path,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    pa = df_a.groupby("variable", as_index=False).agg(rmse=("rmse", "mean"), acc=("acc", "mean"))
    pb = df_b.groupby("variable", as_index=False).agg(rmse=("rmse", "mean"), acc=("acc", "mean"))
    merged = pa.merge(pb, on="variable", suffixes=("_a", "_b"))

    merged["rmse_improvement_pct"] = 100.0 * (merged["rmse_a"] - merged["rmse_b"]) / np.maximum(np.abs(merged["rmse_a"]), 1e-8)
    merged["acc_gain"] = merged["acc_b"] - merged["acc_a"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    rmse_sorted = merged.sort_values("rmse_improvement_pct", ascending=False)
    rmse_colors = ["#2ca02c" if v >= 0.0 else "#d62728" for v in rmse_sorted["rmse_improvement_pct"]]
    axes[0].barh(rmse_sorted["variable"], rmse_sorted["rmse_improvement_pct"], color=rmse_colors)
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set_title(f"Per-variable RMSE improvement (%) ({label_b} vs {label_a})")
    axes[0].set_xlabel("RMSE improvement in % (positive = second model lower RMSE)")

    acc_sorted = merged.sort_values("acc_gain", ascending=False)
    acc_colors = ["#2ca02c" if v >= 0.0 else "#d62728" for v in acc_sorted["acc_gain"]]
    axes[1].barh(acc_sorted["variable"], acc_sorted["acc_gain"], color=acc_colors)
    axes[1].axvline(0.0, color="black", linewidth=1.0)
    axes[1].set_title(f"Per-variable ACC gain ({label_b} - {label_a})")
    axes[1].set_xlabel("ACC gain (positive = second model higher ACC)")

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_grouped_skill_curves(
    out_path: Path,
    rmse_a: np.ndarray,
    acc_a: np.ndarray,
    rmse_b: np.ndarray,
    acc_b: np.ndarray,
    variables: Sequence[str],
    lead_days: np.ndarray,
    label_a: str,
    label_b: str,
) -> None:
    groups = group_indices(variables)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (gname, inds) in enumerate(groups.items()):
        if not inds:
            continue
        c = palette[idx % len(palette)]
        mean_rmse_a = np.nanmean(rmse_a[:, inds], axis=1)
        mean_rmse_b = np.nanmean(rmse_b[:, inds], axis=1)
        axes[0].plot(lead_days, mean_rmse_a, color=c, linewidth=2.0, linestyle="-", label=f"{gname} ({label_a})")
        axes[0].plot(lead_days, mean_rmse_b, color=c, linewidth=2.0, linestyle="--", label=f"{gname} ({label_b})")

        mean_acc_a = np.nanmean(acc_a[:, inds], axis=1)
        mean_acc_b = np.nanmean(acc_b[:, inds], axis=1)
        axes[1].plot(lead_days, mean_acc_a, color=c, linewidth=2.0, linestyle="-", label=f"{gname} ({label_a})")
        axes[1].plot(lead_days, mean_acc_b, color=c, linewidth=2.0, linestyle="--", label=f"{gname} ({label_b})")

    axes[0].set_title("Grouped RMSE trajectories (solid=first model, dashed=second model)")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].set_title("Grouped ACC trajectories (solid=first model, dashed=second model)")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_geopotential_levels(
    out_path: Path,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    level_vars = ["geopotential_plev850", "geopotential_plev500", "geopotential_plev250"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    colors = {
        "geopotential_plev850": "#1f77b4",
        "geopotential_plev500": "#ff7f0e",
        "geopotential_plev250": "#2ca02c",
    }

    for v in level_vars:
        da = df_a[df_a["variable"] == v].sort_values("lead_step")
        db = df_b[df_b["variable"] == v].sort_values("lead_step")
        c = colors[v]
        axes[0].plot(da["lead_day"], da["rmse"], color=c, linestyle="-", linewidth=2.1, label=f"{v} ({label_a})")
        axes[0].plot(db["lead_day"], db["rmse"], color=c, linestyle="--", linewidth=2.1, label=f"{v} ({label_b})")

        axes[1].plot(da["lead_day"], da["acc"], color=c, linestyle="-", linewidth=2.1, label=f"{v} ({label_a})")
        axes[1].plot(db["lead_day"], db["acc"], color=c, linestyle="--", linewidth=2.1, label=f"{v} ({label_b})")

    axes[0].set_title("Geopotential RMSE by pressure level")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].set_title("Geopotential ACC by pressure level")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_compare_heatmaps(
    out_path: Path,
    rmse_a: np.ndarray,
    acc_a: np.ndarray,
    rmse_b: np.ndarray,
    acc_b: np.ndarray,
    variables: Sequence[str],
) -> None:
    # Normalize RMSE by variable median over lead time to make mixed-unit variables comparable.
    rmse_a_scale = np.nanmedian(rmse_a, axis=0, keepdims=True)
    rmse_b_scale = np.nanmedian(rmse_b, axis=0, keepdims=True)
    rmse_a_scale = np.where(np.abs(rmse_a_scale) < 1e-8, 1.0, rmse_a_scale)
    rmse_b_scale = np.where(np.abs(rmse_b_scale) < 1e-8, 1.0, rmse_b_scale)
    rmse_a_n = rmse_a / rmse_a_scale
    rmse_b_n = rmse_b / rmse_b_scale

    rmse_delta = rmse_b_n - rmse_a_n
    acc_delta = acc_b - acc_a

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    v0 = float(np.nanpercentile(np.concatenate([rmse_a_n.ravel(), rmse_b_n.ravel()]), 5.0))
    v1 = float(np.nanpercentile(np.concatenate([rmse_a_n.ravel(), rmse_b_n.ravel()]), 95.0))
    im0 = axes[0, 0].imshow(rmse_a_n.T, aspect="auto", origin="lower", cmap="viridis", vmin=v0, vmax=v1)
    axes[0, 0].set_title("RMSE heatmap: first model (normalized)")
    axes[0, 0].set_ylabel("Variable")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.03, pad=0.02)

    im1 = axes[0, 1].imshow(rmse_b_n.T, aspect="auto", origin="lower", cmap="viridis", vmin=v0, vmax=v1)
    axes[0, 1].set_title("RMSE heatmap: second model (normalized)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.03, pad=0.02)

    lim_r = max(float(np.nanpercentile(np.abs(rmse_delta), 98.0)), 1e-8)
    im2 = axes[1, 0].imshow(rmse_delta.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-lim_r, vmax=lim_r)
    axes[1, 0].set_title("RMSE delta heatmap (normalized second - normalized first)")
    axes[1, 0].set_xlabel("Lead step")
    axes[1, 0].set_ylabel("Variable")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.03, pad=0.02)

    lim_a = max(float(np.nanpercentile(np.abs(acc_delta), 98.0)), 1e-8)
    im3 = axes[1, 1].imshow(acc_delta.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-lim_a, vmax=lim_a)
    axes[1, 1].set_title("ACC delta heatmap (second - first)")
    axes[1, 1].set_xlabel("Lead step")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.03, pad=0.02)

    yticks = np.arange(len(variables))
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_yticks(yticks)
        ax.set_yticklabels(variables, fontsize=8)
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_yticks(yticks)
        ax.set_yticklabels([])

    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_pretraining_scaling(
    out_curve_path: Path,
    out_best_path: Path,
    history_files: Sequence[Path],
    embed_dims: Sequence[int],
) -> None:
    if len(history_files) != len(embed_dims):
        raise ValueError("history-files and embed-dims must have same length")

    data = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    for path, dim in zip(history_files, embed_dims):
        df = pd.read_csv(path)
        data.append(
            {
                "embed_dim": int(dim),
                "best_val": float(df["val_loss"].min()),
                "final_val": float(df["val_loss"].iloc[-1]),
                "final_train": float(df["train_loss"].iloc[-1]),
            }
        )

        axes[0].plot(df["epoch"], df["val_loss"], linewidth=2.0, label=f"val e{dim}")
        axes[1].plot(df["epoch"], df["train_loss"], linewidth=2.0, label=f"train e{dim}")

    axes[0].set_title("Pretraining validation-loss trajectories")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].set_title("Pretraining training-loss trajectories")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_curve_path, dpi=280)
    plt.close(fig)

    sdf = pd.DataFrame(data).sort_values("embed_dim")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    axes[0].plot(sdf["embed_dim"], sdf["best_val"], marker="o", linewidth=2.4, color="#1f77b4")
    axes[0].set_title("Best validation loss vs embedding dimension")
    axes[0].set_xlabel("Embedding dimension")
    axes[0].set_ylabel("Best validation loss")

    axes[1].plot(sdf["embed_dim"], sdf["final_train"], marker="o", linewidth=2.2, color="#2ca02c", label="Final train")
    axes[1].plot(sdf["embed_dim"], sdf["final_val"], marker="o", linewidth=2.2, color="#ff7f0e", label="Final val")
    axes[1].set_title("Final train/val loss vs embedding dimension")
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)

    fig.savefig(out_best_path, dpi=280)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    out_dir = resolve_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style()

    metrics_a = resolve_path(repo_root, args.metrics_a)
    metrics_b = resolve_path(repo_root, args.metrics_b)

    df_a = load_metrics(metrics_a)
    df_b = load_metrics(metrics_b)

    rmse_a, vars_a, lead_days_a = pivot_metric(df_a, "rmse")
    acc_a, vars_acc_a, lead_days_acc_a = pivot_metric(df_a, "acc")
    rmse_b, vars_b, lead_days_b = pivot_metric(df_b, "rmse")
    acc_b, vars_acc_b, lead_days_acc_b = pivot_metric(df_b, "acc")

    if vars_a != vars_b or vars_a != vars_acc_a or vars_a != vars_acc_b:
        raise ValueError("Variable sets differ across inputs")
    if not (np.allclose(lead_days_a, lead_days_b) and np.allclose(lead_days_a, lead_days_acc_a) and np.allclose(lead_days_a, lead_days_acc_b)):
        raise ValueError("Lead-day axes differ across inputs")

    g_a = global_summary(df_a)
    g_b = global_summary(df_b)

    plot_global_skill_panel(out_dir / "publication_global_skill_panel.png", g_a, g_b, args.label_a, args.label_b)
    plot_horizon_bar_summary(out_dir / "publication_horizon_bar_summary.png", df_a, df_b, args.label_a, args.label_b)
    plot_variable_delta(out_dir / "publication_variable_delta.png", df_a, df_b, args.label_a, args.label_b)
    plot_grouped_skill_curves(
        out_dir / "publication_grouped_skill_curves.png",
        rmse_a,
        acc_a,
        rmse_b,
        acc_b,
        vars_a,
        lead_days_a,
        args.label_a,
        args.label_b,
    )
    plot_geopotential_levels(out_dir / "publication_geopotential_levels.png", df_a, df_b, args.label_a, args.label_b)
    plot_compare_heatmaps(out_dir / "publication_compare_heatmaps.png", rmse_a, acc_a, rmse_b, acc_b, vars_a)

    history_paths = [resolve_path(repo_root, p) for p in args.history_files]
    for hp in history_paths:
        if not hp.is_file():
            raise FileNotFoundError(f"Missing history file: {hp}")

    plot_pretraining_scaling(
        out_dir / "publication_pretraining_curves.png",
        out_dir / "publication_pretraining_scaling.png",
        history_paths,
        args.embed_dims,
    )

    print("Publication report figure pack generated at:", out_dir)
    for name in [
        "publication_global_skill_panel.png",
        "publication_horizon_bar_summary.png",
        "publication_variable_delta.png",
        "publication_grouped_skill_curves.png",
        "publication_geopotential_levels.png",
        "publication_compare_heatmaps.png",
        "publication_pretraining_curves.png",
        "publication_pretraining_scaling.png",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
