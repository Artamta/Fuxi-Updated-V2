#!/usr/bin/env python3
"""Build a professor-ready FuXi results bundle under ./prof.

Outputs:
- prof/plots: publication-style PNG plots
- prof/data: CSV tables used by plots
- prof/raw_snapshot: copied logs + metric artifacts for traceability
- prof/report: concise markdown summary
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROF = ROOT / "prof"
PLOTS_DIR = PROF / "plots"
DATA_DIR = PROF / "data"
REPORT_DIR = PROF / "report"
RAW_SNAPSHOT = PROF / "raw_snapshot"

LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
MODELS_PAPER_DIR = ROOT / "Models_paper"


STAGE_RE = re.compile(r"==>\s*Stage:\s*([a-zA-Z0-9_\-]+)")
TRAIN_STEP_RE = re.compile(
    r"\[train\]\s+global_step=(\d+).*?loss=([0-9eE+\-.]+|nan)", re.IGNORECASE
)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|.*?train loss=([0-9eE+\-.]+|nan).*?\|\s*val loss=([0-9eE+\-.]+|nan)",
    re.IGNORECASE,
)
OUT_ROOT_RE = re.compile(r"Output root\s*:\s*(.+)")
RUN_DIR_RE = re.compile(r"Run directory\s*:\s*(.+)")
ENABLE_LORA_RE = re.compile(r"enable_lora\s*=\s*([01])")
RUNTIME_LORA_RE = re.compile(r"Runtime tuning:.*enable_lora=([01])", re.IGNORECASE)
LORA_ENABLED_LINE_RE = re.compile(r"LoRA enabled", re.IGNORECASE)
LORA_RANK_RE = re.compile(r"(?:lora_rank|rank)\s*=\s*([0-9]+)")
NOISE_RE = re.compile(r"(?:noise_std|input_noise_std)\s*=\s*([0-9.]+)")

ERROR_HINT_RE = re.compile(r"Traceback|RuntimeError|TypeError|ERROR:")


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def ensure_dirs() -> None:
    for d in (PROF, PLOTS_DIR, DATA_DIR, REPORT_DIR, RAW_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)


def is_finetune_log(path: Path) -> bool:
    name = path.name
    return any(
        token in name
        for token in (
            "fuxi_lora_",
            "fuxi_a30_ar768_arr_",
            "fuxi_ar768_poster_",
            "fuxi_ar768_def_",
            "lora_midtest_",
            "lora_smoke_",
        )
    )


def classify_run(path_hint: str, lora_enabled: Optional[bool]) -> str:
    hint = path_hint.lower()
    if lora_enabled is True:
        return "lora"
    if "lora" in hint:
        return "lora"
    if "ar768" in hint or "cascade" in hint:
        return "embed768_finetune"
    return "other_finetune"


def clean_run_name(path_hint: str, fallback: str) -> str:
    hint = path_hint.strip()
    if not hint:
        return fallback
    hint = hint.replace(str(ROOT) + "/", "")
    return hint


def parse_log_file(log_path: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    meta: Dict[str, object] = {
        "log_file": str(log_path.relative_to(ROOT)),
        "run_root": "",
        "run_name": log_path.stem,
        "run_type": "other_finetune",
        "lora_enabled": None,
        "has_error": 0,
        "lora_rank": None,
        "noise_std": None,
    }

    stage = "unknown"
    step_rows: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, object]] = []

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return meta, step_rows, epoch_rows

    for line in text.splitlines():
        m = STAGE_RE.search(line)
        if m:
            stage = m.group(1)

        m = OUT_ROOT_RE.search(line)
        if m:
            run_root = m.group(1).strip()
            meta["run_root"] = run_root
            meta["run_name"] = clean_run_name(run_root, log_path.stem)

        m = RUN_DIR_RE.search(line)
        if m and not meta.get("run_root"):
            run_dir = m.group(1).strip()
            # Collapse stage path to a run-root-like key.
            run_root = re.sub(r"/(stage_short|stage_medium|stage_long)$", "", run_dir)
            meta["run_root"] = run_root
            meta["run_name"] = clean_run_name(run_root, log_path.stem)

        m = RUNTIME_LORA_RE.search(line)
        if m:
            meta["lora_enabled"] = m.group(1) == "1"

        if LORA_ENABLED_LINE_RE.search(line):
            meta["lora_enabled"] = True

        m = LORA_RANK_RE.search(line)
        if m and meta.get("lora_rank") is None:
            try:
                meta["lora_rank"] = int(m.group(1))
            except ValueError:
                pass

        m = NOISE_RE.search(line)
        if m and meta.get("noise_std") is None:
            try:
                meta["noise_std"] = float(m.group(1))
            except ValueError:
                pass

        if ERROR_HINT_RE.search(line):
            meta["has_error"] = 1

        m = TRAIN_STEP_RE.search(line)
        if m:
            gstep = int(m.group(1))
            train_loss = safe_float(m.group(2))
            step_rows.append(
                {
                    "run_name": "",
                    "run_type": "",
                    "stage": stage,
                    "global_step": gstep,
                    "train_loss": train_loss,
                    "source": "log",
                    "log_file": str(log_path.relative_to(ROOT)),
                }
            )

        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
            train_loss = safe_float(m.group(2))
            val_loss = safe_float(m.group(3))
            epoch_rows.append(
                {
                    "run_name": "",
                    "run_type": "",
                    "stage": stage,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "global_step": None,
                    "source": "log",
                    "log_file": str(log_path.relative_to(ROOT)),
                }
            )

    run_hint = str(meta.get("run_root") or meta.get("run_name") or log_path.stem)
    run_type = classify_run(run_hint, meta.get("lora_enabled"))
    meta["run_type"] = run_type

    for row in step_rows:
        row["run_name"] = meta["run_name"]
        row["run_type"] = run_type
    for row in epoch_rows:
        row["run_name"] = meta["run_name"]
        row["run_type"] = run_type

    return meta, step_rows, epoch_rows


def discover_log_rows() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    log_meta_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, object]] = []

    for path in sorted(LOGS_DIR.glob("*.out")):
        if not is_finetune_log(path):
            continue
        meta, steps, epochs = parse_log_file(path)
        # Keep only clear finetuning classes.
        if meta.get("run_type") not in {"lora", "embed768_finetune"}:
            continue
        log_meta_rows.append(meta)
        step_rows.extend(steps)
        epoch_rows.extend(epochs)

    return log_meta_rows, step_rows, epoch_rows


def infer_run_name_from_epoch_csv(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    # .../<run_root>/<stage>/epoch_metrics.csv
    if len(parts) >= 3:
        return "/".join(parts[:-2])
    return str(rel)


def infer_stage_from_epoch_csv(path: Path) -> str:
    if len(path.parts) >= 2:
        return path.parent.name
    return "unknown"


def discover_epoch_csv_rows() -> List[Dict[str, object]]:
    candidates: List[Path] = []
    search_roots = [
        RESULTS_DIR / "fast_runs",
        RESULTS_DIR / "final_runs",
        RESULTS_DIR / "test_runs",
        RESULTS_DIR / "poster_runs",
    ]
    for root in search_roots:
        if root.exists():
            candidates.extend(root.glob("**/epoch_metrics.csv"))

    out_rows: List[Dict[str, object]] = []
    for csv_path in sorted(set(candidates)):
        run_name = infer_run_name_from_epoch_csv(csv_path)
        run_type = classify_run(run_name, None)
        if run_type not in {"lora", "embed768_finetune"}:
            continue

        stage = infer_stage_from_epoch_csv(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = safe_float(row.get("epoch"))
                epoch_num = int(epoch) if epoch is not None and not math.isnan(epoch) else None
                out_rows.append(
                    {
                        "run_name": run_name,
                        "run_type": run_type,
                        "stage": stage,
                        "epoch": epoch_num,
                        "train_loss": safe_float(row.get("train_loss")),
                        "val_loss": safe_float(row.get("val_loss")),
                        "global_step": safe_float(row.get("global_step")),
                        "source": "epoch_csv",
                        "log_file": "",
                        "csv_file": str(csv_path.relative_to(ROOT)),
                    }
                )
    return out_rows


def dedupe_epoch_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[str, str, Optional[int]], Dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("run_name")), str(row.get("stage")), row.get("epoch"))
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if current.get("source") != "epoch_csv" and row.get("source") == "epoch_csv":
            best[key] = row
    return sorted(best.values(), key=lambda r: (str(r.get("run_name")), str(r.get("stage")), r.get("epoch") or 0))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def finite(val: Optional[float]) -> bool:
    return val is not None and not (isinstance(val, float) and math.isnan(val))


def plot_train_step_curves(step_rows: List[Dict[str, object]], stage: str, out_path: Path) -> None:
    stage_rows = [r for r in step_rows if r.get("stage") == stage and finite(r.get("train_loss"))]
    if not stage_rows:
        return

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        grouped[str(row["run_name"])].append(row)

    plt.figure(figsize=(13, 7))
    cmap = plt.get_cmap("tab20")
    for idx, (run_name, rows) in enumerate(sorted(grouped.items())):
        rows_sorted = sorted(rows, key=lambda x: int(x["global_step"]))
        x = [int(r["global_step"]) for r in rows_sorted]
        y = [float(r["train_loss"]) for r in rows_sorted]
        run_type = rows_sorted[0]["run_type"]
        style = "-" if run_type == "lora" else "--"
        plt.plot(x, y, style, linewidth=1.8, alpha=0.9, color=cmap(idx % 20), label=run_name.split("/")[-1])

    plt.title(f"All Finetuning Train Loss Curves ({stage})")
    plt.xlabel("Global Step")
    plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_val_epoch_curves(epoch_rows: List[Dict[str, object]], stage: str, out_path: Path) -> None:
    stage_rows = [
        r
        for r in epoch_rows
        if r.get("stage") == stage and r.get("epoch") is not None and finite(r.get("val_loss"))
    ]
    if not stage_rows:
        return

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        grouped[str(row["run_name"])].append(row)

    plt.figure(figsize=(13, 7))
    cmap = plt.get_cmap("tab20")
    for idx, (run_name, rows) in enumerate(sorted(grouped.items())):
        rows_sorted = sorted(rows, key=lambda x: int(x["epoch"]))
        x = [int(r["epoch"]) for r in rows_sorted]
        y = [float(r["val_loss"]) for r in rows_sorted]
        run_type = rows_sorted[0]["run_type"]
        marker = "o" if run_type == "lora" else "s"
        plt.plot(x, y, marker=marker, linewidth=1.8, alpha=0.9, color=cmap(idx % 20), label=run_name.split("/")[-1])

    plt.title(f"All Finetuning Validation Loss Curves ({stage})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def best_val_by_run(epoch_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[str, str], float] = {}
    run_type_map: Dict[Tuple[str, str], str] = {}
    for r in epoch_rows:
        val = r.get("val_loss")
        if not finite(val):
            continue
        key = (str(r["run_name"]), str(r["stage"]))
        fv = float(val)
        if key not in best or fv < best[key]:
            best[key] = fv
            run_type_map[key] = str(r["run_type"])

    rows: List[Dict[str, object]] = []
    for (run_name, stage), val in sorted(best.items(), key=lambda x: x[1]):
        rows.append(
            {
                "run_name": run_name,
                "stage": stage,
                "run_type": run_type_map[(run_name, stage)],
                "best_val_loss": val,
            }
        )
    return rows


def plot_lora_vs_embed768_best_val(best_rows: List[Dict[str, object]], out_path: Path) -> None:
    if not best_rows:
        return

    filtered = [r for r in best_rows if r["run_type"] in {"lora", "embed768_finetune"}]
    filtered = filtered[:20]
    if not filtered:
        return

    labels = [f"{r['run_name'].split('/')[-1]}\n{r['stage']}" for r in filtered]
    vals = [float(r["best_val_loss"]) for r in filtered]
    colors = ["#2a9d8f" if r["run_type"] == "lora" else "#457b9d" for r in filtered]

    plt.figure(figsize=(14, 7))
    x = np.arange(len(filtered))
    plt.bar(x, vals, color=colors, alpha=0.9)
    plt.xticks(x, labels, rotation=65, ha="right", fontsize=8)
    plt.ylabel("Best Validation Loss")
    plt.title("LoRA vs Embed-768 Finetuning (Best Validation Loss)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_comparison_summaries() -> List[Dict[str, object]]:
    files = sorted((RESULTS_DIR / "poster_runs").glob("**/comparison_summary.csv"))
    files += sorted((RESULTS_DIR / "final_reports").glob("**/comparison_summary.csv"))

    rows: List[Dict[str, object]] = []
    for path in files:
        rel = str(path.relative_to(ROOT))
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "source_csv": rel,
                        "checkpoint": row.get("checkpoint", ""),
                        "rollout_steps": safe_float(row.get("rollout_steps")),
                        "eval_samples": safe_float(row.get("eval_samples")),
                        "mean_rmse_overall": safe_float(row.get("mean_rmse_overall")),
                        "mean_acc_overall": safe_float(row.get("mean_acc_overall")),
                        "rmse_lead_day_5_mean": safe_float(row.get("rmse_lead_day_5_mean")),
                        "acc_lead_day_5_mean": safe_float(row.get("acc_lead_day_5_mean")),
                    }
                )
    return rows


def pick_primary_comparison(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_csv"])].append(row)

    # Prefer the source with most checkpoint variants.
    best_source = max(grouped.items(), key=lambda item: len(item[1]))[0]
    return grouped[best_source]


def plot_embed768_eval_compare(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: str(r["checkpoint"]))
    labels = [str(r["checkpoint"]) for r in rows_sorted]
    rmse = [float(r["mean_rmse_overall"]) for r in rows_sorted if finite(r.get("mean_rmse_overall"))]
    acc = [float(r["mean_acc_overall"]) for r in rows_sorted if finite(r.get("mean_acc_overall"))]

    if len(rmse) != len(labels) or len(acc) != len(labels):
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width / 2, rmse, width=width, color="#1d3557", alpha=0.9, label="Mean RMSE")
    ax1.set_ylabel("Mean RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, acc, width=width, color="#2a9d8f", alpha=0.75, label="Mean ACC")
    ax2.set_ylabel("Mean ACC")

    ax1.set_title("Original Embed-768 vs Finetuned Checkpoints (Evaluation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def read_embed768_best_val() -> Optional[float]:
    candidates = [
        MODELS_PAPER_DIR / "pretrain" / "emb_768" / "metrics.json",
        RESULTS_DIR / "checkpoints_archive" / "fuxi_paper_prev" / "Models_paper" / "pretrain" / "emb_768" / "metrics.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            val = safe_float(obj.get("best_val_loss"))
            if finite(val):
                return float(val)
        except Exception:
            continue
    return None


def make_lora_vs_embed_table(best_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    embed_best = read_embed768_best_val()
    if finite(embed_best):
        out.append(
            {
                "model": "embed_768_original_pretrain",
                "run_type": "embed768_base",
                "stage": "pretrain",
                "best_val_loss": float(embed_best),
            }
        )

    lora_best = [r for r in best_rows if r.get("run_type") == "lora"]
    for row in lora_best:
        out.append(
            {
                "model": row["run_name"],
                "run_type": "lora",
                "stage": row["stage"],
                "best_val_loss": row["best_val_loss"],
            }
        )
    return out


def plot_lora_vs_embed_table(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    labels = [
        ("emb_768_base" if r["run_type"] == "embed768_base" else str(r["model"]).split("/")[-1])
        + f"\n{r['stage']}"
        for r in rows
    ]
    vals = [float(r["best_val_loss"]) for r in rows]
    colors = ["#264653" if r["run_type"] == "embed768_base" else "#2a9d8f" for r in rows]

    plt.figure(figsize=(14, 7))
    x = np.arange(len(rows))
    plt.bar(x, vals, color=colors, alpha=0.9)
    plt.xticks(x, labels, rotation=65, ha="right", fontsize=8)
    plt.ylabel("Best Validation Loss")
    plt.title("LoRA Runs vs Original Embed-768 (Best Validation Loss)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def copy_snapshot(files: Iterable[Path]) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for src in sorted(set(files)):
        if not src.exists() or not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        dst = RAW_SNAPSHOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append({"source": str(rel), "copied_to": str(dst.relative_to(ROOT))})
    return manifest


def build_run_summary(log_meta_rows: List[Dict[str, object]], best_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best_map: Dict[Tuple[str, str], float] = {
        (str(r["run_name"]), str(r["stage"])): float(r["best_val_loss"]) for r in best_rows
    }
    summary: List[Dict[str, object]] = []
    for meta in log_meta_rows:
        run_name = str(meta.get("run_name"))
        stages = sorted({k[1] for k in best_map if k[0] == run_name})
        if not stages:
            stages = ["unknown"]
        for stage in stages:
            summary.append(
                {
                    "run_name": run_name,
                    "run_type": meta.get("run_type", ""),
                    "stage": stage,
                    "best_val_loss": best_map.get((run_name, stage), ""),
                    "lora_enabled": meta.get("lora_enabled", ""),
                    "lora_rank": meta.get("lora_rank", ""),
                    "noise_std": meta.get("noise_std", ""),
                    "has_error_hint": meta.get("has_error", 0),
                    "log_file": meta.get("log_file", ""),
                }
            )
    summary.sort(key=lambda r: (str(r["run_type"]), str(r["run_name"]), str(r["stage"])))
    return summary


def write_report(
    log_meta_rows: List[Dict[str, object]],
    step_rows: List[Dict[str, object]],
    epoch_rows: List[Dict[str, object]],
    compare_rows_primary: List[Dict[str, object]],
    run_summary: List[Dict[str, object]],
) -> None:
    lora_runs = sorted({str(r["run_name"]) for r in run_summary if r.get("run_type") == "lora"})
    embed_runs = sorted({str(r["run_name"]) for r in run_summary if r.get("run_type") == "embed768_finetune"})

    valid_val = [float(r["val_loss"]) for r in epoch_rows if finite(r.get("val_loss"))]
    best_global_val = min(valid_val) if valid_val else None

    lines = [
        "# FuXi Finetuning Share Pack",
        "",
        "This folder was auto-generated for advisor sharing.",
        "",
        "## Included",
        "- LoRA vs original embed-768 comparison plots",
        "- Train/validation loss plots across all detected finetuning runs",
        "- CSV tables backing each plot",
        "- Raw copied logs and metric artifacts for traceability",
        "",
        "## Quick Stats",
        f"- Finetuning log files parsed: {len(log_meta_rows)}",
        f"- Train-step points parsed: {len(step_rows)}",
        f"- Epoch-level points parsed (deduped): {len(epoch_rows)}",
        f"- LoRA runs detected: {len(lora_runs)}",
        f"- Embed-768 finetune runs detected: {len(embed_runs)}",
    ]
    if best_global_val is not None:
        lines.append(f"- Best validation loss seen in parsed finetuning: {best_global_val:.6f}")

    if compare_rows_primary:
        lines += [
            "",
            "## Evaluation Comparison Source",
            f"- Source CSV: {compare_rows_primary[0]['source_csv']}",
            "- Checkpoints included: " + ", ".join(str(r["checkpoint"]) for r in compare_rows_primary),
        ]

    lines += [
        "",
        "## Key Files",
        "- data/finetune_step_losses.csv",
        "- data/finetune_epoch_losses.csv",
        "- data/finetune_run_summary.csv",
        "- data/lora_vs_embed768_summary.csv",
        "- data/eval_embed768_compare_all.csv",
        "- plots/finetune_all_train_loss_curves_stage_short.png",
        "- plots/finetune_all_train_loss_curves_stage_medium.png",
        "- plots/finetune_all_val_loss_curves_stage_short.png",
        "- plots/finetune_all_val_loss_curves_stage_medium.png",
        "- plots/lora_vs_embed768_best_val_loss.png",
        "- plots/original_embed768_eval_comparison.png",
    ]

    (REPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfc",
        }
    )

    log_meta_rows, step_rows_log, epoch_rows_log = discover_log_rows()
    epoch_rows_csv = discover_epoch_csv_rows()
    epoch_rows = dedupe_epoch_rows([*epoch_rows_log, *epoch_rows_csv])

    step_rows = [r for r in step_rows_log if finite(r.get("train_loss"))]

    # Persist raw tables.
    write_csv(
        DATA_DIR / "finetune_step_losses.csv",
        step_rows,
        ["run_name", "run_type", "stage", "global_step", "train_loss", "source", "log_file"],
    )
    write_csv(
        DATA_DIR / "finetune_epoch_losses.csv",
        epoch_rows,
        ["run_name", "run_type", "stage", "epoch", "global_step", "train_loss", "val_loss", "source", "log_file", "csv_file"],
    )

    best_rows = best_val_by_run(epoch_rows)
    write_csv(
        DATA_DIR / "finetune_best_val_by_run_stage.csv",
        best_rows,
        ["run_name", "run_type", "stage", "best_val_loss"],
    )

    run_summary = build_run_summary(log_meta_rows, best_rows)
    write_csv(
        DATA_DIR / "finetune_run_summary.csv",
        run_summary,
        [
            "run_name",
            "run_type",
            "stage",
            "best_val_loss",
            "lora_enabled",
            "lora_rank",
            "noise_std",
            "has_error_hint",
            "log_file",
        ],
    )

    # Evaluation comparisons (base vs finetuned checkpoints).
    compare_rows_all = parse_comparison_summaries()
    write_csv(
        DATA_DIR / "eval_embed768_compare_all.csv",
        compare_rows_all,
        [
            "source_csv",
            "checkpoint",
            "rollout_steps",
            "eval_samples",
            "mean_rmse_overall",
            "mean_acc_overall",
            "rmse_lead_day_5_mean",
            "acc_lead_day_5_mean",
        ],
    )
    compare_rows_primary = pick_primary_comparison(compare_rows_all)
    write_csv(
        DATA_DIR / "eval_embed768_compare_primary.csv",
        compare_rows_primary,
        [
            "source_csv",
            "checkpoint",
            "rollout_steps",
            "eval_samples",
            "mean_rmse_overall",
            "mean_acc_overall",
            "rmse_lead_day_5_mean",
            "acc_lead_day_5_mean",
        ],
    )

    # LoRA vs original emb_768 table.
    lora_vs_embed = make_lora_vs_embed_table(best_rows)
    write_csv(
        DATA_DIR / "lora_vs_embed768_summary.csv",
        lora_vs_embed,
        ["model", "run_type", "stage", "best_val_loss"],
    )

    # Plots.
    plot_train_step_curves(
        step_rows,
        "stage_short",
        PLOTS_DIR / "finetune_all_train_loss_curves_stage_short.png",
    )
    plot_train_step_curves(
        step_rows,
        "stage_medium",
        PLOTS_DIR / "finetune_all_train_loss_curves_stage_medium.png",
    )
    plot_val_epoch_curves(
        epoch_rows,
        "stage_short",
        PLOTS_DIR / "finetune_all_val_loss_curves_stage_short.png",
    )
    plot_val_epoch_curves(
        epoch_rows,
        "stage_medium",
        PLOTS_DIR / "finetune_all_val_loss_curves_stage_medium.png",
    )
    plot_lora_vs_embed768_best_val(
        best_rows,
        PLOTS_DIR / "lora_vs_embed768_best_val_by_stage.png",
    )
    plot_lora_vs_embed_table(
        lora_vs_embed,
        PLOTS_DIR / "lora_vs_embed768_best_val_loss.png",
    )
    plot_embed768_eval_compare(
        compare_rows_primary,
        PLOTS_DIR / "original_embed768_eval_comparison.png",
    )

    # Copy raw evidence files into prof/raw_snapshot.
    snapshot_sources: List[Path] = []

    for row in log_meta_rows:
        log_path = ROOT / str(row["log_file"])
        snapshot_sources.append(log_path)
        err_path = log_path.with_suffix(".err")
        if err_path.exists():
            snapshot_sources.append(err_path)

    for row in epoch_rows:
        csv_rel = row.get("csv_file")
        if csv_rel:
            snapshot_sources.append(ROOT / str(csv_rel))

    for row in compare_rows_all:
        snapshot_sources.append(ROOT / str(row["source_csv"]))

    emb_json_candidates = [
        MODELS_PAPER_DIR / "pretrain" / "emb_768" / "metrics.json",
        RESULTS_DIR / "checkpoints_archive" / "fuxi_paper_prev" / "Models_paper" / "pretrain" / "emb_768" / "metrics.json",
    ]
    snapshot_sources.extend([p for p in emb_json_candidates if p.exists()])

    manifest = copy_snapshot(snapshot_sources)
    write_csv(DATA_DIR / "raw_snapshot_manifest.csv", manifest, ["source", "copied_to"])

    write_report(log_meta_rows, step_rows, epoch_rows, compare_rows_primary, run_summary)

    print("Built professor bundle at:", PROF)
    print("Plots:", PLOTS_DIR)
    print("Data:", DATA_DIR)
    print("Report:", REPORT_DIR / "README.md")


if __name__ == "__main__":
    main()
