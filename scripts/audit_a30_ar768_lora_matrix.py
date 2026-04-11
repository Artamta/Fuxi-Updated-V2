#!/usr/bin/env python3
"""Audit AR768 A30 LoRA matrix runs and produce ranking/report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STAGES: Tuple[str, ...] = ("stage_short", "stage_medium", "stage_long")
REQUIRED_STAGE_FILES: Tuple[str, ...] = (
    "best.pt",
    "last.pt",
    "config.json",
    "metrics.json",
    "epoch_metrics.csv",
    "loss_curve.png",
)

WARN_RE = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)
ERROR_RE = re.compile(r"\berror\b", re.IGNORECASE)
PATTERNS = {
    "traceback_hits": re.compile(r"Traceback \(most recent call last\):"),
    "oom_hits": re.compile(r"cuda out of memory|out of memory", re.IGNORECASE),
    "nan_inf_hits": re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE),
    "timeout_hits": re.compile(r"time limit|timeout|timed out", re.IGNORECASE),
    "runtime_error_hits": re.compile(r"RuntimeError", re.IGNORECASE),
    "assertion_error_hits": re.compile(r"AssertionError", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit A30 LoRA matrix submissions")
    p.add_argument("--submissions-csv", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fail-on-incomplete", action="store_true")
    return p.parse_args()


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_epoch_metrics(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "epoch_count": 0,
        "min_val_loss": None,
        "min_val_epoch": None,
        "last_val_loss": None,
    }
    if not path.is_file():
        return out

    min_val_loss: Optional[float] = None
    min_val_epoch: Optional[int] = None
    last_val_loss: Optional[float] = None
    epoch_count = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_count += 1
            epoch_num = safe_float(row.get("epoch"))
            val_loss = safe_float(row.get("val_loss"))
            if val_loss is not None:
                last_val_loss = val_loss
                if min_val_loss is None or val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_epoch = int(epoch_num) if epoch_num is not None else None

    out["epoch_count"] = epoch_count
    out["min_val_loss"] = min_val_loss
    out["min_val_epoch"] = min_val_epoch
    out["last_val_loss"] = last_val_loss
    return out


def pick_long_metrics(compare_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not compare_csv.is_file():
        return None, None, None

    rows = read_csv_rows(compare_csv)
    if not rows:
        return None, None, None

    target_names = {"stage_long", "long", "stage_long_ft"}
    preferred: Optional[Dict[str, str]] = None
    fallback: Optional[Dict[str, str]] = None

    for row in rows:
        ckpt_name = str(row.get("checkpoint", "")).strip()
        if ckpt_name in target_names:
            preferred = row
            break
        if fallback is None and ckpt_name.lower() != "base":
            fallback = row

    selected = preferred or fallback
    if selected is None:
        return None, None, None

    rmse = safe_float(selected.get("mean_rmse_overall"))
    if rmse is None:
        rmse = safe_float(selected.get("mean_rmse"))
    acc = safe_float(selected.get("mean_acc_overall"))
    if acc is None:
        acc = safe_float(selected.get("mean_acc"))

    return rmse, acc, selected.get("checkpoint")


def scan_logs(log_paths: Sequence[Path]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    counts = {
        "warning_lines": 0,
        "error_lines": 0,
        "traceback_hits": 0,
        "oom_hits": 0,
        "nan_inf_hits": 0,
        "timeout_hits": 0,
        "runtime_error_hits": 0,
        "assertion_error_hits": 0,
    }
    snippets: Dict[str, List[str]] = {k: [] for k in counts if k not in {"warning_lines", "error_lines"}}

    for log_path in log_paths:
        if not log_path.is_file():
            continue
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if WARN_RE.search(stripped):
                        counts["warning_lines"] += 1
                    if ERROR_RE.search(stripped):
                        counts["error_lines"] += 1
                    for key, pattern in PATTERNS.items():
                        if pattern.search(stripped):
                            counts[key] += 1
                            if len(snippets[key]) < 3:
                                snippets[key].append(f"{log_path}:{stripped}")
        except Exception:
            continue

    return counts, snippets


def audit_row(row: Dict[str, str]) -> Dict[str, object]:
    variant = row.get("variant", "")
    expected_output_root = Path(row.get("expected_output_root", ""))
    slurm_out_log = Path(row.get("slurm_out_log", ""))
    slurm_err_log = Path(row.get("slurm_err_log", ""))

    result: Dict[str, object] = {
        "variant": variant,
        "job_id": row.get("job_id", ""),
        "array_task_id": row.get("array_task_id", ""),
        "profile": row.get("profile", ""),
        "noise_std": row.get("noise_std", ""),
        "lora_rank": row.get("lora_rank", ""),
        "lora_alpha": row.get("lora_alpha", ""),
        "run_root_base": row.get("run_root_base", ""),
        "expected_output_root": str(expected_output_root),
        "selected_checkpoint": str(expected_output_root / "stage_long" / "best.pt"),
        "compare_summary": str(expected_output_root / "poster_eval_compare_quick" / "comparison_summary.csv"),
    }

    output_exists = expected_output_root.is_dir()
    result["output_root_exists"] = int(output_exists)

    stage_complete_flags: List[bool] = []
    for stage in STAGES:
        stage_dir = expected_output_root / stage
        has_best = (stage_dir / "best.pt").is_file()
        has_metrics = (stage_dir / "metrics.json").is_file()
        stage_complete = has_best and has_metrics
        stage_complete_flags.append(stage_complete)
        result[f"{stage}_complete"] = int(stage_complete)

        for file_name in REQUIRED_STAGE_FILES:
            key = f"{stage}_{file_name.replace('.', '_')}_exists"
            result[key] = int((stage_dir / file_name).is_file())

        epoch_info = parse_epoch_metrics(stage_dir / "epoch_metrics.csv")
        result[f"{stage}_epoch_count"] = epoch_info["epoch_count"]
        result[f"{stage}_min_val_loss"] = epoch_info["min_val_loss"]
        result[f"{stage}_min_val_epoch"] = epoch_info["min_val_epoch"]
        result[f"{stage}_last_val_loss"] = epoch_info["last_val_loss"]

        stage_metrics = read_json(stage_dir / "metrics.json")
        result[f"{stage}_test_loss"] = safe_float(stage_metrics.get("test_loss"))
        result[f"{stage}_test_mae"] = safe_float(stage_metrics.get("test_mae"))
        result[f"{stage}_best_val_loss"] = safe_float(stage_metrics.get("best_val_loss"))

    rmse, acc, metric_checkpoint = pick_long_metrics(expected_output_root / "poster_eval_compare_quick" / "comparison_summary.csv")
    result["metric_checkpoint"] = metric_checkpoint or ""
    result["mean_rmse_overall"] = rmse
    result["mean_acc_overall"] = acc

    log_counts, log_snippets = scan_logs([slurm_out_log, slurm_err_log])
    result.update(log_counts)
    result["slurm_out_log"] = str(slurm_out_log)
    result["slurm_err_log"] = str(slurm_err_log)
    result["log_snippets"] = log_snippets

    stages_complete = all(stage_complete_flags)
    scored = rmse is not None
    critical_log_hits = int(log_counts["traceback_hits"]) + int(log_counts["oom_hits"]) + int(log_counts["runtime_error_hits"]) + int(log_counts["assertion_error_hits"])

    if not output_exists:
        status = "missing_output_root"
    elif stages_complete and scored:
        status = "complete_scored"
    elif stages_complete:
        status = "complete_unscored"
    else:
        status = "incomplete"

    if critical_log_hits > 0:
        status = f"{status}_with_log_issues"

    result["status"] = status
    result["critical_log_hits"] = critical_log_hits
    return result


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "variant",
        "job_id",
        "array_task_id",
        "profile",
        "status",
        "noise_std",
        "lora_rank",
        "lora_alpha",
        "output_root_exists",
        "stage_short_complete",
        "stage_medium_complete",
        "stage_long_complete",
        "stage_short_epoch_count",
        "stage_medium_epoch_count",
        "stage_long_epoch_count",
        "stage_short_min_val_loss",
        "stage_medium_min_val_loss",
        "stage_long_min_val_loss",
        "stage_long_test_mae",
        "stage_long_test_loss",
        "mean_rmse_overall",
        "mean_acc_overall",
        "warning_lines",
        "error_lines",
        "traceback_hits",
        "oom_hits",
        "nan_inf_hits",
        "timeout_hits",
        "runtime_error_hits",
        "assertion_error_hits",
        "critical_log_hits",
        "expected_output_root",
        "selected_checkpoint",
        "compare_summary",
        "slurm_out_log",
        "slurm_err_log",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def pick_best_variant(rows: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    scored = []
    for row in rows:
        rmse = safe_float(row.get("mean_rmse_overall"))
        acc = safe_float(row.get("mean_acc_overall"))
        if rmse is None:
            continue
        score = (rmse, -(acc if acc is not None else -1.0))
        scored.append((score, row))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def render_report(path: Path, rows: Sequence[Dict[str, object]], best: Optional[Dict[str, object]]) -> None:
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    lines: List[str] = []
    lines.append("# A30 LoRA Matrix Audit Report")
    lines.append("")
    lines.append(f"Total variants: {len(rows)}")
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines.append("")
    if best is not None:
        lines.append("## Selected best variant (lowest mean RMSE overall)")
        lines.append("")
        lines.append(f"- Variant: {best.get('variant', '')}")
        lines.append(f"- Job ID: {best.get('job_id', '')}")
        lines.append(f"- Mean RMSE overall: {best.get('mean_rmse_overall', '')}")
        lines.append(f"- Mean ACC overall: {best.get('mean_acc_overall', '')}")
        lines.append(f"- Checkpoint: {best.get('selected_checkpoint', '')}")
    else:
        lines.append("## Selected best variant")
        lines.append("")
        lines.append("No scored variants yet (comparison summary missing or incomplete).")

    lines.append("")
    lines.append("## Variant table")
    lines.append("")
    lines.append("| Variant | Job | Status | RMSE | ACC | Warn | Err | Output root |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {variant} | {job} | {status} | {rmse} | {acc} | {warn} | {err} | {out} |".format(
                variant=row.get("variant", ""),
                job=row.get("job_id", ""),
                status=row.get("status", ""),
                rmse=row.get("mean_rmse_overall", ""),
                acc=row.get("mean_acc_overall", ""),
                warn=row.get("warning_lines", ""),
                err=row.get("error_lines", ""),
                out=row.get("expected_output_root", ""),
            )
        )

    issues = [r for r in rows if int(r.get("critical_log_hits", 0) or 0) > 0]
    lines.append("")
    lines.append("## Critical log issues")
    lines.append("")
    if not issues:
        lines.append("No critical log signatures detected (traceback/OOM/runtime/assertion).")
    else:
        for row in issues:
            lines.append(
                f"- {row.get('variant', '')}: critical_hits={row.get('critical_log_hits', 0)}, out={row.get('slurm_out_log', '')}, err={row.get('slurm_err_log', '')}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    submissions_csv = args.submissions_csv.expanduser().resolve()
    if not submissions_csv.is_file():
        raise FileNotFoundError(f"Submissions CSV not found: {submissions_csv}")

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else submissions_csv.parent / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_rows = read_csv_rows(submissions_csv)
    audited_rows = [audit_row(row) for row in submission_rows]

    summary_csv = out_dir / "matrix_summary.csv"
    summary_json = out_dir / "matrix_summary.json"
    report_md = out_dir / "matrix_report.md"

    write_summary_csv(summary_csv, audited_rows)
    summary_json.write_text(json.dumps(audited_rows, indent=2), encoding="utf-8")

    best = pick_best_variant(audited_rows)
    render_report(report_md, audited_rows, best)

    print(f"Wrote summary CSV : {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote report      : {report_md}")
    if best is not None:
        print(
            "Best variant      : "
            f"{best.get('variant', '')} "
            f"(rmse={best.get('mean_rmse_overall', '')}, acc={best.get('mean_acc_overall', '')})"
        )
    else:
        print("Best variant      : unavailable (no scored variants)")

    if args.fail_on_incomplete:
        incomplete = [r for r in audited_rows if str(r.get("status", "")).startswith("incomplete") or str(r.get("status", "")) == "missing_output_root"]
        if incomplete:
            print(f"Found {len(incomplete)} incomplete variants.")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
