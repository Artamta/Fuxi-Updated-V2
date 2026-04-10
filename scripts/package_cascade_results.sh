#!/usr/bin/env bash
set -euo pipefail

# Collect checkpoints + CSV summaries + spectra into a clean, shareable directory.
# Intentionally excludes heatmaps.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

RUN_ROOT=${1:-"results/poster_runs/ar768_cascade_job7707"}
OUT_DIR=${2:-"results/final_reports/$(basename "${RUN_ROOT}")_pack"}
mkdir -p "${OUT_DIR}" "${OUT_DIR}/checkpoints" "${OUT_DIR}/metrics" "${OUT_DIR}/spectra"

copy_if_exists() {
  local src=$1
  local dst=$2
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
}

# Checkpoints
copy_if_exists "${RUN_ROOT}/stage_short/best.pt" "${OUT_DIR}/checkpoints/fuxi_stage_short_best.pt"
copy_if_exists "${RUN_ROOT}/stage_medium/best.pt" "${OUT_DIR}/checkpoints/fuxi_stage_medium_best.pt"
copy_if_exists "${RUN_ROOT}/stage_long/best.pt" "${OUT_DIR}/checkpoints/fuxi_stage_long_best.pt"

# Stage metrics if present
copy_if_exists "${RUN_ROOT}/stage_short/metrics.json" "${OUT_DIR}/metrics/stage_short_metrics.json"
copy_if_exists "${RUN_ROOT}/stage_medium/metrics.json" "${OUT_DIR}/metrics/stage_medium_metrics.json"
copy_if_exists "${RUN_ROOT}/stage_long/metrics.json" "${OUT_DIR}/metrics/stage_long_metrics.json"

# Poster compare outputs (no heatmaps copied)
for compare_root in "${RUN_ROOT}/poster_eval_compare" "${RUN_ROOT}/poster_eval_compare_quick"; do
  if [[ -d "${compare_root}" ]]; then
    tag=$(basename "${compare_root}")
    mkdir -p "${OUT_DIR}/metrics/${tag}" "${OUT_DIR}/spectra/${tag}"

    copy_if_exists "${compare_root}/comparison_summary.csv" "${OUT_DIR}/metrics/${tag}/comparison_summary.csv"

    for curve in \
      "comparison_rmse_acc.png" \
      "comparison_rmse_geopotential_plev500.png" \
      "comparison_acc_geopotential_plev500.png" \
      "comparison_rmse_2m_temperature.png" \
      "comparison_acc_2m_temperature.png" \
      "comparison_rmse_surface_pressure.png" \
      "comparison_acc_surface_pressure.png"; do
      copy_if_exists "${compare_root}/${curve}" "${OUT_DIR}/metrics/${tag}/${curve}"
    done

    for stage in base stage_short stage_medium stage_long; do
      stage_dir="${compare_root}/checkpoint_${stage}"
      if [[ -d "${stage_dir}" ]]; then
        mkdir -p "${OUT_DIR}/metrics/${tag}/${stage}" "${OUT_DIR}/spectra/${tag}/${stage}"
        copy_if_exists "${stage_dir}/summary.json" "${OUT_DIR}/metrics/${tag}/${stage}/summary.json"
        copy_if_exists "${stage_dir}/metrics_per_lead.csv" "${OUT_DIR}/metrics/${tag}/${stage}/metrics_per_lead.csv"
        copy_if_exists "${stage_dir}/metrics_per_day.csv" "${OUT_DIR}/metrics/${tag}/${stage}/metrics_per_day.csv"
        copy_if_exists "${stage_dir}/selected_rmse_curves.png" "${OUT_DIR}/metrics/${tag}/${stage}/selected_rmse_curves.png"
        copy_if_exists "${stage_dir}/selected_acc_curves.png" "${OUT_DIR}/metrics/${tag}/${stage}/selected_acc_curves.png"

        if [[ -d "${stage_dir}/spectra" ]]; then
          find "${stage_dir}/spectra" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.png" \) \
            -exec cp {} "${OUT_DIR}/spectra/${tag}/${stage}/" \;
        fi
      fi
    done
  fi
done

cat > "${OUT_DIR}/README.md" <<EOF
# Packaged Cascade Results

Source run root: ${RUN_ROOT}

This package includes:
- checkpoints (short/medium/long when available)
- CSV metric summaries (per-lead, per-day, comparison)
- ACC/RMSE curve plots
- power spectra CSV/plots

Excluded intentionally:
- heatmap plots

Folder map:
- checkpoints/
- metrics/
- spectra/
EOF

echo "Packaged results -> ${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 4 -type f | sort
