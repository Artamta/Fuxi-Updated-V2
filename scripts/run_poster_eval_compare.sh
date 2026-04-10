#!/usr/bin/env bash
set -euo pipefail

# Simple poster comparison runner.
# It evaluates available checkpoints (base/short/medium/long) and produces:
# - RMSE/ACC comparison tables and plots
# - Isotropic power spectra plots from saved prediction samples

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

RUN_ROOT=${1:-"results/poster_runs/ar768_cascade_job7608"}
OUTPUT_ROOT=${2:-"${RUN_ROOT}/poster_eval_compare"}

PYTHON_BIN=${PYTHON_BIN:-"/home/raj.ayush/.conda/envs/weather_forecast/bin/python"}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

ROLL_STEPS=${ROLL_STEPS:-60}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SAMPLES=${MAX_SAMPLES:-256}
SAVE_N_SAMPLES=${SAVE_N_SAMPLES:-24}
AMP_MODE=${AMP_MODE:-bf16}
DEVICE_MODE=${DEVICE_MODE:-cuda}
NO_HEATMAPS=${NO_HEATMAPS:-1}

BASE_CKPT=${BASE_CKPT:-"results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt"}
SHORT_CKPT="${RUN_ROOT}/stage_short/best.pt"
MEDIUM_CKPT="${RUN_ROOT}/stage_medium/best.pt"
LONG_CKPT="${RUN_ROOT}/stage_long/best.pt"

CHECKPOINT_SPECS=()
add_if_exists() {
  local name=$1
  local path=$2
  if [[ -f "${path}" ]]; then
    CHECKPOINT_SPECS+=("${name}=${path}")
  else
    echo "[warn] missing checkpoint for ${name}: ${path}"
  fi
}

add_if_exists "base" "${BASE_CKPT}"
add_if_exists "stage_short" "${SHORT_CKPT}"
add_if_exists "stage_medium" "${MEDIUM_CKPT}"
add_if_exists "stage_long" "${LONG_CKPT}"

if (( ${#CHECKPOINT_SPECS[@]} == 0 )); then
  echo "ERROR: no checkpoints found to evaluate" >&2
  exit 1
fi

echo "============================================================"
echo "FuXi poster comparison evaluation"
echo "Run root     : ${RUN_ROOT}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Checkpoints  : ${CHECKPOINT_SPECS[*]}"
echo "No heatmaps  : ${NO_HEATMAPS}"
echo "============================================================"

EXTRA_ARGS=()
if [[ "${NO_HEATMAPS}" == "1" ]]; then
  EXTRA_ARGS+=(--no-heatmaps)
fi

"${PYTHON_BIN}" -m src.evaluation.evaluate_poster_compare \
  --checkpoints "${CHECKPOINT_SPECS[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --python "${PYTHON_BIN}" \
  --rollout-steps "${ROLL_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --save-n-samples "${SAVE_N_SAMPLES}" \
  --amp "${AMP_MODE}" \
  --device "${DEVICE_MODE}" \
  "${EXTRA_ARGS[@]}" \
  --skip-existing

echo "Done. Poster comparison outputs: ${OUTPUT_ROOT}"
