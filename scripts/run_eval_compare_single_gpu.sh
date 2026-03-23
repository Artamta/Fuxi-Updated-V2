#!/usr/bin/env bash
set -euo pipefail

# Simple single-GPU checkpoint comparison launcher.
# Checkpoints are passed via env vars so this script stays reusable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh" ]]; then
  source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
  conda activate weather_forecast || true
fi

EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/eval}"
GPU_ID="${GPU_ID:-0}"
AMP_MODE="${AMP_MODE:-bf16}"
ROLL_OUT_STEPS="${ROLL_OUT_STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAP_VARIABLE="${MAP_VARIABLE:-2m_temperature}"
MAP_LEAD_STEP="${MAP_LEAD_STEP:-20}"

CKPT_A="${CKPT_A:-${ROOT_DIR}/Models_paper/pretrain/emb_768/best.pt}"
CKPT_B="${CKPT_B:-${ROOT_DIR}/Models_paper/pretrain/emb_1024/best.pt}"

if [[ ! -f "${CKPT_A}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_A}"
  exit 1
fi
if [[ ! -f "${CKPT_B}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_B}"
  exit 1
fi

CMD=(
  python -m src.evaluation.eval_compare_checkpoints
  --checkpoints
  "emb_768=${CKPT_A}"
  "emb_1024=${CKPT_B}"
  --eval-root "${EVAL_ROOT}"
  --device cuda
  --gpus "${GPU_ID}"
  --amp "${AMP_MODE}"
  --rollout-steps "${ROLL_OUT_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --stats-samples 256
  --save-n-samples 1
  --map-variable "${MAP_VARIABLE}"
  --map-lead-step "${MAP_LEAD_STEP}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Running single-GPU comparison on GPU ${GPU_ID}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done. Outputs: ${EVAL_ROOT}"
