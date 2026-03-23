#!/usr/bin/env bash
set -euo pipefail

# Simple 2-GPU checkpoint comparison launcher.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh" ]]; then
  source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
  conda activate weather_forecast || true
fi

EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/eval}"
GPU_IDS="${GPU_IDS:-0,1}"
AMP_MODE="${AMP_MODE:-bf16}"
ROLL_OUT_STEPS="${ROLL_OUT_STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAP_VARIABLE="${MAP_VARIABLE:-2m_temperature}"
MAP_LEAD_STEP="${MAP_LEAD_STEP:-20}"

CKPT_768="${CKPT_768:-${ROOT_DIR}/Models_paper/pretrain/emb_768/best.pt}"
CKPT_1024="${CKPT_1024:-${ROOT_DIR}/Models_paper/pretrain/emb_1024/best.pt}"

if [[ ! -f "${CKPT_768}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_768}"
  exit 1
fi
if [[ ! -f "${CKPT_1024}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_1024}"
  exit 1
fi

CMD=(
  python -m src.evaluation.eval_compare_checkpoints
  --checkpoints
  "emb_768=${CKPT_768}"
  "emb_1024=${CKPT_1024}"
  --eval-root "${EVAL_ROOT}"
  --device cuda
  --gpus "${GPU_IDS}"
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

echo "Running 2-GPU comparison on GPUs ${GPU_IDS}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done. Outputs: ${EVAL_ROOT}"
