#!/bin/bash
#SBATCH --job-name=fuxi_ar
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/fuxi_ar_%j.out
#SBATCH -e logs/fuxi_ar_%j.err

# Single-node autoregressive FuXi training (multi-GPU DataParallel).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_common.sh"

ROOT_DIR=$(slurm_root)
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

slurm_activate_conda

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
slurm_require_zarr "${ZARR_STORE}"

# Horizon / schedule
FORECAST_STEPS="${FORECAST_STEPS:-20}"
TEACHER_FORCING="${TEACHER_FORCING:-0.0}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
MAX_ITERS="${MAX_ITERS:-}"  # empty means no cap
PATIENCE="${PATIENCE:-8}"
EVAL_EVERY="${EVAL_EVERY:-1}"

# Optimizer
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Model overrides (optional)
PRESET="${PRESET:-paper}"
EMBED_DIM="${EMBED_DIM:-}"
WINDOW_SIZE="${WINDOW_SIZE:-}"
DROP_PATH="${DROP_PATH:-}"
NUM_HEADS="${NUM_HEADS:-}"
DEPTH_PRE="${DEPTH_PRE:-}"
DEPTH_MID="${DEPTH_MID:-}"
DEPTH_POST="${DEPTH_POST:-}"
MLP_RATIO="${MLP_RATIO:-}"
MC_DROPOUT="${MC_DROPOUT:-}"
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"  # set to 0 to disable

# Data + batching
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HISTORY_STEPS="${HISTORY_STEPS:-2}"
TRAIN_START="${TRAIN_START:-1979-01-01}"
TRAIN_END="${TRAIN_END:-2015-12-31}"
VAL_START="${VAL_START:-2016-01-01}"
VAL_END="${VAL_END:-2017-12-31}"
TEST_START="${TEST_START:-2018-01-01}"
TEST_END="${TEST_END:-2018-12-31}"
PRESSURE_VARS="${PRESSURE_VARS:-}"
SURFACE_VARS="${SURFACE_VARS:-}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-}"

# I/O
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
EXP_NAME="${EXP_NAME:-ar_h${FORECAST_STEPS}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${RUN_DIR}"

# Resume logic
RESUME_FLAG=""
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  RESUME_FLAG="--resume ${RUN_DIR}/last.pt"
fi
if [[ -n "${RESUME_CKPT:-}" ]]; then
  RESUME_FLAG="--resume ${RESUME_CKPT}"
fi

NGPUS=$(slurm_detect_gpus)

slurm_export_nccl_defaults

echo "============================================================"
echo "FuXi autoregressive (DataParallel)"
echo "Host        : $(hostname)"
echo "GPUs        : ${NGPUS}"
echo "Experiment  : ${EXP_NAME}"
echo "Zarr        : ${ZARR_STORE}"
echo "Horizon     : ${FORECAST_STEPS} (teacher_forcing=${TEACHER_FORCING})"
echo "Batch       : ${BATCH_SIZE}"
echo "Resume      : ${RESUME_FLAG:-<none>}"
echo "============================================================"
nvidia-smi

python -m src.training.train_autoregressive \
  --zarr-store "${ZARR_STORE}" \
  --train-start "${TRAIN_START}" \
  --train-end "${TRAIN_END}" \
  --val-start "${VAL_START}" \
  --val-end "${VAL_END}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --history-steps "${HISTORY_STEPS}" \
  --forecast-steps "${FORECAST_STEPS}" \
  --teacher-forcing "${TEACHER_FORCING}" \
  --max-epochs "${MAX_EPOCHS}" \
  ${MAX_ITERS:+--max-iters "${MAX_ITERS}"} \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --patience "${PATIENCE}" \
  --eval-every "${EVAL_EVERY}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --grad-clip "${GRAD_CLIP}" \
  --preset "${PRESET}" \
  ${EMBED_DIM:+--embed-dim "${EMBED_DIM}"} \
  ${WINDOW_SIZE:+--window-size "${WINDOW_SIZE}"} \
  ${DROP_PATH:+--drop-path-rate "${DROP_PATH}"} \
  ${NUM_HEADS:+--num-heads "${NUM_HEADS}"} \
  ${DEPTH_PRE:+--depth-pre "${DEPTH_PRE}"} \
  ${DEPTH_MID:+--depth-mid "${DEPTH_MID}"} \
  ${DEPTH_POST:+--depth-post "${DEPTH_POST}"} \
  ${MLP_RATIO:+--mlp-ratio "${MLP_RATIO}"} \
  ${MC_DROPOUT:+--mc-dropout "${MC_DROPOUT}"} \
  $( [[ "${USE_CHECKPOINT}" == "0" ]] || echo --use-checkpoint ) \
  ${PRESSURE_VARS:+--pressure-vars "${PRESSURE_VARS}"} \
  ${SURFACE_VARS:+--surface-vars "${SURFACE_VARS}"} \
  ${PRESSURE_LEVELS:+--pressure-levels "${PRESSURE_LEVELS}"} \
  --output-root "${OUTPUT_ROOT}" \
  --exp-name "${EXP_NAME}" \
  --fp16 \
  --gpus "${NGPUS}" \
  ${RESUME_FLAG}

echo "Done. Outputs: ${RUN_DIR}"
