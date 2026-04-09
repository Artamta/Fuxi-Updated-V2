#!/bin/bash
#SBATCH --job-name=fuxi_cascade_tiny
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/fuxi_cascade_tiny_%j.out
#SBATCH -e logs/fuxi_cascade_tiny_%j.err

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_common.sh"

ROOT_DIR=$(slurm_root)
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

slurm_activate_conda

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
slurm_require_zarr "${ZARR_STORE}"

BASE_RUNS_DIR="${BASE_RUNS_DIR:-results/pretrain_tiny}"
BASE_EXP_NAME="${BASE_EXP_NAME:-tiny_base}"
BASE_CKPT="${BASE_RUNS_DIR}/${BASE_EXP_NAME}/best.pt"

CONFIG_SHORT="${CONFIG_SHORT:-configs/cascade_tiny_short.yaml}"
CONFIG_MEDIUM="${CONFIG_MEDIUM:-configs/cascade_tiny_medium.yaml}"
CONFIG_LONG="${CONFIG_LONG:-configs/cascade_tiny_long.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/cascade_tiny}"
CKPT_NAME="${CKPT_NAME:-best.pt}"
EXP_SHORT="${EXP_SHORT:-stage_short}"
EXP_MEDIUM="${EXP_MEDIUM:-stage_medium}"
EXP_LONG="${EXP_LONG:-stage_long}"

echo "============================================================"
echo "FuXi tiny cascade smoke/submission job"
echo "Host        : $(hostname)"
echo "Zarr        : ${ZARR_STORE}"
echo "Base run dir: ${BASE_RUNS_DIR}/${BASE_EXP_NAME}"
echo "Cascade out : ${OUTPUT_ROOT}"
echo "============================================================"

python -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start "${PRETRAIN_TRAIN_START:-1979-01-01}" \
  --train-end "${PRETRAIN_TRAIN_END:-1979-01-05}" \
  --val-start "${PRETRAIN_VAL_START:-2016-01-01}" \
  --val-end "${PRETRAIN_VAL_END:-2016-01-03}" \
  --test-start "${PRETRAIN_TEST_START:-2018-01-01}" \
  --test-end "${PRETRAIN_TEST_END:-2018-01-03}" \
  --pressure-vars "temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind" \
  --surface-vars "2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_column_water_vapour" \
  --pressure-levels "850,500,250" \
  --exp-name "${BASE_EXP_NAME}" \
  --runs-dir "${BASE_RUNS_DIR}" \
  --embed-dim "${EMBED_DIM:-256}" \
  --num-heads "${NUM_HEADS:-6}" \
  --window-size "${WINDOW_SIZE:-8}" \
  --depth-pre "${DEPTH_PRE:-1}" \
  --depth-mid "${DEPTH_MID:-6}" \
  --depth-post "${DEPTH_POST:-1}" \
  --drop-path-rate "${DROP_PATH:-0.05}" \
  --use-checkpoint \
  --batch-size "${BATCH_SIZE:-1}" \
  --accum-steps "${ACCUM_STEPS:-1}" \
  --max-epochs "${PRETRAIN_MAX_EPOCHS:-3}" \
  --max-iters "${PRETRAIN_MAX_ITERS:-150}" \
  --patience "${PRETRAIN_PATIENCE:-2}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --lr "${PRETRAIN_LR:-0.00025}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --beta1 "${BETA1:-0.9}" \
  --beta2 "${BETA2:-0.95}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --loss "${LOSS:-l1}" \
  --device "${DEVICE:-auto}" \
  --amp "${AMP:-bf16}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: Tiny pretrain did not produce base checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi

export ZARR_STORE
export BASE_CKPT
export CONFIG_SHORT
export CONFIG_MEDIUM
export CONFIG_LONG
export OUTPUT_ROOT
export CKPT_NAME
export EXP_SHORT
export EXP_MEDIUM
export EXP_LONG

bash scripts/finetune_cascade.sh

echo "Done. Tiny cascade checkpoints are under ${OUTPUT_ROOT}/"
