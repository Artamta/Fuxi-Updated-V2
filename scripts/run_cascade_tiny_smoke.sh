#!/usr/bin/env bash
set -euo pipefail

# Tiny end-to-end smoke on a single MIG slice.
# Usage (from login node):
#   ssh cn3 'cd /home/raj.ayush/fuxi-final/fuxi_new && bash scripts/run_cascade_tiny_smoke.sh'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
MIG_UUID="${MIG_UUID:-}"
GPU_INDEX="${GPU_INDEX:-0}"

BASE_RUNS_DIR="${BASE_RUNS_DIR:-results/pretrain_tiny_smoke}"
BASE_EXP_NAME="${BASE_EXP_NAME:-tiny_base_smoke}"
BASE_CKPT="${BASE_RUNS_DIR}/${BASE_EXP_NAME}/best.pt"

if [[ -n "${MIG_UUID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MIG_UUID}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
fi

echo "============================================================"
echo "Tiny cascade smoke"
echo "Host                : $(hostname)"
echo "Python              : ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Zarr                : ${ZARR_STORE}"
echo "============================================================"

"${PYTHON_BIN}" - <<'PYCODE'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("torch gpu0:", torch.cuda.get_device_name(0))
PYCODE

"${PYTHON_BIN}" -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 1979-01-03 \
  --val-start 2016-01-01 \
  --val-end 2016-01-02 \
  --test-start 2018-01-01 \
  --test-end 2018-01-02 \
  --pressure-vars "temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind" \
  --surface-vars "2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_column_water_vapour" \
  --pressure-levels "850,500,250" \
  --exp-name "${BASE_EXP_NAME}" \
  --runs-dir "${BASE_RUNS_DIR}" \
  --embed-dim 128 \
  --num-heads 4 \
  --window-size 8 \
  --depth-pre 1 \
  --depth-mid 4 \
  --depth-post 1 \
  --drop-path-rate 0.05 \
  --use-checkpoint \
  --batch-size 1 \
  --accum-steps 1 \
  --max-epochs 1 \
  --max-iters 30 \
  --patience 1 \
  --num-workers 2 \
  --lr 0.00025 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --grad-clip 1.0 \
  --loss l1 \
  --device auto \
  --amp fp16

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: base checkpoint not created: ${BASE_CKPT}" >&2
  exit 1
fi

export BASE_CKPT
export PYTHON_BIN
export CONFIG_SHORT="${CONFIG_SHORT:-configs/cascade_tiny_smoke_short.yaml}"
export CONFIG_MEDIUM="${CONFIG_MEDIUM:-configs/cascade_tiny_smoke_medium.yaml}"
export CONFIG_LONG="${CONFIG_LONG:-configs/cascade_tiny_smoke_long.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/cascade_tiny_smoke}"
export CKPT_NAME="${CKPT_NAME:-best.pt}"
export EXP_SHORT="${EXP_SHORT:-stage_short}"
export EXP_MEDIUM="${EXP_MEDIUM:-stage_medium}"
export EXP_LONG="${EXP_LONG:-stage_long}"

bash scripts/finetune_cascade.sh

echo "Done. Smoke outputs in ${OUTPUT_ROOT}/"
