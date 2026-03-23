#!/bin/bash
#SBATCH --job-name=fuxi_pre_full
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=cn3,cn15
#SBATCH -o logs/fuxi_pre_full_%j.out
#SBATCH -e logs/fuxi_pre_full_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr}"

if [ ! -d "${ZARR_STORE}" ]; then
  echo "ERROR: zarr store not found: ${ZARR_STORE}"
  exit 1
fi

EMBED_DIM="${EMBED_DIM:-1536}"
NUM_HEADS="${NUM_HEADS:-8}"
WINDOW="${WINDOW:-8}"
DEPTH_PRE="${DEPTH_PRE:-2}"
DEPTH_MID="${DEPTH_MID:-44}"
DEPTH_POST="${DEPTH_POST:-2}"

BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MAX_ITERS="${MAX_ITERS:-40000}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
DROP_PATH="${DROP_PATH:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-2}"

RUNS_DIR="${RUNS_DIR:-Models_paper/pretrain}"
EXP_NAME="${EXP_NAME:-emb_${EMBED_DIM}}"
LAST_CKPT="${RUNS_DIR}/${EXP_NAME}/last.pt"
RESUME_FLAG=""
if [ -f "${LAST_CKPT}" ]; then
  RESUME_FLAG="--resume ${LAST_CKPT}"
fi
if [ -n "${RESUME_CKPT:-}" ]; then
  RESUME_FLAG="--resume ${RESUME_CKPT}"
fi

NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [ -z "${NGPUS}" ]; then
  NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
if [ "${NGPUS}" -lt 1 ]; then
  echo "ERROR: no GPUs visible on this node."
  exit 1
fi

MASTER_PORT=$((10000 + (RANDOM % 50000)))

export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONFAULTHANDLER=1

echo "============================================================"
echo "FuXi pretraining full run (A100)"
echo "Host        : $(hostname)"
echo "GPUs        : ${NGPUS}"
echo "Experiment  : ${EXP_NAME}"
echo "Zarr        : ${ZARR_STORE}"
echo "Model       : embed=${EMBED_DIM}, depths=${DEPTH_PRE}/${DEPTH_MID}/${DEPTH_POST}"
echo "Training    : batch=${BATCH_SIZE}, accum=${ACCUM_STEPS}, max_iters=${MAX_ITERS}, amp=bf16"
echo "Resume flag : ${RESUME_FLAG:-<none>}"
echo "============================================================"
nvidia-smi

accelerate launch \
  --multi_gpu \
  --num_processes=${NGPUS} \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port=${MASTER_PORT} \
  --dynamo_backend=no \
  --mixed_precision=bf16 \
  --module src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 2018-12-31 \
  --val-start 2019-01-01 \
  --val-end 2020-12-31 \
  --test-start 2021-01-01 \
  --test-end 2022-12-31 \
  --exp-name "${EXP_NAME}" \
  --runs-dir "${RUNS_DIR}" \
  --embed-dim ${EMBED_DIM} \
  --num-heads ${NUM_HEADS} \
  --window-size ${WINDOW} \
  --depth-pre ${DEPTH_PRE} \
  --depth-mid ${DEPTH_MID} \
  --depth-post ${DEPTH_POST} \
  --drop-path-rate ${DROP_PATH} \
  --use-checkpoint \
  --batch-size ${BATCH_SIZE} \
  --accum-steps ${ACCUM_STEPS} \
  --max-epochs ${MAX_EPOCHS} \
  --max-iters ${MAX_ITERS} \
  --patience ${PATIENCE} \
  --num-workers ${NUM_WORKERS} \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --beta1 ${BETA1} \
  --beta2 ${BETA2} \
  --device cuda \
  --amp bf16 \
  ${RESUME_FLAG}

echo "Done. Outputs: ${RUNS_DIR}/${EXP_NAME}"
