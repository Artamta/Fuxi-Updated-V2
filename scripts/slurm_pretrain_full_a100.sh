#!/bin/bash
#SBATCH --job-name=fuxi_pre_full
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/fuxi_pre_full_%j.out
#SBATCH -e logs/fuxi_pre_full_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"

ROOT_DIR=$(slurm_root)
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

NUM_WORKERS=8
ACCUM_STEPS=8
NUM_HEADS=12
BATCH_SIZE=2  

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

slurm_activate_conda

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr}"
slurm_require_zarr "${ZARR_STORE}"

EMBED_DIM="${EMBED_DIM:-1536}"
WINDOW="${WINDOW:-8}"
DEPTH_PRE="${DEPTH_PRE:-16}"
DEPTH_MID="${DEPTH_MID:-16}"
DEPTH_POST="${DEPTH_POST:-16}"

ACCUM_STEPS="${ACCUM_STEPS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MAX_ITERS="${MAX_ITERS:-100000}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
DROP_PATH="${DROP_PATH:-0.2}"

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

NGPUS=$(slurm_detect_gpus)
MASTER_PORT=$(slurm_pick_port)

slurm_export_nccl_defaults

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
  --train-end 2019-12-31 \
  --val-start 2020-01-01 \
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
  --grad-clip 1.0 \
  --amp bf16 \
  ${RESUME_FLAG}

echo "Done. Outputs: ${RUNS_DIR}/${EXP_NAME}"
