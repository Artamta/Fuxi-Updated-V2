#!/bin/bash
#SBATCH --job-name=fuxi_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o logs/fuxi_pre_smoke_%j.out
#SBATCH -e logs/fuxi_pre_smoke_%j.err

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

NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [ -z "${NGPUS}" ]; then
  NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
if [ "${NGPUS}" -lt 1 ]; then
  echo "ERROR: no GPUs visible on this node."
  exit 1
fi

MASTER_PORT=$((10000 + (RANDOM % 50000)))
EXP_NAME="smoke_2gpu_$(date +%Y%m%d_%H%M%S)"

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

echo "============================================================"
echo "FuXi pretraining smoke test (2 GPUs)"
echo "Host        : $(hostname)"
echo "GPUs        : ${NGPUS}"
echo "Experiment  : ${EXP_NAME}"
echo "Zarr        : ${ZARR_STORE}"
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
  --train-end 1979-01-10 \
  --val-start 2016-01-01 \
  --val-end 2016-01-03 \
  --test-start 2018-01-01 \
  --test-end 2018-01-03 \
  --exp-name "${EXP_NAME}" \
  --runs-dir Models_paper/pretrain \
  --embed-dim 256 \
  --num-heads 8 \
  --window-size 8 \
  --depth-pre 2 \
  --depth-mid 8 \
  --depth-post 2 \
  --batch-size 1 \
  --accum-steps 1 \
  --max-epochs 1 \
  --max-iters 100 \
  --patience 2 \
  --num-workers 2 \
  --device cuda \
  --amp bf16

echo "Done. Outputs: Models_paper/pretrain/${EXP_NAME}"
