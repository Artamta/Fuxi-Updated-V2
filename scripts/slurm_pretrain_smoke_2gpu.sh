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

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_common.sh"

ROOT_DIR=$(slurm_root)
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

slurm_activate_conda

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr}"
slurm_require_zarr "${ZARR_STORE}"

NGPUS=$(slurm_detect_gpus)
MASTER_PORT=$(slurm_pick_port)
EXP_NAME="${EXP_NAME:-smoke_2gpu_$(date +%Y%m%d_%H%M%S)}"

slurm_export_nccl_defaults

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
