#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/fuxi_pre_smoke_%j.out
#SBATCH -e logs/fuxi_pre_smoke_%j.err

set -euo pipefail

# -------------------------------
# PATH SETUP
# -------------------------------
ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "${ROOT_DIR}"
mkdir -p logs

# -------------------------------
# ACTIVATE ENV (FORCE CORRECT ONE)
# -------------------------------
# Try module first
# -------------------------------
# ACTIVATE ENV (CORRECT WAY)
# -------------------------------

source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh

conda activate weather_forecast

echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"
python -c "import torch; print('Torch OK:', torch.__version__)"

# -------------------------------
# DATA PATH
# -------------------------------
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

if [ ! -d "$ZARR_STORE" ]; then
    echo "ERROR: Zarr store not found!"
    exit 1
fi

# -------------------------------
# GPU + NCCL DEBUG
# -------------------------------
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# -------------------------------
# INFO
# -------------------------------
echo "============================================================"
echo "FuXi pretraining smoke test (2 GPUs)"
echo "Host        : $(hostname)"
echo "Working Dir : $(pwd)"
echo "Experiment  : smoke_$(date +%Y%m%d_%H%M%S)"
echo "Zarr        : ${ZARR_STORE}"
echo "============================================================"

nvidia-smi

# -------------------------------
# RUN
# -------------------------------
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port=29501 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 1979-01-10 \
  --val-start 2016-01-01 \
  --val-end 2016-01-03 \
  --test-start 2018-01-01 \
  --test-end 2018-01-03 \
  --exp-name "smoke_run" \
  --runs-dir Models_paper/pretrain \
  --embed-dim 256 \
  --num-heads 8 \
  --window-size 8 \
  --depth-pre 2 \
  --depth-mid 8 \
  --depth-post 2 \
  --batch-size 1 \
  --accum-steps 1 \
  --max-epochs 4 \
  --max-iters 100 \
  --patience 2 \
  --num-workers 10 \
  --device cuda \
  --amp bf16

echo "Done. Outputs: Models_paper/pretrain/"