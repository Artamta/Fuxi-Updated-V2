#!/bin/bash
#SBATCH --job-name=2048
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-24:00:00
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# -------------------------------
# INFO
# -------------------------------
echo "============================================================"
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
  --num_processes=6 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port=29501 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 2019-12-31 \
  --val-start 2020-01-01 \
  --val-end 2020-12-31 \
  --test-start 2021-01-01 \
  --test-end 2022-12-31 \
  --exp-name "scaling-test" \
  --runs-dir Models_paper/pretrain \
  --embed-dim 2048 \
  --num-heads 32 \
  --window-size 8 \
  --depth-pre 8 \
  --depth-mid 32 \
  --depth-post 8 \
  --batch-size 1 \
  --accum-steps 1 \
  --max-epochs 50 \
  --max-iters 40000 \
  --patience 5 \
  --num-workers 10 \
  --grad-clip 1.0 \
  --device cuda \
  --amp bf16

echo "Done. Outputs: Models_paper/pretrain/"