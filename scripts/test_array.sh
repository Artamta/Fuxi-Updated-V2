#!/bin/bash
#SBATCH --job-name=fuxi_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=7-24:00:00
#SBATCH --array=0-3
#SBATCH -o logs/fuxi_%A_%a.out
#SBATCH -e logs/fuxi_%A_%a.err

set -euo pipefail

# -------------------------------
# CONFIG GRID
# -------------------------------
EMBED_DIMS=(256 512 768 1024)
DEPTH_MID=(8 12 16 20)

IDX=$SLURM_ARRAY_TASK_ID

EMBED_DIM=${EMBED_DIMS[$IDX]}
DEPTH=${DEPTH_MID[$IDX]}

echo "Running job $IDX → embed_dim=$EMBED_DIM depth_mid=$DEPTH"

# -------------------------------
# PATH
# -------------------------------
ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "${ROOT_DIR}"
mkdir -p logs

# -------------------------------
# ENV
# -------------------------------
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

# -------------------------------
# DATA
# -------------------------------
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

# -------------------------------
# GPU
# -------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

nvidia-smi

# -------------------------------
# RUN
# -------------------------------
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --main_process_port=2950$IDX \
  --mixed_precision=bf16 \
  -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  \
  --train-start 1979-01-01 \
  --train-end   2015-12-31 \
  --val-start   2016-01-01 \
  --val-end     2018-12-31 \
  --test-start  2019-01-01 \
  --test-end    2022-12-31 \
  \
  --exp-name "exp_${IDX}_dim${EMBED_DIM}_depth${DEPTH}" \
  --runs-dir Models_paper/pretrain \
  \
  --embed-dim ${EMBED_DIM} \
  --depth-mid ${DEPTH} \
  --depth-pre 2 \
  --depth-post 2 \
  \
  --num-heads 8 \
  --window-size 8 \
  \
  --batch-size 8 \
  --accum-steps 1 \
  --max-epochs 10 \
  --patience 5 \
  \
  --num-workers 42 \
  --device cuda \
  --amp bf16

echo "Done job $IDX"