#!/bin/bash
#SBATCH --job-name=embed_scale
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-4%2
#SBATCH -o logs/embed_scale_%A_%a.out
#SBATCH -e logs/embed_scale_%A_%a.err

set -euo pipefail

ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "$ROOT_DIR"
mkdir -p logs

# Environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO

echo "Python: $(which python)"
echo "Env   : $CONDA_DEFAULT_ENV"
python -c "import torch; print('Torch OK:', torch.__version__)"

# Dataset
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

if [ ! -d "$ZARR_STORE" ]; then
    echo "ERROR: Zarr store not found: $ZARR_STORE"
    exit 1
fi

# Scaling configs
EMBEDS=(512 768 1024 1536 2048)
HEADS=(8 12 16 24 32)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#EMBEDS[@]}" ]; then
    echo "ERROR: invalid array index $TASK_ID"
    exit 1
fi

EMBED_DIM="${EMBEDS[$TASK_ID]}"
NUM_HEADS="${HEADS[$TASK_ID]}"

# Fixed architecture for fair comparison
WINDOW_SIZE=8
DEPTH_PRE=8
DEPTH_MID=12
DEPTH_POST=8

# Fixed training settings
BATCH_SIZE=1
ACCUM_STEPS=1
MAX_EPOCHS=50
MAX_ITERS=40000
PATIENCE=5
NUM_WORKERS=8
LR=2.5e-4
WEIGHT_DECAY=0.1
BETA1=0.9
BETA2=0.95
GRAD_CLIP=1.0
DROP_PATH=0.2
MIXED_PRECISION=bf16

EXP_NAME="embed_${EMBED_DIM}_8-32-8"
MASTER_PORT=$((29500 + TASK_ID))

echo "============================================================"
echo "Embedding scaling run"
echo "Task ID     : $TASK_ID"
echo "Experiment  : $EXP_NAME"
echo "Embed dim   : $EMBED_DIM"
echo "Num heads   : $NUM_HEADS"
echo "Depth       : ${DEPTH_PRE}/${DEPTH_MID}/${DEPTH_POST}"
echo "GPUs/job    : 4"
echo "Port        : $MASTER_PORT"
echo "Zarr        : $ZARR_STORE"
echo "============================================================"

nvidia-smi

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port="${MASTER_PORT}" \
  --dynamo_backend=no \
  --mixed_precision="${MIXED_PRECISION}" \
  -m src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 2019-12-31 \
  --val-start 2020-01-01 \
  --val-end 2020-12-31 \
  --test-start 2021-01-01 \
  --test-end 2022-12-31 \
  --exp-name "${EXP_NAME}" \
  --runs-dir Models_paper/pretrain \
  --embed-dim "${EMBED_DIM}" \
  --num-heads "${NUM_HEADS}" \
  --window-size "${WINDOW_SIZE}" \
  --depth-pre "${DEPTH_PRE}" \
  --depth-mid "${DEPTH_MID}" \
  --depth-post "${DEPTH_POST}" \
  --mlp-ratio 4.0 \
  --drop-path-rate "${DROP_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --accum-steps "${ACCUM_STEPS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --max-iters "${MAX_ITERS}" \
  --patience "${PATIENCE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --grad-clip "${GRAD_CLIP}" \
  --device cuda \
  --amp bf16

echo "Done. Outputs: Models_paper/pretrain/${EXP_NAME}"