#!/bin/bash
#SBATCH --job-name=fuxi_all
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/fuxi_%A_%a.out
#SBATCH -e logs/fuxi_%A_%a.err

set -euo pipefail
mkdir -p logs

# -------------------------------
# CONFIG GRID (SCALING STUDY)
# -------------------------------
EMBED_DIMS=(256 768)
DEPTHS=(8 16)

IDX=$SLURM_ARRAY_TASK_ID
EMBED_DIM=${EMBED_DIMS[$IDX]}
DEPTH_MID=${DEPTHS[$IDX]}

# Auto heads (important)
NUM_HEADS=$((EMBED_DIM / 64))

echo "Running config: dim=${EMBED_DIM}, depth_mid=${DEPTH_MID}, heads=${NUM_HEADS}"

# -------------------------------
# PATH
# -------------------------------
ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "${ROOT_DIR}"

# -------------------------------
# ENV
# -------------------------------
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

# -------------------------------
# NCCL SAFE SETTINGS
# -------------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

# -------------------------------
# MASTER NODE
# -------------------------------
HOSTNAMES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${HOSTNAMES[0]}
MASTER_PORT=$((29500 + IDX))

echo "Nodes: ${HOSTNAMES[@]}"
echo "MASTER_ADDR=$MASTER_ADDR PORT=$MASTER_PORT"

# -------------------------------
# DATA
# -------------------------------
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

# -------------------------------
# BATCH (EFFECTIVE = 8)
# -------------------------------
# GPUs = 4 → 1 × 4 × 2 = 8
BATCH_SIZE=1
ACCUM_STEPS=2

# -------------------------------
# TRAINING SETTINGS
# -------------------------------
MAX_ITERS=40000
LR=2.5e-4

EXP_NAME="fuxi_dim${EMBED_DIM}_$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# OPTIONAL FINETUNE (ONLY FOR 1536)
# -------------------------------
RESUME_FLAG=""

if [ "${EMBED_DIM}" -eq 1536 ]; then
    echo ">>> FULL MODEL DETECTED → enabling finetuning mode"

    # Put your pretrained checkpoint here
    PRETRAIN_CKPT="Models_paper/pretrain/best_1536.pt"

    if [ -f "$PRETRAIN_CKPT" ]; then
        RESUME_FLAG="--resume ${PRETRAIN_CKPT}"
    fi

    MAX_ITERS=20000   # shorter finetune
    LR=1e-4           # smaller LR
fi

# -------------------------------
# RUN (TORCHRUN MULTI-NODE)
# -------------------------------
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.pretraining.pretrain \
  \
  --zarr-store "${ZARR_STORE}" \
  \
  --train-start 1979-01-01 \
  --train-end   2019-12-31 \
  --val-start   2020-01-01 \
  --val-end     2020-12-31 \
  --test-start  2021-01-01 \
  --test-end    2023-12-31 \
  \
  --exp-name "${EXP_NAME}" \
  --runs-dir Models_paper/pretrain \
  \
  --embed-dim ${EMBED_DIM} \
  --num-heads ${NUM_HEADS} \
  --window-size 8 \
  --depth-pre 2 \
  --depth-mid ${DEPTH_MID} \
  --depth-post 2 \
  --drop-path-rate 0.2 \
  --use-checkpoint \
  \
  --batch-size ${BATCH_SIZE} \
  --accum-steps ${ACCUM_STEPS} \
  \
  --max-iters ${MAX_ITERS} \
  --patience 10 \
  \
  --num-workers 8 \
  \
  --lr ${LR} \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  \
  --device cuda \
  --amp bf16 \
  ${RESUME_FLAG}

echo "DONE ${EXP_NAME}"