#!/bin/bash
#SBATCH --job-name=fuxi_test_2
#SBATCH --partition=gpu
#SBATCH --nodes=8   
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/fuxi_test_%j.out
#SBATCH -e logs/fuxi_test_%j.err

set -euo pipefail

# -------------------------------
# SETUP
# -------------------------------
ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "${ROOT_DIR}"
mkdir -p logs

source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

# -------------------------------
# DATA
# -------------------------------
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

# -------------------------------
# MODEL (SMALL TEST CONFIG)
# -------------------------------
EMBED_DIM=256
NUM_HEADS=8
DEPTH_PRE=2
DEPTH_MID=8
DEPTH_POST=2

# -------------------------------
# TRAINING (FAST TEST)
# -------------------------------
BATCH_SIZE=2
ACCUM_STEPS=1
MAX_ITERS=10000
MAX_EPOCHS=2
LR=2.5e-4

# -------------------------------
# NCCL (SAFE FOR A30)
# -------------------------------
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib

# -------------------------------
# RUN
# -------------------------------
echo "===== TEST RUN ====="
echo "Embed: ${EMBED_DIM}, Depth: ${DEPTH_PRE}-${DEPTH_MID}-${DEPTH_POST}"
echo "Iters: ${MAX_ITERS}"

srun python -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  src/pretraining/pretrain.py \
  --zarr-store "${ZARR_STORE}" \
  --train-start 1979-01-01 \
  --train-end 1979-01-05 \
  --val-start 2016-01-01 \
  --val-end 2016-01-03 \
  --test-start 2019-01-01 \
  --test-end 2019-01-03 \
  --exp-name "test_small_model" \
  --runs-dir Models_paper/pretrain \
  --embed-dim ${EMBED_DIM} \
  --num-heads ${NUM_HEADS} \
  --window-size 8 \
  --depth-pre ${DEPTH_PRE} \
  --depth-mid ${DEPTH_MID} \
  --depth-post ${DEPTH_POST} \
  --batch-size ${BATCH_SIZE} \
  --accum-steps ${ACCUM_STEPS} \
  --max-epochs ${MAX_EPOCHS} \
  --max-iters ${MAX_ITERS} \
  --patience 5 \
  --num-workers 2 \
  --lr ${LR} \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --grad-clip 1.0 \
  --device cuda \
  --amp bf16

echo "Done."