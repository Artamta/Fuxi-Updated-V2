#!/bin/bash
#SBATCH --job-name=fuxi_train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # 1 launcher per node
#SBATCH --gres=gpu:2               # 2 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/fuxi_%j.out
#SBATCH -e logs/fuxi_%j.err

set -euo pipefail
mkdir -p logs

# -------------------------------
# PATH
# -------------------------------
ROOT_DIR="$HOME/fuxi-final/fuxi_new"
cd "$ROOT_DIR"

# -------------------------------
# ENV
# -------------------------------
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

echo "Python: $(which python)"

# -------------------------------
# MASTER NODE SETUP
# -------------------------------
HOSTNAMES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${HOSTNAMES[0]}
MASTER_PORT=29500

echo "Nodes: ${HOSTNAMES[@]}"
echo "MASTER_ADDR=$MASTER_ADDR"

# -------------------------------
# NCCL (STABLE SETTINGS)
# -------------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# IMPORTANT: let SLURM manage GPUs
export SLURM_GPU_BIND=none

# -------------------------------
# DATA
# -------------------------------
ZARR_STORE="/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

# -------------------------------
# MODEL (SAFE CONFIG FIRST)
# -------------------------------
EMBED_DIM=256
DEPTH_PRE=2
DEPTH_MID=8
DEPTH_POST=2
NUM_HEADS=2

# -------------------------------
# TRAINING
# -------------------------------
BATCH_SIZE=1
ACCUM_STEPS=1
MAX_ITERS=40000
LR=2.5e-4

EXP_NAME="fuxi_run_$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# RUN (CLEAN TORCHRUN)
# -------------------------------
srun --ntasks=1 --nodes=1 \
accelerate launch \
  --num_machines $SLURM_NNODES \
  --num_processes 8 \
  --machine_rank $SLURM_NODEID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  -m src.pretraining.pretrain \
  --zarr-store "$ZARR_STORE" \
  --train-start 1979-01-01 \
  --train-end   1979-01-05 \   
   #2019-12-31
  --val-start   2020-01-01 \
  --val-end     2020-01-02 \       
  #2020-12-31
  --test-start  2021-01-01 \
  --test-end    2021-01-02 \        
  #2023-12-31
  --exp-name "$EXP_NAME" \
  --runs-dir Models_paper/pretrain \
  --embed-dim $EMBED_DIM \
  --num-heads 8 \
  --window-size 8 \
  --depth-pre $DEPTH_PRE \
  --depth-mid $DEPTH_MID \
  --depth-post $DEPTH_POST \
  --drop-path-rate 0.2 \
  --use-checkpoint \
  --batch-size $BATCH_SIZE \
  --accum-steps $ACCUM_STEPS \
  --max-iters $MAX_ITERS \
  --patience 10 \
  --num-workers 4 \
  --lr $LR \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --device cuda \
  --amp bf16

echo "DONE ${EXP_NAME}"