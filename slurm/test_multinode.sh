#!/bin/bash
#SBATCH --job-name=mn_test
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/mn_%j.out
#SBATCH -e logs/mn_%j.err

set -euo pipefail

source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

# MASTER NODE
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

# MIG SAFE SETTINGS (VERY IMPORTANT)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=ib

echo "MASTER_ADDR=$MASTER_ADDR"

srun python /home/raj.ayush/fuxi-final/fuxi_new/tests/test_multi_node.py