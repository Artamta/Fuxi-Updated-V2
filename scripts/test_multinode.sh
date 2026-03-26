#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

set -euo pipefail
mkdir -p logs

# -------------------------------
# ENV
# -------------------------------
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

# -------------------------------
# NODE INFO
# -------------------------------
HOSTNAMES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${HOSTNAMES[0]}
MASTER_PORT=29500

echo "Nodes: ${HOSTNAMES[@]}"
echo "MASTER_ADDR=$MASTER_ADDR"

# -------------------------------
# NCCL SAFE SETTINGS
# -------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


# -------------------------------
# LAUNCH ON EACH NODE MANUALLY
# -------------------------------
for i in "${!HOSTNAMES[@]}"; do
    NODE=${HOSTNAMES[$i]}
    echo "Starting node $NODE rank $i"

    srun --nodes=1 --ntasks=1 -w $NODE \
        bash -c "
        cd $HOME/fuxi-final/fuxi_new &&
        torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=2 \
        --node_rank=$i \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        scripts/test_ddp.py
        " &
done

wait