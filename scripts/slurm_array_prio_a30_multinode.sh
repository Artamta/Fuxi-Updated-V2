#!/bin/bash
#SBATCH --job-name=fuxi_a30_mn_arr
#SBATCH --partition=gpu_prio
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH -o logs/fuxi_a30_mn_%A_%a.out
#SBATCH -e logs/fuxi_a30_mn_%A_%a.err

set -euo pipefail

COMMON_SH=""
if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/scripts/slurm_common.sh" ]]; then
  COMMON_SH="${PROJECT_ROOT}/scripts/slurm_common.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
elif [[ -f "/home/raj.ayush/fuxi-final/fuxi_new/scripts/slurm_common.sh" ]]; then
  COMMON_SH="/home/raj.ayush/fuxi-final/fuxi_new/scripts/slurm_common.sh"
else
  echo "ERROR: Could not locate scripts/slurm_common.sh" >&2
  exit 1
fi
source "${COMMON_SH}"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  ROOT_DIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="/home/raj.ayush/fuxi-final/fuxi_new"
fi
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

slurm_activate_conda

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
slurm_require_zarr "${ZARR_STORE}"

MODES=("auto" "force_mig_safe")
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX < 0 || IDX >= ${#MODES[@]} )); then
  echo "ERROR: array index ${IDX} is out of range" >&2
  exit 1
fi
MODE=${MODES[$IDX]}

HOSTNAMES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if (( ${#HOSTNAMES[@]} < 2 )); then
  echo "ERROR: expected at least 2 nodes, got ${#HOSTNAMES[@]}" >&2
  exit 1
fi
PRIMARY_NODE=${HOSTNAMES[0]}
SECONDARY_NODE=${HOSTNAMES[1]}

MASTER_ADDR=${PRIMARY_NODE}
MASTER_PORT=$((32000 + (SLURM_JOB_ID % 1000) + IDX))
export MASTER_ADDR MASTER_PORT

RUN_ROOT="results/prio_runs/a30_multinode/${MODE}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
CASCADE_ROOT="${RUN_ROOT}/cascade"
EVAL_DIR="${RUN_ROOT}/eval_secondary"
CFG_DIR="${RUN_ROOT}/configs"
mkdir -p "${RUN_ROOT}" "${CFG_DIR}"

CFG_SHORT="${CFG_DIR}/cascade_short.yaml"
CFG_MEDIUM="${CFG_DIR}/cascade_medium.yaml"
CFG_LONG="${CFG_DIR}/cascade_long.yaml"

prepare_cfg() {
  local src=$1
  local dst=$2
  local stage_name=$3
  cp "${src}" "${dst}"
  sed -i "s|^  output_root:.*|  output_root: ${CASCADE_ROOT}|" "${dst}"
  sed -i "s|^  exp_name:.*|  exp_name: ${stage_name}|" "${dst}"
}

prepare_cfg "configs/cascade_tiny_smoke_short.yaml" "${CFG_SHORT}" "stage_short"
prepare_cfg "configs/cascade_tiny_smoke_medium.yaml" "${CFG_MEDIUM}" "stage_medium"
prepare_cfg "configs/cascade_tiny_smoke_long.yaml" "${CFG_LONG}" "stage_long"

get_mig_uuid() {
  local node=$1
  srun --nodes=1 --ntasks=1 -w "${node}" bash -lc "nvidia-smi -L | awk '/MIG/ {u=\$NF; gsub(/[()]/, \"\", u); sub(\"UUID:\", \"\", u); print u; exit}'" | tail -n 1
}

PRIMARY_MIG_UUID=$(get_mig_uuid "${PRIMARY_NODE}" || true)
SECONDARY_MIG_UUID=$(get_mig_uuid "${SECONDARY_NODE}" || true)

PRIMARY_IS_MIG=0
SECONDARY_IS_MIG=0
if [[ -n "${PRIMARY_MIG_UUID}" ]]; then
  PRIMARY_IS_MIG=1
fi
if [[ -n "${SECONDARY_MIG_UUID}" ]]; then
  SECONDARY_IS_MIG=1
fi

MIG_SAFE=0
if [[ "${MODE}" == "force_mig_safe" ]]; then
  MIG_SAFE=1
elif (( PRIMARY_IS_MIG == 1 || SECONDARY_IS_MIG == 1 )); then
  MIG_SAFE=1
fi

export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
if (( MIG_SAFE == 1 )); then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=ib
else
  export NCCL_P2P_DISABLE=0
  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME=^lo,docker0
fi

echo "============================================================"
echo "A30 priority multinode array"
echo "Mode               : ${MODE}"
echo "Nodes              : ${HOSTNAMES[*]}"
echo "Primary node       : ${PRIMARY_NODE}"
echo "Secondary node     : ${SECONDARY_NODE}"
echo "Primary MIG UUID   : ${PRIMARY_MIG_UUID:-<none>}"
echo "Secondary MIG UUID : ${SECONDARY_MIG_UUID:-<none>}"
echo "MIG safe NCCL      : ${MIG_SAFE}"
echo "Run root           : ${RUN_ROOT}"
echo "============================================================"

srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 bash -lc 'echo "--- $(hostname) ---"; nvidia-smi -L || true' | tee "${RUN_ROOT}/gpu_report.txt"

DIST_STATUS="failed"
if srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 "${PYTHON_BIN}" "${ROOT_DIR}/tests/test_multi_node.py"; then
  DIST_STATUS="passed"
fi

srun --nodes=1 --ntasks=1 -w "${PRIMARY_NODE}" env \
  PYTHON_BIN="${PYTHON_BIN}" \
  ZARR_STORE="${ZARR_STORE}" \
  BASE_RUNS_DIR="${RUN_ROOT}/pretrain" \
  BASE_EXP_NAME="tiny_base" \
  OUTPUT_ROOT="${CASCADE_ROOT}" \
  CONFIG_SHORT="${CFG_SHORT}" \
  CONFIG_MEDIUM="${CFG_MEDIUM}" \
  CONFIG_LONG="${CFG_LONG}" \
  MIG_UUID="${PRIMARY_MIG_UUID}" \
  GPU_INDEX=0 \
  bash "${ROOT_DIR}/scripts/run_cascade_tiny_smoke.sh"

SECONDARY_VISIBLE="${SECONDARY_MIG_UUID:-0}"
srun --nodes=1 --ntasks=1 -w "${SECONDARY_NODE}" env \
  CUDA_VISIBLE_DEVICES="${SECONDARY_VISIBLE}" \
  "${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
    --checkpoint "${CASCADE_ROOT}/stage_long/best.pt" \
    --rollout-steps 3 \
    --batch-size 1 \
    --num-workers 2 \
    --max-samples 8 \
    --amp fp16 \
    --device cuda \
    --output-dir "${EVAL_DIR}"

cat > "${RUN_ROOT}/summary.txt" <<EOF
mode=${MODE}
partition=${SLURM_JOB_PARTITION:-gpu_prio}
nodes=${HOSTNAMES[*]}
primary_mig_uuid=${PRIMARY_MIG_UUID:-none}
secondary_mig_uuid=${SECONDARY_MIG_UUID:-none}
mig_safe_nccl=${MIG_SAFE}
distributed_test=${DIST_STATUS}
final_checkpoint=${CASCADE_ROOT}/stage_long/best.pt
eval_summary=${EVAL_DIR}/summary.json
EOF

echo "Done. Outputs saved in ${RUN_ROOT}"
