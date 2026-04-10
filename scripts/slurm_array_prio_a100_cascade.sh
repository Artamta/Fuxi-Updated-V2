#!/bin/bash
#SBATCH --job-name=fuxi_a100_arr
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH -o logs/fuxi_a100_arr_%A_%a.out
#SBATCH -e logs/fuxi_a100_arr_%A_%a.err

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

PROFILES=("a100_2gpu" "a100_4gpu")
GPU_COUNTS=(2 4)
EMBED_DIMS=(256 384)
NUM_HEADS=(4 6)
DEPTH_MID=(6 8)
PRETRAIN_MAX_ITERS=(250 400)
CASCADE_MAX_ITERS=(120 180)
CASCADE_MAX_EPOCHS=(3 4)
EVAL_MAX_SAMPLES=(16 24)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX < 0 || IDX >= ${#PROFILES[@]} )); then
  echo "ERROR: array index ${IDX} is out of range" >&2
  exit 1
fi

PROFILE=${PROFILES[$IDX]}
GPU_COUNT=${GPU_COUNTS[$IDX]}
EMBED_DIM=${EMBED_DIMS[$IDX]}
HEADS=${NUM_HEADS[$IDX]}
MID_DEPTH=${DEPTH_MID[$IDX]}
PRETRAIN_ITERS=${PRETRAIN_MAX_ITERS[$IDX]}
CASCADE_ITERS=${CASCADE_MAX_ITERS[$IDX]}
CASCADE_EPOCHS=${CASCADE_MAX_EPOCHS[$IDX]}
MAX_SAMPLES=${EVAL_MAX_SAMPLES[$IDX]}

GPU_LIST=$(seq 0 $((GPU_COUNT - 1)) | paste -sd, -)
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

MIG_MODE="full"
if nvidia-smi -L | grep -q "MIG"; then
  MIG_MODE="mig"
fi

RUN_ROOT="results/prio_runs/a100/${PROFILE}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
PRETRAIN_ROOT="${RUN_ROOT}/pretrain"
BASE_EXP_NAME="tiny_base"
BASE_CKPT="${PRETRAIN_ROOT}/${BASE_EXP_NAME}/best.pt"
CASCADE_ROOT="${RUN_ROOT}/cascade"
EVAL_DIR="${RUN_ROOT}/eval_stage_long"
CFG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CFG_DIR}" "${RUN_ROOT}"

CFG_SHORT="${CFG_DIR}/cascade_short.yaml"
CFG_MEDIUM="${CFG_DIR}/cascade_medium.yaml"
CFG_LONG="${CFG_DIR}/cascade_long.yaml"

make_cfg() {
  local src=$1
  local dst=$2
  local stage_name=$3
  local stage_lr=$4
  local stage_train_start=$5
  local stage_train_end=$6

  cp "${src}" "${dst}"
  sed -i "s|^  train_start:.*|  train_start: ${stage_train_start}|" "${dst}"
  sed -i "s|^  train_end:.*|  train_end: ${stage_train_end}|" "${dst}"
  sed -i "s|^  val_start:.*|  val_start: 2016-01-01|" "${dst}"
  sed -i "s|^  val_end:.*|  val_end: 2016-01-10|" "${dst}"
  sed -i "s|^  test_start:.*|  test_start: 2018-01-01|" "${dst}"
  sed -i "s|^  test_end:.*|  test_end: 2018-01-10|" "${dst}"

  sed -i "s|^  embed_dim:.*|  embed_dim: ${EMBED_DIM}|" "${dst}"
  sed -i "s|^  num_heads:.*|  num_heads: ${HEADS}|" "${dst}"
  sed -i "s|^  depth_mid:.*|  depth_mid: ${MID_DEPTH}|" "${dst}"

  sed -i "s|^  max_epochs:.*|  max_epochs: ${CASCADE_EPOCHS}|" "${dst}"
  sed -i "s|^  max_iters:.*|  max_iters: ${CASCADE_ITERS}|" "${dst}"
  sed -i "s|^  lr:.*|  lr: ${stage_lr}|" "${dst}"

  sed -i "s|^  output_root:.*|  output_root: ${CASCADE_ROOT}|" "${dst}"
  sed -i "s|^  exp_name:.*|  exp_name: ${stage_name}|" "${dst}"
}

make_cfg "configs/cascade_tiny_short.yaml" "${CFG_SHORT}" "stage_short" "0.00025" "1979-01-01" "1979-02-15"
make_cfg "configs/cascade_tiny_medium.yaml" "${CFG_MEDIUM}" "stage_medium" "0.00010" "1979-02-16" "1979-03-31"
make_cfg "configs/cascade_tiny_long.yaml" "${CFG_LONG}" "stage_long" "0.00005" "1979-04-01" "1979-05-15"

MAIN_PORT=$((30000 + (SLURM_JOB_ID % 1000) + (10 * IDX)))

echo "============================================================"
echo "A100 priority array profile"
echo "Profile            : ${PROFILE}"
echo "Host               : $(hostname)"
echo "Partition          : ${SLURM_JOB_PARTITION:-GPU-AI_prio}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MIG mode detected  : ${MIG_MODE}"
echo "Run root           : ${RUN_ROOT}"
echo "============================================================"

if command -v accelerate >/dev/null 2>&1; then
  accelerate launch \
    --multi_gpu \
    --num_processes="${GPU_COUNT}" \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port="${MAIN_PORT}" \
    --mixed_precision=bf16 \
    --module src.pretraining.pretrain \
    --zarr-store "${ZARR_STORE}" \
    --train-start "1979-01-01" \
    --train-end "1979-05-15" \
    --val-start "2016-01-01" \
    --val-end "2016-01-10" \
    --test-start "2018-01-01" \
    --test-end "2018-01-10" \
    --pressure-vars "temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind" \
    --surface-vars "2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_column_water_vapour" \
    --pressure-levels "850,500,250" \
    --exp-name "${BASE_EXP_NAME}" \
    --runs-dir "${PRETRAIN_ROOT}" \
    --embed-dim "${EMBED_DIM}" \
    --num-heads "${HEADS}" \
    --window-size 8 \
    --depth-pre 1 \
    --depth-mid "${MID_DEPTH}" \
    --depth-post 1 \
    --drop-path-rate 0.05 \
    --use-checkpoint \
    --batch-size 1 \
    --accum-steps 1 \
    --max-epochs 3 \
    --max-iters "${PRETRAIN_ITERS}" \
    --patience 2 \
    --num-workers 8 \
    --lr 0.00025 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad-clip 1.0 \
    --loss l1 \
    --device cuda \
    --amp bf16
else
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes="${GPU_COUNT}" \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port="${MAIN_PORT}" \
    --mixed_precision=bf16 \
    --module src.pretraining.pretrain \
    --zarr-store "${ZARR_STORE}" \
    --train-start "1979-01-01" \
    --train-end "1979-05-15" \
    --val-start "2016-01-01" \
    --val-end "2016-01-10" \
    --test-start "2018-01-01" \
    --test-end "2018-01-10" \
    --pressure-vars "temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind" \
    --surface-vars "2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_column_water_vapour" \
    --pressure-levels "850,500,250" \
    --exp-name "${BASE_EXP_NAME}" \
    --runs-dir "${PRETRAIN_ROOT}" \
    --embed-dim "${EMBED_DIM}" \
    --num-heads "${HEADS}" \
    --window-size 8 \
    --depth-pre 1 \
    --depth-mid "${MID_DEPTH}" \
    --depth-post 1 \
    --drop-path-rate 0.05 \
    --use-checkpoint \
    --batch-size 1 \
    --accum-steps 1 \
    --max-epochs 3 \
    --max-iters "${PRETRAIN_ITERS}" \
    --patience 2 \
    --num-workers 8 \
    --lr 0.00025 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad-clip 1.0 \
    --loss l1 \
    --device cuda \
    --amp bf16
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: base checkpoint missing: ${BASE_CKPT}" >&2
  exit 1
fi

export PYTHON_BIN
export ZARR_STORE
export BASE_CKPT
export CONFIG_SHORT="${CFG_SHORT}"
export CONFIG_MEDIUM="${CFG_MEDIUM}"
export CONFIG_LONG="${CFG_LONG}"
export OUTPUT_ROOT="${CASCADE_ROOT}"
export CKPT_NAME="best.pt"
export EXP_SHORT="stage_short"
export EXP_MEDIUM="stage_medium"
export EXP_LONG="stage_long"

bash scripts/finetune_cascade.sh

"${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
  --checkpoint "${CASCADE_ROOT}/stage_long/best.pt" \
  --rollout-steps 20 \
  --batch-size 1 \
  --num-workers 4 \
  --max-samples "${MAX_SAMPLES}" \
  --amp bf16 \
  --device cuda \
  --output-dir "${EVAL_DIR}"

cat > "${RUN_ROOT}/summary.txt" <<EOF
profile=${PROFILE}
partition=${SLURM_JOB_PARTITION:-GPU-AI_prio}
host=$(hostname)
mig_mode_detected=${MIG_MODE}
gpus_requested=${GPU_COUNT}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
base_checkpoint=${BASE_CKPT}
final_checkpoint=${CASCADE_ROOT}/stage_long/best.pt
eval_summary=${EVAL_DIR}/summary.json
EOF

echo "Done. Outputs saved in ${RUN_ROOT}"
