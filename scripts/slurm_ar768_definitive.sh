#!/bin/bash
#SBATCH --job-name=fuxi_ar768_def
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/fuxi_ar768_def_%j.out
#SBATCH -e logs/fuxi_ar768_def_%j.err

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
slurm_export_nccl_defaults

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
slurm_require_zarr "${ZARR_STORE}"

BASE_CKPT="${BASE_CKPT:-/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/poster_runs/ar768_definitive_job${SLURM_JOB_ID}}"
CONFIG_SHORT="${CONFIG_SHORT:-configs/cascade_ar768_short.yaml}"
CONFIG_MEDIUM="${CONFIG_MEDIUM:-configs/cascade_ar768_medium.yaml}"
CONFIG_LONG="${CONFIG_LONG:-configs/cascade_ar768_long.yaml}"

EVAL_ROLLOUT_STEPS="${EVAL_ROLLOUT_STEPS:-60}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-256}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Definitive defaults for a full-finetune cascade run.
export AR_BATCH_SIZE="${AR_BATCH_SIZE:-12}"
export AR_NUM_WORKERS="${AR_NUM_WORKERS:-8}"
export AR_PREFETCH_FACTOR="${AR_PREFETCH_FACTOR:-3}"
export AR_EVAL_EVERY="${AR_EVAL_EVERY:-2}"
export AR_SKIP_FINAL_TEST_EVAL="${AR_SKIP_FINAL_TEST_EVAL:-1}"
export AR_MAX_EPOCHS="${AR_MAX_EPOCHS:-40}"
export AR_PATIENCE="${AR_PATIENCE:-10}"
export AR_LR="${AR_LR:-1e-4}"
export AR_INPUT_NOISE_STD="${AR_INPUT_NOISE_STD:-0.0}"
export AR_NOISE_COARSE_FACTOR="${AR_NOISE_COARSE_FACTOR:-16}"
export AR_STATUS_EVERY_STEPS="${AR_STATUS_EVERY_STEPS:-100}"
export AR_CONSTANT_LR="${AR_CONSTANT_LR:-0}"
export AR_ENABLE_LORA="${AR_ENABLE_LORA:-0}"

echo "============================================================"
echo "FuXi poster run: AR cascade 768 (definitive)"
echo "Host                : $(hostname)"
echo "Partition           : ${SLURM_JOB_PARTITION:-GPU-AI_prio}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Zarr                : ${ZARR_STORE}"
echo "Base checkpoint     : ${BASE_CKPT}"
echo "Output root         : ${OUTPUT_ROOT}"
echo "AR tuning           : batch=${AR_BATCH_SIZE}, workers=${AR_NUM_WORKERS}, prefetch=${AR_PREFETCH_FACTOR}, eval_every=${AR_EVAL_EVERY}, max_epochs=${AR_MAX_EPOCHS}, lr=${AR_LR}, constant_lr=${AR_CONSTANT_LR}, patience=${AR_PATIENCE}, skip_final_test_eval=${AR_SKIP_FINAL_TEST_EVAL}, enable_lora=${AR_ENABLE_LORA}"
echo "============================================================"

export PYTHON_BIN
export ZARR_STORE
export BASE_CKPT
export CONFIG_SHORT
export CONFIG_MEDIUM
export CONFIG_LONG
export OUTPUT_ROOT
export CKPT_NAME="best.pt"
export EXP_SHORT="stage_short"
export EXP_MEDIUM="stage_medium"
export EXP_LONG="stage_long"

bash scripts/finetune_cascade_ar.sh

FINAL_CKPT="${OUTPUT_ROOT}/stage_long/best.pt"
if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "ERROR: final checkpoint missing: ${FINAL_CKPT}" >&2
  exit 1
fi

EVAL_DIR="${OUTPUT_ROOT}/stage_long/evaluation_rollout${EVAL_ROLLOUT_STEPS}"
"${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
  --checkpoint "${FINAL_CKPT}" \
  --rollout-steps "${EVAL_ROLLOUT_STEPS}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-samples "${EVAL_MAX_SAMPLES}" \
  --amp bf16 \
  --device cuda \
  --output-dir "${EVAL_DIR}"

cat > "${OUTPUT_ROOT}/summary.txt" <<EOF
partition=${SLURM_JOB_PARTITION:-GPU-AI_prio}
host=$(hostname)
base_checkpoint=${BASE_CKPT}
final_checkpoint=${FINAL_CKPT}
rollout_steps=${EVAL_ROLLOUT_STEPS}
eval_summary=${EVAL_DIR}/summary.json
EOF

echo "Done. Definitive poster outputs: ${OUTPUT_ROOT}"