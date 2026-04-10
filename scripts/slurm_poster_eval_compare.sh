#!/bin/bash
#SBATCH --job-name=fuxi_eval_poster
#SBATCH --partition=gpu_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o logs/fuxi_eval_poster_%j.out
#SBATCH -e logs/fuxi_eval_poster_%j.err

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
cd "${ROOT_DIR}"
mkdir -p logs

slurm_activate_conda

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_ROOT=${RUN_ROOT:-}
if [[ -z "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT must be provided (example: results/poster_runs/ar768_cascade_job7707)" >&2
  exit 1
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"${RUN_ROOT}/poster_eval_compare"}

# Eval tuning knobs (override via --export)
export ROLL_STEPS=${ROLL_STEPS:-60}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_WORKERS=${NUM_WORKERS:-4}
export MAX_SAMPLES=${MAX_SAMPLES:-256}
export SAVE_N_SAMPLES=${SAVE_N_SAMPLES:-24}
export AMP_MODE=${AMP_MODE:-fp16}
export DEVICE_MODE=${DEVICE_MODE:-cuda}
export PYTHON_BIN

echo "============================================================"
echo "FuXi poster evaluation compare"
echo "Host         : $(hostname)"
echo "Run root     : ${RUN_ROOT}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "============================================================"

bash scripts/run_poster_eval_compare.sh "${RUN_ROOT}" "${OUTPUT_ROOT}"

echo "Done. Poster comparison outputs in ${OUTPUT_ROOT}"
