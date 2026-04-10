#!/bin/bash
#SBATCH --job-name=fuxi_pack_results
#SBATCH --partition=gpu_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o logs/fuxi_pack_results_%j.out
#SBATCH -e logs/fuxi_pack_results_%j.err

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  ROOT_DIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="/home/raj.ayush/fuxi-final/fuxi_new"
fi

cd "${ROOT_DIR}"
mkdir -p logs

RUN_ROOT=${RUN_ROOT:-}
if [[ -z "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT must be set" >&2
  exit 1
fi

PACKAGE_OUT=${PACKAGE_OUT:-"results/final_reports/$(basename "${RUN_ROOT}")_pack"}

bash scripts/package_cascade_results.sh "${RUN_ROOT}" "${PACKAGE_OUT}"

echo "Pack complete: ${PACKAGE_OUT}"
