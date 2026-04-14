#!/usr/bin/env bash
set -euo pipefail

# One-command poster pipeline launcher for AR-768.
#
# Steps:
#   1) submit the short->medium->long training job
#   2) submit the dependent poster rollout/compare job
#   3) optionally submit the packaging job
#
# All runtime overrides are passed through from the current environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/slurm_ar768_definitive.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/slurm_poster_eval_compare.sh}"
PACKAGE_SCRIPT="${PACKAGE_SCRIPT:-scripts/slurm_package_cascade_results.sh}"

SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_PACKAGE="${SUBMIT_PACKAGE:-1}"
NO_HEATMAPS="${NO_HEATMAPS:-1}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ "${SUBMIT_EVAL}" == "1" && ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi
if [[ "${SUBMIT_PACKAGE}" == "1" && ! -f "${PACKAGE_SCRIPT}" ]]; then
  echo "ERROR: package script not found: ${PACKAGE_SCRIPT}" >&2
  exit 1
fi

echo "============================================================"
echo "FuXi AR-768 poster pipeline launcher"
echo "Root            : ${ROOT_DIR}"
echo "Training script : ${TRAIN_SCRIPT}"
echo "Eval script     : ${EVAL_SCRIPT}"
echo "Package script  : ${PACKAGE_SCRIPT}"
echo "Submit eval     : ${SUBMIT_EVAL}"
echo "Submit package  : ${SUBMIT_PACKAGE}"
echo "============================================================"

TRAIN_JOBID="$(sbatch --parsable --export=ALL "${TRAIN_SCRIPT}")"

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}"
else
  RUN_ROOT="results/poster_runs/ar768_cascade_job${TRAIN_JOBID}"
fi

echo "Training job id : ${TRAIN_JOBID}"
echo "Run root        : ${RUN_ROOT}"

EVAL_JOBID=""
if [[ "${SUBMIT_EVAL}" == "1" ]]; then
  EVAL_JOBID="$(RUN_ROOT="${RUN_ROOT}" OUTPUT_ROOT="${RUN_ROOT}/poster_eval_compare" NO_HEATMAPS="${NO_HEATMAPS}" \
    sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --export=ALL "${EVAL_SCRIPT}")"
  echo "Eval job id     : ${EVAL_JOBID}"
fi

if [[ "${SUBMIT_PACKAGE}" == "1" ]]; then
  PACKAGE_DEP="${EVAL_JOBID:-${TRAIN_JOBID}}"
  PACKAGE_OUT="${PACKAGE_OUT:-results/final_reports/$(basename "${RUN_ROOT}")_pack}"
  PACKAGE_JOBID="$(RUN_ROOT="${RUN_ROOT}" PACKAGE_OUT="${PACKAGE_OUT}" \
    sbatch --parsable --dependency=afterok:${PACKAGE_DEP} --export=ALL "${PACKAGE_SCRIPT}")"
  echo "Package job id  : ${PACKAGE_JOBID}"
  echo "Package output   : ${PACKAGE_OUT}"
fi

echo "============================================================"
echo "Queued poster pipeline for ${RUN_ROOT}"
echo "============================================================"
