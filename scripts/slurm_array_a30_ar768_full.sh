#!/bin/bash
#SBATCH --job-name=fuxi_a30_ar768_arr
#SBATCH --partition=gpu_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --array=0-1
#SBATCH --exclude=cn3
#SBATCH -o logs/fuxi_a30_ar768_arr_%A_%a.out
#SBATCH -e logs/fuxi_a30_ar768_arr_%A_%a.err

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

CONFIG_SHORT="${CONFIG_SHORT:-configs/cascade_ar768_short.yaml}"
CONFIG_MEDIUM="${CONFIG_MEDIUM:-configs/cascade_ar768_medium.yaml}"
CONFIG_LONG="${CONFIG_LONG:-configs/cascade_ar768_long.yaml}"

PROFILES=("a30_1gpu" "a30_2gpu")
REQUESTED_GPUS=(1 2)
BATCH_SIZES=(2 4)
NUM_WORKERS=(6 8)
PREFETCH_FACTORS=(2 3)
MAX_ITERS=(12000 12000)
MAX_EPOCHS=(40 40)
PATIENCE=(8 8)
LRS=(0.0000003 0.0000003)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX < 0 || IDX >= ${#PROFILES[@]} )); then
  echo "ERROR: array index ${IDX} is out of range" >&2
  exit 1
fi

PROFILE=${PROFILES[$IDX]}
REQ_GPUS=${REQUESTED_GPUS[$IDX]}

# Respect Slurm GPU binding. For single-GPU profile, pin to local device 0
# only when Slurm did not set CUDA visibility explicitly.
if (( REQ_GPUS == 1 )); then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
else
  VISIBLE_GPUS=0
fi

if [[ -z "${VISIBLE_GPUS}" ]]; then
  VISIBLE_GPUS=0
fi

if (( VISIBLE_GPUS <= 0 )); then
  echo "ERROR: No visible CUDA GPUs on host $(hostname)" >&2
  exit 1
fi

EFFECTIVE_GPUS=${REQ_GPUS}
if (( REQ_GPUS > VISIBLE_GPUS )); then
  echo "WARN: Requested ${REQ_GPUS} GPU(s) but only ${VISIBLE_GPUS} visible; falling back to ${VISIBLE_GPUS}." >&2
  EFFECTIVE_GPUS=${VISIBLE_GPUS}
fi

export AR_BATCH_SIZE=${BATCH_SIZES[$IDX]}
export AR_NUM_WORKERS=${NUM_WORKERS[$IDX]}
export AR_PREFETCH_FACTOR=${PREFETCH_FACTORS[$IDX]}
export AR_MAX_ITERS=${MAX_ITERS[$IDX]}
export AR_MAX_EPOCHS=${MAX_EPOCHS[$IDX]}
export AR_PATIENCE=${PATIENCE[$IDX]}
export AR_LR=${LRS[$IDX]}

# If the 2-GPU profile lands on a 1-GPU allocation, downgrade to stable knobs.
if (( REQ_GPUS >= 2 && EFFECTIVE_GPUS < 2 )); then
  export AR_BATCH_SIZE=2
  export AR_NUM_WORKERS=6
  export AR_PREFETCH_FACTOR=2
fi

export AR_EVAL_EVERY=${AR_EVAL_EVERY:-2}
export AR_SKIP_FINAL_TEST_EVAL=${AR_SKIP_FINAL_TEST_EVAL:-1}

RUN_ROOT_BASE=${RUN_ROOT_BASE:-results/final_runs}
OUTPUT_ROOT="${RUN_ROOT_BASE}/ar768_a30_array_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_${PROFILE}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "============================================================"
echo "FuXi A30 AR768 full cascade (array profile)"
echo "Host               : $(hostname)"
echo "Partition          : ${SLURM_JOB_PARTITION:-gpu_prio}"
echo "Job/Task           : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Profile            : ${PROFILE}"
echo "Requested GPUs     : ${REQ_GPUS}"
echo "Visible GPUs       : ${VISIBLE_GPUS}"
echo "Effective GPUs     : ${EFFECTIVE_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Output root        : ${OUTPUT_ROOT}"
echo "AR tuning          : batch=${AR_BATCH_SIZE}, workers=${AR_NUM_WORKERS}, prefetch=${AR_PREFETCH_FACTOR}, max_iters=${AR_MAX_ITERS}, max_epochs=${AR_MAX_EPOCHS}, lr=${AR_LR}"
echo "============================================================"

probe_cuda_stack() {
  "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
    n = torch.cuda.device_count()
    if n < 1:
        raise RuntimeError("No CUDA devices detected")
    for i in range(n):
        _ = torch.cuda.get_device_properties(i).name
except Exception as exc:  # pragma: no cover
    print(f"CUDA probe failed: {exc}", file=sys.stderr)
    raise
PY
}

if ! probe_cuda_stack; then
  if (( REQ_GPUS >= 2 )); then
    echo "WARN: CUDA probe failed for multi-GPU startup; retrying with single-GPU fallback." >&2
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)"
    else
      export CUDA_VISIBLE_DEVICES=0
    fi
    EFFECTIVE_GPUS=1
    export AR_BATCH_SIZE=2
    export AR_NUM_WORKERS=6
    export AR_PREFETCH_FACTOR=2
    echo "Fallback settings   : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, batch=${AR_BATCH_SIZE}, workers=${AR_NUM_WORKERS}, prefetch=${AR_PREFETCH_FACTOR}"
    probe_cuda_stack
  else
    echo "ERROR: CUDA probe failed for single-GPU profile." >&2
    exit 1
  fi
fi

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

LONG_CKPT="${OUTPUT_ROOT}/stage_long/best.pt"
if [[ ! -f "${LONG_CKPT}" ]]; then
  echo "ERROR: final checkpoint missing: ${LONG_CKPT}" >&2
  exit 1
fi

# Quick comparison eval focused on ACC/RMSE curves + spectra (no heatmaps).
NO_HEATMAPS=1 \
ROLL_STEPS=${ROLL_STEPS:-20} \
BATCH_SIZE=${EVAL_BATCH_SIZE:-1} \
NUM_WORKERS=${EVAL_NUM_WORKERS:-4} \
MAX_SAMPLES=${EVAL_MAX_SAMPLES:-128} \
SAVE_N_SAMPLES=${SAVE_N_SAMPLES:-16} \
AMP_MODE=${AMP_MODE:-fp16} \
DEVICE_MODE=${DEVICE_MODE:-cuda} \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_poster_eval_compare.sh "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/poster_eval_compare_quick"

PACKAGE_OUT="results/final_reports/$(basename "${OUTPUT_ROOT}")_pack"
bash scripts/package_cascade_results.sh "${OUTPUT_ROOT}" "${PACKAGE_OUT}"

cat > "${OUTPUT_ROOT}/summary.txt" <<EOF
job_id=${SLURM_JOB_ID}
array_task_id=${SLURM_ARRAY_TASK_ID}
profile=${PROFILE}
partition=${SLURM_JOB_PARTITION:-gpu_prio}
host=$(hostname)
base_checkpoint=${BASE_CKPT}
final_checkpoint=${LONG_CKPT}
compare_summary=${OUTPUT_ROOT}/poster_eval_compare_quick/comparison_summary.csv
packaged_output=${PACKAGE_OUT}
EOF

echo "Done. Full A30 cascade outputs: ${OUTPUT_ROOT}"
