#!/bin/bash
#SBATCH --job-name=fuxi_a30_check_short
#SBATCH --partition=gpu_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -o logs/fuxi_a30_check_short_%j.out
#SBATCH -e logs/fuxi_a30_check_short_%j.err

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

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
slurm_require_zarr "${ZARR_STORE}"

BASE_CKPT="${BASE_CKPT:-/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/final_runs/ar768_a30_check_job${SLURM_JOB_ID}}"
RT_CFG_DIR="${OUTPUT_ROOT}/_runtime_configs"
RT_CFG_SHORT="${RT_CFG_DIR}/cascade_short.yaml"
mkdir -p "${RT_CFG_DIR}"

cp "configs/cascade_ar768_short.yaml" "${RT_CFG_SHORT}"

# A30 single-GPU sanity settings: fast enough to validate correctness and produce metrics.
sed -i "s|^  zarr_store:.*|  zarr_store: ${ZARR_STORE}|" "${RT_CFG_SHORT}"
sed -i "s|^  output_root:.*|  output_root: ${OUTPUT_ROOT}|" "${RT_CFG_SHORT}"
sed -i "s|^  exp_name:.*|  exp_name: stage_short|" "${RT_CFG_SHORT}"
sed -i "s|^  batch_size:.*|  batch_size: 2|" "${RT_CFG_SHORT}"
sed -i "s|^  num_workers:.*|  num_workers: 6|" "${RT_CFG_SHORT}"
sed -i "s|^  max_iters:.*|  max_iters: 1200|" "${RT_CFG_SHORT}"
sed -i "s|^  max_epochs:.*|  max_epochs: 20|" "${RT_CFG_SHORT}"
sed -i "s|^  eval_every:.*|  eval_every: 1|" "${RT_CFG_SHORT}"
sed -i "s|^  patience:.*|  patience: 6|" "${RT_CFG_SHORT}"
if grep -q "^  prefetch_factor:" "${RT_CFG_SHORT}"; then
  sed -i "s|^  prefetch_factor:.*|  prefetch_factor: 2|" "${RT_CFG_SHORT}"
else
  sed -i "/^  num_workers:/a\  prefetch_factor: 2" "${RT_CFG_SHORT}"
fi

EVAL_DIR="${OUTPUT_ROOT}/stage_short/evaluation_rollout20"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "============================================================"
echo "FuXi A30 one-GPU check (stage_short)"
echo "Host               : $(hostname)"
echo "Partition          : ${SLURM_JOB_PARTITION:-gpu_prio}"
echo "Output root        : ${OUTPUT_ROOT}"
echo "Runtime config     : ${RT_CFG_SHORT}"
echo "============================================================"

"${PYTHON_BIN}" -u -m src.training.train_autoregressive \
  --config "${RT_CFG_SHORT}" \
  --resume "${BASE_CKPT}" \
  --resume-model-only \
  --skip-final-test-eval \
  --status-every-steps 50

SHORT_CKPT="${OUTPUT_ROOT}/stage_short/best.pt"
if [[ ! -f "${SHORT_CKPT}" ]]; then
  echo "ERROR: stage_short checkpoint missing: ${SHORT_CKPT}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
  --checkpoint "${SHORT_CKPT}" \
  --rollout-steps 20 \
  --batch-size 1 \
  --num-workers 4 \
  --max-samples 64 \
  --save-n-samples 16 \
  --amp fp16 \
  --device cuda \
  --no-heatmaps \
  --output-dir "${EVAL_DIR}"

cat > "${OUTPUT_ROOT}/summary.txt" <<EOF
job_id=${SLURM_JOB_ID}
partition=${SLURM_JOB_PARTITION:-gpu_prio}
host=$(hostname)
base_checkpoint=${BASE_CKPT}
stage_short_checkpoint=${SHORT_CKPT}
stage_short_eval_summary=${EVAL_DIR}/summary.json
EOF

echo "Done. A30 check outputs: ${OUTPUT_ROOT}"
