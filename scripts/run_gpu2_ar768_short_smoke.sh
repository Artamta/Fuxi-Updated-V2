#!/usr/bin/env bash
set -euo pipefail

# Direct short-stage smoke test for a 2-GPU gpu2 allocation.
# Trains only stage_short and then evaluates a 20-step rollout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
BASE_CKPT="${BASE_CKPT:-results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt}"
ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${ZARR_STORE}" ]]; then
  echo "ERROR: ZARR_STORE not found: ${ZARR_STORE}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-results/poster_runs/ar768_short_gpu2_smoke}"
CFG_RT="${OUT_ROOT}/_runtime_configs/cascade_short.yaml"
mkdir -p "${OUT_ROOT}/_runtime_configs"

cp configs/cascade_ar768_short.yaml "${CFG_RT}"
sed -i "s|^  zarr_store:.*|  zarr_store: ${ZARR_STORE}|" "${CFG_RT}"
sed -i "s|^  output_root:.*|  output_root: ${OUT_ROOT}|" "${CFG_RT}"
sed -i "s|^  exp_name:.*|  exp_name: stage_short|" "${CFG_RT}"
sed -i "s|^  batch_size:.*|  batch_size: 4|" "${CFG_RT}"
sed -i "s|^  num_workers:.*|  num_workers: 4|" "${CFG_RT}"
sed -i "s|^  max_epochs:.*|  max_epochs: 2|" "${CFG_RT}"
sed -i "s|^  max_iters:.*|  max_iters: 200|" "${CFG_RT}"
sed -i "s|^  lr:.*|  lr: 0.0001|" "${CFG_RT}"
sed -i "s|^  constant_lr:.*|  constant_lr: false|" "${CFG_RT}"
sed -i "s|^  patience:.*|  patience: 2|" "${CFG_RT}"
sed -i "s|^  eval_every:.*|  eval_every: 1|" "${CFG_RT}"

if grep -q "^  prefetch_factor:" "${CFG_RT}"; then
  sed -i "s|^  prefetch_factor:.*|  prefetch_factor: 2|" "${CFG_RT}"
else
  sed -i "/^  num_workers:/a\  prefetch_factor: 2" "${CFG_RT}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "============================================================"
echo "FuXi AR-768 gpu2 short smoke"
echo "Host               : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output root        : ${OUT_ROOT}"
echo "Runtime config     : ${CFG_RT}"
echo "============================================================"

"${PYTHON_BIN}" -u -m src.training.train_autoregressive \
  --config "${CFG_RT}" \
  --resume "${BASE_CKPT}" \
  --resume-model-only \
  --skip-final-test-eval \
  --status-every-steps 50

SHORT_CKPT="${OUT_ROOT}/stage_short/best.pt"
if [[ ! -f "${SHORT_CKPT}" ]]; then
  echo "ERROR: stage_short checkpoint missing: ${SHORT_CKPT}" >&2
  exit 1
fi

EVAL_DIR="${OUT_ROOT}/stage_short/evaluation_rollout20"
"${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
  --checkpoint "${SHORT_CKPT}" \
  --rollout-steps 20 \
  --batch-size 1 \
  --num-workers 4 \
  --max-samples 64 \
  --save-n-samples 16 \
  --amp bf16 \
  --device cuda \
  --no-heatmaps \
  --output-dir "${EVAL_DIR}"

cat > "${OUT_ROOT}/summary.txt" <<EOF
host=$(hostname)
base_checkpoint=${BASE_CKPT}
stage_short_checkpoint=${SHORT_CKPT}
stage_short_eval_summary=${EVAL_DIR}/summary.json
EOF

echo "Done. Smoke outputs: ${OUT_ROOT}"