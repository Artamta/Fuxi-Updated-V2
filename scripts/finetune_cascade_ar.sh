#!/usr/bin/env bash
set -euo pipefail

# Autoregressive cascade fine-tuning runner.
# - Runs short -> medium -> long stages with checkpoint handoff.
# - Uses AR trainer (src.training.train_autoregressive).
# - Rewrites runtime config copies so env overrides are honored.

ZARR_STORE=${ZARR_STORE:-"/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"}
BASE_CKPT=${BASE_CKPT:-"/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt"}
CONFIG_SHORT=${CONFIG_SHORT:-"configs/cascade_ar768_short.yaml"}
CONFIG_MEDIUM=${CONFIG_MEDIUM:-"configs/cascade_ar768_medium.yaml"}
CONFIG_LONG=${CONFIG_LONG:-"configs/cascade_ar768_long.yaml"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/cascade_ar768"}
CKPT_NAME=${CKPT_NAME:-"best.pt"}
PYTHON_BIN=${PYTHON_BIN:-"python"}

EXP_SHORT=${EXP_SHORT:-"stage_short"}
EXP_MEDIUM=${EXP_MEDIUM:-"stage_medium"}
EXP_LONG=${EXP_LONG:-"stage_long"}

# Runtime tuning knobs (safe defaults for 4xA100).
AR_BATCH_SIZE=${AR_BATCH_SIZE:-8}
AR_NUM_WORKERS=${AR_NUM_WORKERS:-16}
AR_PREFETCH_FACTOR=${AR_PREFETCH_FACTOR:-4}
AR_EVAL_EVERY=${AR_EVAL_EVERY:-2}
AR_SKIP_FINAL_TEST_EVAL=${AR_SKIP_FINAL_TEST_EVAL:-1}
AR_MAX_ITERS=${AR_MAX_ITERS:-}
AR_MAX_EPOCHS=${AR_MAX_EPOCHS:-}
AR_PATIENCE=${AR_PATIENCE:-}
AR_LR=${AR_LR:-}

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for cfg in "${CONFIG_SHORT}" "${CONFIG_MEDIUM}" "${CONFIG_LONG}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: config not found: ${cfg}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/_runtime_configs"

prepare_cfg() {
  local src=$1
  local dst=$2
  local exp=$3

  cp "${src}" "${dst}"
  sed -i "s|^  zarr_store:.*|  zarr_store: ${ZARR_STORE}|" "${dst}"
  sed -i "s|^  output_root:.*|  output_root: ${OUTPUT_ROOT}|" "${dst}"
  sed -i "s|^  exp_name:.*|  exp_name: ${exp}|" "${dst}"
  sed -i "s|^  batch_size:.*|  batch_size: ${AR_BATCH_SIZE}|" "${dst}"
  sed -i "s|^  num_workers:.*|  num_workers: ${AR_NUM_WORKERS}|" "${dst}"
  sed -i "s|^  eval_every:.*|  eval_every: ${AR_EVAL_EVERY}|" "${dst}"

  if [[ -n "${AR_MAX_ITERS}" ]]; then
    sed -i "s|^  max_iters:.*|  max_iters: ${AR_MAX_ITERS}|" "${dst}"
  fi
  if [[ -n "${AR_MAX_EPOCHS}" ]]; then
    sed -i "s|^  max_epochs:.*|  max_epochs: ${AR_MAX_EPOCHS}|" "${dst}"
  fi
  if [[ -n "${AR_PATIENCE}" ]]; then
    sed -i "s|^  patience:.*|  patience: ${AR_PATIENCE}|" "${dst}"
  fi
  if [[ -n "${AR_LR}" ]]; then
    sed -i "s|^  lr:.*|  lr: ${AR_LR}|" "${dst}"
  fi

  if grep -q "^  prefetch_factor:" "${dst}"; then
    sed -i "s|^  prefetch_factor:.*|  prefetch_factor: ${AR_PREFETCH_FACTOR}|" "${dst}"
  else
    sed -i "/^  num_workers:/a\  prefetch_factor: ${AR_PREFETCH_FACTOR}" "${dst}"
  fi
}

CFG_SHORT_RT="${OUTPUT_ROOT}/_runtime_configs/cascade_short.yaml"
CFG_MEDIUM_RT="${OUTPUT_ROOT}/_runtime_configs/cascade_medium.yaml"
CFG_LONG_RT="${OUTPUT_ROOT}/_runtime_configs/cascade_long.yaml"

prepare_cfg "${CONFIG_SHORT}" "${CFG_SHORT_RT}" "${EXP_SHORT}"
prepare_cfg "${CONFIG_MEDIUM}" "${CFG_MEDIUM_RT}" "${EXP_MEDIUM}"
prepare_cfg "${CONFIG_LONG}" "${CFG_LONG_RT}" "${EXP_LONG}"

echo "Runtime tuning: batch=${AR_BATCH_SIZE}, workers=${AR_NUM_WORKERS}, prefetch=${AR_PREFETCH_FACTOR}, eval_every=${AR_EVAL_EVERY}, skip_final_test_eval=${AR_SKIP_FINAL_TEST_EVAL}, max_iters=${AR_MAX_ITERS:-config}, max_epochs=${AR_MAX_EPOCHS:-config}, patience=${AR_PATIENCE:-config}, lr=${AR_LR:-config}"

run_stage() {
  local cfg=$1
  local exp=$2
  local resume=$3
  local extra_args=()
  echo ""
  echo "==> Stage: ${exp}"
  if [[ "${AR_SKIP_FINAL_TEST_EVAL}" == "1" ]]; then
    extra_args+=(--skip-final-test-eval)
  fi
  "${PYTHON_BIN}" -u -m src.training.train_autoregressive \
    --config "${cfg}" \
    --resume "${resume}" \
    --resume-model-only \
    "${extra_args[@]}"
}

SHORT_CKPT="${OUTPUT_ROOT}/${EXP_SHORT}/${CKPT_NAME}"
MEDIUM_CKPT="${OUTPUT_ROOT}/${EXP_MEDIUM}/${CKPT_NAME}"
LONG_CKPT="${OUTPUT_ROOT}/${EXP_LONG}/${CKPT_NAME}"

run_stage "${CFG_SHORT_RT}" "${EXP_SHORT}" "${BASE_CKPT}"
if [[ ! -f "${SHORT_CKPT}" ]]; then
  echo "ERROR: Expected short-stage checkpoint not found: ${SHORT_CKPT}" >&2
  exit 1
fi

run_stage "${CFG_MEDIUM_RT}" "${EXP_MEDIUM}" "${SHORT_CKPT}"
if [[ ! -f "${MEDIUM_CKPT}" ]]; then
  echo "ERROR: Expected medium-stage checkpoint not found: ${MEDIUM_CKPT}" >&2
  exit 1
fi

run_stage "${CFG_LONG_RT}" "${EXP_LONG}" "${MEDIUM_CKPT}"
if [[ ! -f "${LONG_CKPT}" ]]; then
  echo "ERROR: Expected long-stage checkpoint not found: ${LONG_CKPT}" >&2
  exit 1
fi

echo "Done. Final checkpoint: ${LONG_CKPT}"
