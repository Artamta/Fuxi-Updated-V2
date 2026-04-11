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
AR_INPUT_NOISE_STD=${AR_INPUT_NOISE_STD:-0.0}
AR_NOISE_COARSE_FACTOR=${AR_NOISE_COARSE_FACTOR:-16}
AR_STATUS_EVERY_STEPS=${AR_STATUS_EVERY_STEPS:-200}
AR_ENABLE_LORA=${AR_ENABLE_LORA:-0}
AR_LORA_RANK=${AR_LORA_RANK:-}
AR_LORA_ALPHA=${AR_LORA_ALPHA:-}
AR_LORA_DROPOUT=${AR_LORA_DROPOUT:-}
AR_LORA_TARGET_MODULES=${AR_LORA_TARGET_MODULES:-}
AR_LORA_BIAS=${AR_LORA_BIAS:-}
AR_LORA_TRAIN_BASE=${AR_LORA_TRAIN_BASE:-0}

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
  if grep -q "^  input_noise_std:" "${dst}"; then
    sed -i "s|^  input_noise_std:.*|  input_noise_std: ${AR_INPUT_NOISE_STD}|" "${dst}"
  else
    sed -i "/^  eval_every:/a\  input_noise_std: ${AR_INPUT_NOISE_STD}" "${dst}"
  fi
  if grep -q "^  noise_coarse_factor:" "${dst}"; then
    sed -i "s|^  noise_coarse_factor:.*|  noise_coarse_factor: ${AR_NOISE_COARSE_FACTOR}|" "${dst}"
  else
    sed -i "/^  input_noise_std:/a\  noise_coarse_factor: ${AR_NOISE_COARSE_FACTOR}" "${dst}"
  fi
  if grep -q "^  status_every_steps:" "${dst}"; then
    sed -i "s|^  status_every_steps:.*|  status_every_steps: ${AR_STATUS_EVERY_STEPS}|" "${dst}"
  else
    sed -i "/^  eval_every:/a\  status_every_steps: ${AR_STATUS_EVERY_STEPS}" "${dst}"
  fi

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

echo "Runtime tuning: batch=${AR_BATCH_SIZE}, workers=${AR_NUM_WORKERS}, prefetch=${AR_PREFETCH_FACTOR}, eval_every=${AR_EVAL_EVERY}, status_every=${AR_STATUS_EVERY_STEPS}, noise_std=${AR_INPUT_NOISE_STD}, noise_coarse_factor=${AR_NOISE_COARSE_FACTOR}, skip_final_test_eval=${AR_SKIP_FINAL_TEST_EVAL}, max_iters=${AR_MAX_ITERS:-config}, max_epochs=${AR_MAX_EPOCHS:-config}, patience=${AR_PATIENCE:-config}, lr=${AR_LR:-config}, enable_lora=${AR_ENABLE_LORA}, lora_rank=${AR_LORA_RANK:-config}, lora_alpha=${AR_LORA_ALPHA:-config}, lora_dropout=${AR_LORA_DROPOUT:-config}, lora_target_modules=${AR_LORA_TARGET_MODULES:-config}, lora_bias=${AR_LORA_BIAS:-config}, lora_train_base=${AR_LORA_TRAIN_BASE}"

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
  extra_args+=(--input-noise-std "${AR_INPUT_NOISE_STD}")
  extra_args+=(--noise-coarse-factor "${AR_NOISE_COARSE_FACTOR}")
  extra_args+=(--status-every-steps "${AR_STATUS_EVERY_STEPS}")
  if [[ "${AR_ENABLE_LORA}" == "1" ]]; then
    extra_args+=(--enable-lora)
    if [[ -n "${AR_LORA_RANK}" ]]; then
      extra_args+=(--lora-rank "${AR_LORA_RANK}")
    fi
    if [[ -n "${AR_LORA_ALPHA}" ]]; then
      extra_args+=(--lora-alpha "${AR_LORA_ALPHA}")
    fi
    if [[ -n "${AR_LORA_DROPOUT}" ]]; then
      extra_args+=(--lora-dropout "${AR_LORA_DROPOUT}")
    fi
    if [[ -n "${AR_LORA_TARGET_MODULES}" ]]; then
      extra_args+=(--lora-target-modules "${AR_LORA_TARGET_MODULES}")
    fi
    if [[ -n "${AR_LORA_BIAS}" ]]; then
      extra_args+=(--lora-bias "${AR_LORA_BIAS}")
    fi
    if [[ "${AR_LORA_TRAIN_BASE}" == "1" ]]; then
      extra_args+=(--lora-train-base)
    fi
  fi
  "${PYTHON_BIN}" -u -m src.training.train_autoregressive \
    --config "${cfg}" \
    --resume "${resume}" \
    --resume-model-only \
    "${extra_args[@]}"
}

resolve_stage_ckpt() {
  local exp=$1
  local preferred="${OUTPUT_ROOT}/${exp}/${CKPT_NAME}"
  local fallback="${OUTPUT_ROOT}/${exp}/last.pt"

  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  if [[ -f "${fallback}" ]]; then
    echo "WARN: ${preferred} missing; falling back to ${fallback}" >&2
    if [[ "${preferred}" != "${fallback}" ]]; then
      ln -sfn "$(basename "${fallback}")" "${preferred}" 2>/dev/null || cp -f "${fallback}" "${preferred}"
      echo "INFO: Materialized ${preferred} from fallback checkpoint." >&2
    fi
    echo "${preferred}"
    return 0
  fi

  echo "ERROR: No checkpoint found for stage ${exp}. Expected ${preferred} or ${fallback}" >&2
  return 1
}

run_stage "${CFG_SHORT_RT}" "${EXP_SHORT}" "${BASE_CKPT}"
SHORT_CKPT="$(resolve_stage_ckpt "${EXP_SHORT}")" || exit 1

run_stage "${CFG_MEDIUM_RT}" "${EXP_MEDIUM}" "${SHORT_CKPT}"
MEDIUM_CKPT="$(resolve_stage_ckpt "${EXP_MEDIUM}")" || exit 1

run_stage "${CFG_LONG_RT}" "${EXP_LONG}" "${MEDIUM_CKPT}"
LONG_CKPT="$(resolve_stage_ckpt "${EXP_LONG}")" || exit 1

echo "Done. Final checkpoint: ${LONG_CKPT}"
