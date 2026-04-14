#!/usr/bin/env bash
set -euo pipefail

# Dedicated AR LoRA fine-tuning runner for 2xA100.
# - Starts from emb_768 pretrained checkpoint.
# - Uses LoRA adapters on qkv/proj/fc1/fc2.
# - Runs autoregressive horizon=20 (5 days at 6h step).
# - Uses cosine LR decay (constant_lr=false) and early stopping.
# - Fails fast if CUDA is unavailable (no silent CPU fallback).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/poster_lora/ar768_lora_5day_short.yaml}"
BASE_CKPT="${BASE_CKPT:-results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/poster_runs/lora_ar768_2gpu_${RUN_ID}}"
EXP_NAME="${EXP_NAME:-stage_short_lora_5day}"

GPUS="${GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Start high to maximize memory usage, then back off automatically on OOM.
BATCH_CANDIDATES="${BATCH_CANDIDATES:-12 10 8 6 4 2 1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

MAX_EPOCHS="${MAX_EPOCHS:-30}"
MAX_ITERS="${MAX_ITERS:-12000}"
PATIENCE="${PATIENCE:-6}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LR="${LR:-2e-4}"
STATUS_EVERY_STEPS="${STATUS_EVERY_STEPS:-100}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv,proj,fc1,fc2}"
LORA_BIAS="${LORA_BIAS:-none}"

FORECAST_STEPS="${FORECAST_STEPS:-20}"
TARGET_OFFSET_STEPS="${TARGET_OFFSET_STEPS:-0}"
TEACHER_FORCING="${TEACHER_FORCING:-0.0}"
INPUT_NOISE_STD="${INPUT_NOISE_STD:-0.0}"
NOISE_COARSE_FACTOR="${NOISE_COARSE_FACTOR:-16}"
FORCE_FP16="${FORCE_FP16:-0}"
RETRY_NON_OOM_CUDA="${RETRY_NON_OOM_CUDA:-1}"
ALLOW_SINGLE_GPU_FALLBACK="${ALLOW_SINGLE_GPU_FALLBACK:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --config PATH                 Config file path
  --resume PATH                 Base checkpoint path
  --output-root PATH            Training output root
  --exp-name NAME               Experiment name under output root
  --python-bin PATH             Python executable
  --gpus N                      Number of GPUs for DataParallel
  --cuda-visible-devices LIST   CUDA_VISIBLE_DEVICES value (e.g., 0,1)
  --batch-candidates "LIST"      Batch size fallback list (space or comma separated)
  --num-workers N               DataLoader workers
  --prefetch-factor N           DataLoader prefetch factor
  --max-epochs N                Max epochs
  --max-iters N                 Max optimizer steps
  --patience N                  Early stopping patience
  --eval-every N                Validate every N epochs
  --lr FLOAT                    Learning rate
  --status-every-steps N        In-epoch logging frequency
  --force-fp16                  Force --fp16 on top of config
  --retry-non-oom-cuda          Retry smaller batches on CUDA/segfault failures
  --no-retry-non-oom-cuda       Disable non-OOM CUDA retry
  --allow-single-gpu-fallback   If 2-GPU retries fail, retry on 1 GPU
  --no-single-gpu-fallback      Disable single-GPU fallback
  --smoke                       Enable quick smoke mode
  --help                        Show this message

Notes:
  - CLI flags override environment variables.
  - Environment variables remain supported.
  - Precision follows config by default; set FORCE_FP16=1 or use --force-fp16 to force fp16.
  - By default, non-OOM CUDA failures (e.g., illegal memory access, cublas failures,
    segmentation fault) trigger fallback to smaller batch sizes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -lt 2 ]] && { echo "ERROR: --config requires a value" >&2; exit 2; }
      CONFIG_PATH="$2"
      shift 2
      ;;
    --resume)
      [[ $# -lt 2 ]] && { echo "ERROR: --resume requires a value" >&2; exit 2; }
      BASE_CKPT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -lt 2 ]] && { echo "ERROR: --output-root requires a value" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --exp-name)
      [[ $# -lt 2 ]] && { echo "ERROR: --exp-name requires a value" >&2; exit 2; }
      EXP_NAME="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -lt 2 ]] && { echo "ERROR: --python-bin requires a value" >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -lt 2 ]] && { echo "ERROR: --gpus requires a value" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      [[ $# -lt 2 ]] && { echo "ERROR: --cuda-visible-devices requires a value" >&2; exit 2; }
      export CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --batch-candidates)
      [[ $# -lt 2 ]] && { echo "ERROR: --batch-candidates requires a value" >&2; exit 2; }
      BATCH_CANDIDATES="${2//,/ }"
      shift 2
      ;;
    --num-workers)
      [[ $# -lt 2 ]] && { echo "ERROR: --num-workers requires a value" >&2; exit 2; }
      NUM_WORKERS="$2"
      shift 2
      ;;
    --prefetch-factor)
      [[ $# -lt 2 ]] && { echo "ERROR: --prefetch-factor requires a value" >&2; exit 2; }
      PREFETCH_FACTOR="$2"
      shift 2
      ;;
    --max-epochs)
      [[ $# -lt 2 ]] && { echo "ERROR: --max-epochs requires a value" >&2; exit 2; }
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --max-iters)
      [[ $# -lt 2 ]] && { echo "ERROR: --max-iters requires a value" >&2; exit 2; }
      MAX_ITERS="$2"
      shift 2
      ;;
    --patience)
      [[ $# -lt 2 ]] && { echo "ERROR: --patience requires a value" >&2; exit 2; }
      PATIENCE="$2"
      shift 2
      ;;
    --eval-every)
      [[ $# -lt 2 ]] && { echo "ERROR: --eval-every requires a value" >&2; exit 2; }
      EVAL_EVERY="$2"
      shift 2
      ;;
    --lr)
      [[ $# -lt 2 ]] && { echo "ERROR: --lr requires a value" >&2; exit 2; }
      LR="$2"
      shift 2
      ;;
    --status-every-steps)
      [[ $# -lt 2 ]] && { echo "ERROR: --status-every-steps requires a value" >&2; exit 2; }
      STATUS_EVERY_STEPS="$2"
      shift 2
      ;;
    --force-fp16)
      FORCE_FP16=1
      shift
      ;;
    --retry-non-oom-cuda)
      RETRY_NON_OOM_CUDA=1
      shift
      ;;
    --no-retry-non-oom-cuda)
      RETRY_NON_OOM_CUDA=0
      shift
      ;;
    --allow-single-gpu-fallback)
      ALLOW_SINGLE_GPU_FALLBACK=1
      shift
      ;;
    --no-single-gpu-fallback)
      ALLOW_SINGLE_GPU_FALLBACK=0
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Quick debug mode.
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  MAX_EPOCHS="${MAX_EPOCHS_SMOKE:-2}"
  MAX_ITERS="${MAX_ITERS_SMOKE:-300}"
  PATIENCE="${PATIENCE_SMOKE:-2}"
  BATCH_CANDIDATES="${BATCH_CANDIDATES_SMOKE:-4 2 1}"
  NUM_WORKERS="${NUM_WORKERS_SMOKE:-4}"
  PREFETCH_FACTOR="${PREFETCH_FACTOR_SMOKE:-2}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 1
fi

export GPUS
"${PYTHON_BIN}" - <<'PY'
import os
import torch

need = int(os.environ.get("GPUS", "2"))
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. Refusing CPU fallback.")

visible = torch.cuda.device_count()
if visible < need:
    raise SystemExit(f"ERROR: Need {need} visible GPUs, but only {visible} found.")

names = [torch.cuda.get_device_name(i) for i in range(need)]
print("CUDA healthy. Using GPUs:", names)

for i in range(need):
    x = torch.randn((1024, 1024), device=f"cuda:{i}")
    del x
PY

RUN_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${RUN_DIR}"

is_retryable_cuda_failure() {
  local log_file="$1"
  local rc="$2"

  # Common shell-level crash codes from native CUDA failures.
  if [[ "${rc}" == "139" || "${rc}" == "134" ]]; then
    return 0
  fi

  if grep -qiE "out of memory|cuda out of memory|illegal memory access|cublas_status|cudnn_status|segmentation fault|core dumped|cuda error|unable to find an engine to execute this computation|find was unable to find an engine" "${log_file}"; then
    return 0
  fi
  return 1
}

train_with_bs() {
  local bs="$1"
  local gpus_for_run="$2"
  local log_file="${RUN_DIR}/train_bs${bs}_g${gpus_for_run}.log"
  local fp16_note="from config"
  local -a precision_args=()
  if [[ "${FORCE_FP16}" == "1" ]]; then
    precision_args+=(--fp16)
    fp16_note="forced on"
  fi

  echo "============================================================"
  echo "Trying batch_size=${bs}"
  echo "Output root   : ${OUTPUT_ROOT}"
  echo "Experiment    : ${EXP_NAME}"
  echo "Config path   : ${CONFIG_PATH}"
  echo "Base checkpoint: ${BASE_CKPT}"
  echo "GPUs requested: ${gpus_for_run}"
  echo "CUDA devices  : ${CUDA_VISIBLE_DEVICES}"
  echo "FP16          : ${fp16_note}"
  echo "LR schedule   : cosine (constant_lr=false)"
  echo "============================================================"

  set +e
  "${PYTHON_BIN}" -u -m src.training.train_autoregressive \
    --config "${CONFIG_PATH}" \
    --resume "${BASE_CKPT}" \
    --resume-model-only \
    --output-root "${OUTPUT_ROOT}" \
    --exp-name "${EXP_NAME}" \
    --gpus "${gpus_for_run}" \
    "${precision_args[@]}" \
    --forecast-steps "${FORECAST_STEPS}" \
    --target-offset-steps "${TARGET_OFFSET_STEPS}" \
    --teacher-forcing "${TEACHER_FORCING}" \
    --input-noise-std "${INPUT_NOISE_STD}" \
    --noise-coarse-factor "${NOISE_COARSE_FACTOR}" \
    --batch-size "${bs}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-iters "${MAX_ITERS}" \
    --patience "${PATIENCE}" \
    --eval-every "${EVAL_EVERY}" \
    --lr "${LR}" \
    --status-every-steps "${STATUS_EVERY_STEPS}" \
    --enable-lora \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-target-modules "${LORA_TARGET_MODULES}" \
    --lora-bias "${LORA_BIAS}" \
    --skip-final-test-eval \
    2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  LAST_TRAIN_RC="${rc}"
  LAST_LOG_FILE="${log_file}"
  set -e

  return "${rc}"
}

SELECTED_BS=""
SELECTED_GPUS="${GPUS}"
LAST_TRAIN_RC=0
LAST_LOG_FILE=""
SAW_RETRYABLE_FAILURE=0

run_batch_search() {
  local gpus_for_run="$1"
  SELECTED_BS=""
  SAW_RETRYABLE_FAILURE=0

  for bs in ${BATCH_CANDIDATES}; do
    if [[ "${gpus_for_run}" -gt 1 && "${bs}" -lt "${gpus_for_run}" ]]; then
      echo "Skipping batch_size=${bs} for ${gpus_for_run} GPUs (insufficient per-GPU shard)."
      continue
    fi

    if train_with_bs "${bs}" "${gpus_for_run}"; then
      SELECTED_BS="${bs}"
      return 0
    fi

    if is_retryable_cuda_failure "${LAST_LOG_FILE}" "${LAST_TRAIN_RC}"; then
      SAW_RETRYABLE_FAILURE=1
      if grep -qiE "out of memory|cuda out of memory" "${LAST_LOG_FILE}"; then
        echo "OOM at batch_size=${bs}. Trying smaller batch..."
      elif [[ "${RETRY_NON_OOM_CUDA}" == "1" ]]; then
        echo "Retryable CUDA failure at batch_size=${bs} (rc=${LAST_TRAIN_RC}). Trying smaller batch..."
      else
        echo "Training failed for non-OOM CUDA reason and retry is disabled. See ${LAST_LOG_FILE}" >&2
        return 2
      fi
      continue
    fi

    echo "Training failed for non-retryable reason. See ${LAST_LOG_FILE}" >&2
    return 2
  done

  return 0
}

if ! run_batch_search "${GPUS}"; then
  exit 1
fi

if [[ -z "${SELECTED_BS}" && "${ALLOW_SINGLE_GPU_FALLBACK}" == "1" && "${GPUS}" -gt 1 && "${SAW_RETRYABLE_FAILURE}" == "1" ]]; then
  echo "All batch sizes failed with retryable CUDA errors on ${GPUS} GPUs. Retrying on 1 GPU..."
  if ! run_batch_search "1"; then
    exit 1
  fi
  if [[ -n "${SELECTED_BS}" ]]; then
    SELECTED_GPUS="1"
  fi
fi

if [[ -z "${SELECTED_BS}" ]]; then
  echo "ERROR: All candidate batch sizes failed. Last log: ${LAST_LOG_FILE}" >&2
  exit 1
fi

BEST_CKPT="${RUN_DIR}/best.pt"
LAST_CKPT="${RUN_DIR}/last.pt"
CKPT_TO_EVAL="${BEST_CKPT}"
if [[ ! -f "${CKPT_TO_EVAL}" ]]; then
  CKPT_TO_EVAL="${LAST_CKPT}"
fi
if [[ ! -f "${CKPT_TO_EVAL}" ]]; then
  echo "ERROR: no checkpoint found in ${RUN_DIR}" >&2
  exit 1
fi

EVAL_DIR="${RUN_DIR}/evaluation_rollout20"
"${PYTHON_BIN}" -m src.evaluation.evaluate_checkpoint \
  --checkpoint "${CKPT_TO_EVAL}" \
  --rollout-steps 20 \
  --batch-size 1 \
  --num-workers 4 \
  --max-samples 256 \
  --save-n-samples 24 \
  --amp bf16 \
  --device cuda \
  --no-heatmaps \
  --output-dir "${EVAL_DIR}"

echo "Done."
echo "Selected batch size : ${SELECTED_BS}"
echo "Selected GPUs       : ${SELECTED_GPUS}"
echo "Run directory       : ${RUN_DIR}"
echo "Eval summary        : ${EVAL_DIR}/summary.json"
