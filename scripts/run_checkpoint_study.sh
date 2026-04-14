#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-study runner with non-overwriting experiment folders.
#
# Required env:
#   CHECKPOINTS="name1=/path/to/ckpt1.pt name2=/path/to/ckpt2.pt"
#
# Common optional env:
#   HORIZON_DAYS="5,10,15"
#   ROLLOUT_STEPS=""                # default: max(HORIZON_DAYS) * 4
#   RESULTS_BASE="results_new"
#   EXP_TAG="checkpoint_study"
#   RUN_ID=""                        # default: timestamp
#   PYTHON_BIN="/home/raj.ayush/.conda/envs/weather_forecast/bin/python"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
CHECKPOINTS="${CHECKPOINTS:-}"
HORIZON_DAYS="${HORIZON_DAYS:-5,10,15}"
ROLL_OUT_STEPS="${ROLLOUT_STEPS:-}"
RESULTS_BASE="${RESULTS_BASE:-results_new}"
EXP_TAG="${EXP_TAG:-checkpoint_study}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

FORECAST_FILE_NAME="${FORECAST_FILE_NAME:-forecast.nc}"
MAX_INITS="${MAX_INITS:-1}"
INIT_STRIDE="${INIT_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_MODE="${AMP_MODE:-bf16}"
DEVICE="${DEVICE:-auto}"
STATS_SAMPLES="${STATS_SAMPLES:-256}"
MAX_OUTPUT_GB="${MAX_OUTPUT_GB:-8}"

ENABLE_LORA="${ENABLE_LORA:-0}"
NO_HEATMAPS="${NO_HEATMAPS:-0}"
SKIP_DAY1_PLOT="${SKIP_DAY1_PLOT:-0}"
SKIP_T2M_PLOT="${SKIP_T2M_PLOT:-0}"

EXTRA_FORECAST_ARGS="${EXTRA_FORECAST_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

if [[ -z "${CHECKPOINTS}" ]]; then
  echo "ERROR: CHECKPOINTS is required." >&2
  echo "Example:" >&2
  echo "  CHECKPOINTS='emb768=results/a/best.pt emb1536=results/b/best.pt' bash scripts/run_checkpoint_study.sh" >&2
  exit 1
fi

max_horizon_day=0
IFS=',' read -r -a _horizon_parts <<< "${HORIZON_DAYS}"
for raw in "${_horizon_parts[@]}"; do
  day="${raw//[[:space:]]/}"
  [[ -z "${day}" ]] && continue
  if ! [[ "${day}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: HORIZON_DAYS contains non-integer value: '${day}'" >&2
    exit 1
  fi
  if (( day > max_horizon_day )); then
    max_horizon_day="${day}"
  fi
done

if (( max_horizon_day <= 0 )); then
  echo "ERROR: HORIZON_DAYS must include at least one positive day." >&2
  exit 1
fi

if [[ -z "${ROLL_OUT_STEPS}" ]]; then
  ROLL_OUT_STEPS="$((max_horizon_day * 4))"
fi

RUN_ROOT="${RESULTS_BASE}/${EXP_TAG}_${RUN_ID}"
if [[ -e "${RUN_ROOT}" ]]; then
  suffix=1
  while [[ -e "${RUN_ROOT}_${suffix}" ]]; do
    ((suffix++))
  done
  RUN_ROOT="${RUN_ROOT}_${suffix}"
fi
mkdir -p "${RUN_ROOT}"

read -r -a CKPT_ARRAY <<< "${CHECKPOINTS}"
if [[ ${#CKPT_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: unable to parse CHECKPOINTS into an array." >&2
  exit 1
fi

echo "============================================================"
echo "Checkpoint Study Runner"
echo "Host                : $(hostname)"
echo "Python              : ${PYTHON_BIN}"
echo "Experiment root     : ${RUN_ROOT}"
echo "Checkpoints         : ${CHECKPOINTS}"
echo "Horizon days        : ${HORIZON_DAYS}"
echo "Rollout steps       : ${ROLL_OUT_STEPS}"
echo "============================================================"

forecast_cmd=(
  "${PYTHON_BIN}" -m src.checkpoint_study.forecast_to_nc
  --checkpoints
  "${CKPT_ARRAY[@]}"
  --results-root "${RUN_ROOT}"
  --forecast-file-name "${FORECAST_FILE_NAME}"
  --rollout-steps "${ROLL_OUT_STEPS}"
  --max-inits "${MAX_INITS}"
  --init-stride "${INIT_STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --amp "${AMP_MODE}"
  --device "${DEVICE}"
  --stats-samples "${STATS_SAMPLES}"
  --max-output-gb "${MAX_OUTPUT_GB}"
)

[[ -n "${ZARR_STORE:-}" ]] && forecast_cmd+=(--zarr-store "${ZARR_STORE}")
[[ -n "${TRAIN_START:-}" ]] && forecast_cmd+=(--train-start "${TRAIN_START}")
[[ -n "${TRAIN_END:-}" ]] && forecast_cmd+=(--train-end "${TRAIN_END}")
[[ -n "${VAL_START:-}" ]] && forecast_cmd+=(--val-start "${VAL_START}")
[[ -n "${VAL_END:-}" ]] && forecast_cmd+=(--val-end "${VAL_END}")
[[ -n "${TEST_START:-}" ]] && forecast_cmd+=(--test-start "${TEST_START}")
[[ -n "${TEST_END:-}" ]] && forecast_cmd+=(--test-end "${TEST_END}")
[[ -n "${PRESSURE_VARS:-}" ]] && forecast_cmd+=(--pressure-vars "${PRESSURE_VARS}")
[[ -n "${SURFACE_VARS:-}" ]] && forecast_cmd+=(--surface-vars "${SURFACE_VARS}")
[[ -n "${PRESSURE_LEVELS:-}" ]] && forecast_cmd+=(--pressure-levels "${PRESSURE_LEVELS}")
[[ -n "${INIT_TIMES:-}" ]] && forecast_cmd+=(--init-times "${INIT_TIMES}")
[[ -n "${INIT_START:-}" ]] && forecast_cmd+=(--init-start "${INIT_START}")
[[ -n "${INIT_END:-}" ]] && forecast_cmd+=(--init-end "${INIT_END}")
[[ -n "${DAY1_PLOT_VARS:-}" ]] && forecast_cmd+=(--day1-plot-vars "${DAY1_PLOT_VARS}")
[[ -n "${T2M_VAR_NAME:-}" ]] && forecast_cmd+=(--t2m-var-name "${T2M_VAR_NAME}")
[[ -n "${T2M_PLOT_DAYS:-}" ]] && forecast_cmd+=(--t2m-plot-days "${T2M_PLOT_DAYS}")

if [[ "${ENABLE_LORA}" == "1" ]]; then
  forecast_cmd+=(--enable-lora)
  [[ -n "${LORA_BASE_CHECKPOINT:-}" ]] && forecast_cmd+=(--lora-base-checkpoint "${LORA_BASE_CHECKPOINT}")
  [[ -n "${LORA_RANK:-}" ]] && forecast_cmd+=(--lora-rank "${LORA_RANK}")
  [[ -n "${LORA_ALPHA:-}" ]] && forecast_cmd+=(--lora-alpha "${LORA_ALPHA}")
  [[ -n "${LORA_DROPOUT:-}" ]] && forecast_cmd+=(--lora-dropout "${LORA_DROPOUT}")
  [[ -n "${LORA_TARGET_MODULES:-}" ]] && forecast_cmd+=(--lora-target-modules "${LORA_TARGET_MODULES}")
  [[ -n "${LORA_BIAS:-}" ]] && forecast_cmd+=(--lora-bias "${LORA_BIAS}")
fi

[[ "${SKIP_DAY1_PLOT}" == "1" ]] && forecast_cmd+=(--skip-day1-plot)
[[ "${SKIP_T2M_PLOT}" == "1" ]] && forecast_cmd+=(--skip-t2m-plot)

if [[ -n "${EXTRA_FORECAST_ARGS}" ]]; then
  read -r -a _extra_forecast <<< "${EXTRA_FORECAST_ARGS}"
  forecast_cmd+=("${_extra_forecast[@]}")
fi

printf 'Running forecast command:\n  '
printf '%q ' "${forecast_cmd[@]}"
printf '\n'
"${forecast_cmd[@]}"

eval_cmd=(
  "${PYTHON_BIN}" -m src.checkpoint_study.evaluate_forecast_nc
  --results-root "${RUN_ROOT}"
  --horizon-days "${HORIZON_DAYS}"
)

[[ -n "${CLIMATOLOGY_STORE:-}" ]] && eval_cmd+=(--climatology-store "${CLIMATOLOGY_STORE}")
[[ -n "${PLOT_VARS:-}" ]] && eval_cmd+=(--plot-vars "${PLOT_VARS}")
[[ "${NO_HEATMAPS}" == "1" ]] && eval_cmd+=(--no-heatmaps)

if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
  read -r -a _extra_eval <<< "${EXTRA_EVAL_ARGS}"
  eval_cmd+=("${_extra_eval[@]}")
fi

printf 'Running eval command:\n  '
printf '%q ' "${eval_cmd[@]}"
printf '\n'
"${eval_cmd[@]}"

echo "============================================================"
echo "Done. Outputs written under: ${RUN_ROOT}"
echo "============================================================"
