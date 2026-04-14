#!/usr/bin/env bash
set -euo pipefail

# Dedicated cascade checkpoint-study runner.
#
# Required env:
#   SHORT_CHECKPOINT=/path/to/short.pt
#   MEDIUM_CHECKPOINT=/path/to/medium.pt
#   LONG_CHECKPOINT=/path/to/long.pt
#
# Common optional env:
#   CASCADE_NAME="cascade"
#   HORIZON_DAYS="5,10,15"
#   STAGE_STEPS=""                 # default: even split of max(HORIZON_DAYS) * 4
#   RESULTS_BASE="results_new"
#   EXP_TAG="cascade_study"
#   RUN_ID=""                       # default: timestamp

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"

SHORT_CHECKPOINT="${SHORT_CHECKPOINT:-}"
MEDIUM_CHECKPOINT="${MEDIUM_CHECKPOINT:-}"
LONG_CHECKPOINT="${LONG_CHECKPOINT:-}"

CASCADE_NAME="${CASCADE_NAME:-cascade}"
HORIZON_DAYS="${HORIZON_DAYS:-5,10,15}"
STAGE_STEPS="${STAGE_STEPS:-}"

RESULTS_BASE="${RESULTS_BASE:-results_new}"
EXP_TAG="${EXP_TAG:-cascade_study}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

FORECAST_FILE_NAME="${FORECAST_FILE_NAME:-forecast_cascade.nc}"
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

if [[ -z "${SHORT_CHECKPOINT}" || -z "${MEDIUM_CHECKPOINT}" || -z "${LONG_CHECKPOINT}" ]]; then
  echo "ERROR: SHORT_CHECKPOINT, MEDIUM_CHECKPOINT, and LONG_CHECKPOINT are required." >&2
  echo "Example:" >&2
  echo "  SHORT_CHECKPOINT=... MEDIUM_CHECKPOINT=... LONG_CHECKPOINT=... bash scripts/run_cascade_checkpoint_study.sh" >&2
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

if [[ -z "${STAGE_STEPS}" ]]; then
  total_steps="$((max_horizon_day * 4))"
  s1="$((total_steps / 3))"
  s2="$((total_steps / 3))"
  s3="$((total_steps - s1 - s2))"
  STAGE_STEPS="${s1},${s2},${s3}"
fi

IFS=',' read -r -a _stage_parts <<< "${STAGE_STEPS}"
if [[ ${#_stage_parts[@]} -ne 3 ]]; then
  echo "ERROR: STAGE_STEPS must contain exactly three integers (short,medium,long)." >&2
  exit 1
fi

stage_total=0
for raw in "${_stage_parts[@]}"; do
  s="${raw//[[:space:]]/}"
  if ! [[ "${s}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STAGE_STEPS contains non-integer value: '${s}'" >&2
    exit 1
  fi
  if (( s <= 0 )); then
    echo "ERROR: STAGE_STEPS must be positive, got ${s}" >&2
    exit 1
  fi
  stage_total="$((stage_total + s))"
done

needed_steps="$((max_horizon_day * 4))"
if (( stage_total < needed_steps )); then
  echo "ERROR: stage total steps (${stage_total}) is smaller than required by HORIZON_DAYS (${needed_steps})." >&2
  exit 1
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

echo "============================================================"
echo "Cascade Checkpoint Study Runner"
echo "Host                : $(hostname)"
echo "Python              : ${PYTHON_BIN}"
echo "Experiment root     : ${RUN_ROOT}"
echo "Cascade name        : ${CASCADE_NAME}"
echo "Stage steps         : ${STAGE_STEPS}"
echo "Horizon days        : ${HORIZON_DAYS}"
echo "Short checkpoint    : ${SHORT_CHECKPOINT}"
echo "Medium checkpoint   : ${MEDIUM_CHECKPOINT}"
echo "Long checkpoint     : ${LONG_CHECKPOINT}"
echo "============================================================"

forecast_cmd=(
  "${PYTHON_BIN}" -m src.checkpoint_study.forecast_cascade_to_nc
  --short-checkpoint "${SHORT_CHECKPOINT}"
  --medium-checkpoint "${MEDIUM_CHECKPOINT}"
  --long-checkpoint "${LONG_CHECKPOINT}"
  --cascade-name "${CASCADE_NAME}"
  --stage-steps "${STAGE_STEPS}"
  --results-root "${RUN_ROOT}"
  --forecast-file-name "${FORECAST_FILE_NAME}"
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

printf 'Running cascade forecast command:\n  '
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
