#!/bin/bash
# Multi-stage FuXi pretraining sweep using a Slurm job array.

set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash scripts/sweep_pretraining.sh [stage] [--dry-run]

Stages:
  stage1 (default), stage2, stage3
USAGE
}

STAGE="${1:-stage1}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]] || [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  if [[ "${1:-}" == "--dry-run" ]]; then
    STAGE="stage1"
  fi
fi

case "${STAGE}" in
  broad) STAGE="stage1" ;;
  focused) STAGE="stage2" ;;
  final) STAGE="stage3" ;;
  stage1|stage2|stage3) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "ERROR: unknown stage '${STAGE}'"; usage; exit 1 ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs sweeps

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr}"
if [ ! -d "${ZARR_STORE}" ]; then
  echo "ERROR: zarr store not found: ${ZARR_STORE}"
  exit 1
fi

PARTITION="${PARTITION:-GPU-AI}"
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM_PER_JOB="${MEM_PER_JOB:-180G}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUNS_ROOT="${RUNS_ROOT:-Models_paper/pretrain_sweep}"
CONDA_ENV="${CONDA_ENV:-weather_forecast}"
AMP="${AMP:-bf16}"
LOSS="${LOSS:-l1}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_TEST_EVAL="${RUN_TEST_EVAL:-0}"

TRAIN_START="1979-01-01"
TRAIN_END="2018-12-31"
VAL_START="2019-01-01"
VAL_END="2020-12-31"
TEST_START="2021-01-01"
TEST_END="2022-12-31"

NUM_HEADS=8
WINDOW_SIZE=8
DEPTH_PRE=2
DEPTH_POST=2
BETA1=0.9
BETA2=0.95
USE_CHECKPOINT_FLAG="--use-checkpoint"

case "${STAGE}" in
  stage1)
    TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
    MAX_EPOCHS="${MAX_EPOCHS:-8}"
    MAX_ITERS="${MAX_ITERS:-5000}"
    PATIENCE="${PATIENCE:-3}"
    EXPERIMENTS=(
      "s1_e768_d24_lr15_wd05_dp10_gc10|768|24|1.5e-4|0.05|0.10|1.0"
      "s1_e1024_d32_lr12_wd05_dp10_gc10|1024|32|1.2e-4|0.05|0.10|1.0"
      "s1_e1280_d36_lr10_wd03_dp10_gc10|1280|36|1.0e-4|0.03|0.10|1.0"
      "s1_e1536_d40_lr10_wd05_dp12_gc10|1536|40|1.0e-4|0.05|0.12|1.0"
    )
    ;;
  stage2)
    TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
    MAX_EPOCHS="${MAX_EPOCHS:-18}"
    MAX_ITERS="${MAX_ITERS:-18000}"
    PATIENCE="${PATIENCE:-5}"
    EXPERIMENTS=(
      "s2_e1024_d32_lr10_wd03_dp10_gc10|1024|32|1.0e-4|0.03|0.10|1.0"
      "s2_e1280_d36_lr11_wd05_dp12_gc10|1280|36|1.1e-4|0.05|0.12|1.0"
      "s2_e1536_d44_lr12_wd07_dp15_gc12|1536|44|1.2e-4|0.07|0.15|1.2"
    )
    ;;
  stage3)
    TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
    MAX_EPOCHS="${MAX_EPOCHS:-50}"
    MAX_ITERS="${MAX_ITERS:-40000}"
    PATIENCE="${PATIENCE:-10}"
    EXPERIMENTS=(
      "s3_final_a_e1280_d36_lr10_wd05_dp12_gc10|1280|36|1.0e-4|0.05|0.12|1.0"
      "s3_final_b_e1536_d40_lr10_wd05_dp12_gc10|1536|40|1.0e-4|0.05|0.12|1.0"
    )
    ;;
esac

CONFIG_FILE="sweeps/${STAGE}_$(date +%Y%m%d_%H%M%S).tsv"
printf "name\tembed\tdepth_mid\tlr\tweight_decay\tdrop_path\tgrad_clip\n" > "${CONFIG_FILE}"
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME EMBED DMID LR WD DP GC <<< "${exp}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${NAME}" "${EMBED}" "${DMID}" "${LR}" "${WD}" "${DP}" "${GC}" >> "${CONFIG_FILE}"
done

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Stage ${STAGE}: ${NUM_EXPERIMENTS} experiments"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Dry run mode: not submitting."
  sed -n '1,40p' "${CONFIG_FILE}"
  exit 0
fi

RUNS_DIR_STAGE="${RUNS_ROOT}/${STAGE}"

export ROOT_DIR CONFIG_FILE CONDA_ENV GPUS_PER_JOB RUNS_DIR_STAGE ZARR_STORE
export TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END
export NUM_HEADS WINDOW_SIZE DEPTH_PRE DEPTH_POST USE_CHECKPOINT_FLAG
export BATCH_PER_GPU ACCUM_STEPS MAX_EPOCHS MAX_ITERS PATIENCE NUM_WORKERS
export BETA1 BETA2 LOSS AMP RUN_TEST_EVAL

ARRAY_SPEC="0-$((NUM_EXPERIMENTS - 1))%${MAX_PARALLEL}"

SUBMIT_OUT=$(sbatch \
  --job-name="fuxi_${STAGE}" \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS_PER_JOB}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM_PER_JOB}" \
  --time="${TIME_LIMIT}" \
  --array="${ARRAY_SPEC}" \
  --output="logs/${STAGE}_%A_%a.out" \
  --error="logs/${STAGE}_%A_%a.err" \
  <<'SBATCH_SCRIPT'
#!/bin/bash
set -euo pipefail

source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${CONFIG_FILE}")"
IFS=$'\t' read -r NAME EMBED DEPTH_MID LR WEIGHT_DECAY DROP_PATH GRAD_CLIP <<< "${LINE}"

NGPUS="${SLURM_GPUS_ON_NODE:-${GPUS_PER_JOB}}"
MASTER_PORT=$((10000 + (RANDOM % 50000)))

RUN_TEST_FLAG=""
if [ "${RUN_TEST_EVAL}" = "1" ]; then
  RUN_TEST_FLAG="--run-test-eval"
fi

accelerate launch \
  --multi_gpu \
  --num_processes="${NGPUS}" \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port="${MASTER_PORT}" \
  --dynamo_backend=no \
  --mixed_precision="${AMP}" \
  --module src.pretraining.pretrain \
  --zarr-store "${ZARR_STORE}" \
  --train-start "${TRAIN_START}" \
  --train-end "${TRAIN_END}" \
  --val-start "${VAL_START}" \
  --val-end "${VAL_END}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --exp-name "${NAME}" \
  --runs-dir "${RUNS_DIR_STAGE}" \
  --embed-dim "${EMBED}" \
  --num-heads "${NUM_HEADS}" \
  --window-size "${WINDOW_SIZE}" \
  --depth-pre "${DEPTH_PRE}" \
  --depth-mid "${DEPTH_MID}" \
  --depth-post "${DEPTH_POST}" \
  --drop-path-rate "${DROP_PATH}" \
  ${USE_CHECKPOINT_FLAG} \
  --batch-size "${BATCH_PER_GPU}" \
  --accum-steps "${ACCUM_STEPS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --max-iters "${MAX_ITERS}" \
  --patience "${PATIENCE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --grad-clip "${GRAD_CLIP}" \
  --loss "${LOSS}" \
  --device cuda \
  --amp "${AMP}" \
  ${RUN_TEST_FLAG}
SBATCH_SCRIPT
)

echo "${SUBMIT_OUT}"
JOB_ID=$(echo "${SUBMIT_OUT}" | awk '{print $4}')
echo "Submitted stage '${STAGE}' as job array ${JOB_ID}."
