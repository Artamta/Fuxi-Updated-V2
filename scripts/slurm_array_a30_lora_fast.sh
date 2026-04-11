#!/bin/bash
#SBATCH --job-name=fuxi_lora_a30_fast
#SBATCH --partition=gpu_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-18:00:00
#SBATCH --array=1-6
#SBATCH --exclude=cn3
#SBATCH -o logs/fuxi_lora_a30_fast_%A_%a.out
#SBATCH -e logs/fuxi_lora_a30_fast_%A_%a.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"
mkdir -p logs results

COMMON_SH="${ROOT_DIR}/scripts/slurm_common.sh"
if [[ ! -f "${COMMON_SH}" ]]; then
  echo "ERROR: missing ${COMMON_SH}" >&2
  exit 1
fi
source "${COMMON_SH}"

slurm_activate_conda
slurm_export_nccl_defaults

PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/weather_forecast/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
BASE_CKPT="${BASE_CKPT:-/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt}"

if [[ ! -d "${ZARR_STORE}" ]]; then
  echo "ERROR: zarr store not found: ${ZARR_STORE}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi

NOISE_STD="${NOISE_STD:-0.02}"
TEACHER_FORCING="${TEACHER_FORCING:-0.1}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv,proj,fc1,fc2}"

MAX_ITERS="${MAX_ITERS:-10000}"
MAX_EPOCHS="${MAX_EPOCHS:-35}"
PATIENCE="${PATIENCE:-6}"
STATUS_EVERY_STEPS="${STATUS_EVERY_STEPS:-200}"
LR="${LR:-1e-4}"

case "${SLURM_ARRAY_TASK_ID}" in
  1) LORA_RANK=8;  NOISE_COARSE_FACTOR=8 ;;
  2) LORA_RANK=8;  NOISE_COARSE_FACTOR=16 ;;
  3) LORA_RANK=16; NOISE_COARSE_FACTOR=8 ;;
  4) LORA_RANK=16; NOISE_COARSE_FACTOR=16 ;;
  5) LORA_RANK=32; NOISE_COARSE_FACTOR=8 ;;
  6) LORA_RANK=32; NOISE_COARSE_FACTOR=16 ;;
  *)
    echo "ERROR: unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

LORA_ALPHA=$(( LORA_RANK * 2 ))
NOISE_TAG="${NOISE_STD//./p}"
EXP_NAME="r${LORA_RANK}_a${LORA_ALPHA}_n${NOISE_TAG}_c${NOISE_COARSE_FACTOR}_t${SLURM_ARRAY_TASK_ID}"

RUN_TAG="${RUN_TAG:-a30_lora_simple_job${SLURM_JOB_ID}}"
RESULT_ROOT="${RESULT_ROOT:-results/fast_runs/${RUN_TAG}}"
RUN_DIR="${RESULT_ROOT}/job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}_${EXP_NAME}"
mkdir -p "${RUN_DIR}"

# Persist all stdout/stderr to a stable run-level log in addition to Slurm logs.
exec > >(tee -a "${RUN_DIR}/logs.txt") 2>&1

count_visible_devices() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES}"
    local count
    count=$(awk -F',' '{print NF}' <<< "${csv}")
    echo "${count}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
    return 0
  fi

  echo "0"
}

detect_mig_mode() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q 'MIG'; then
    echo "1"
  else
    echo "0"
  fi
}

first_visible_gpu_mem_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi

  local mem
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
  if [[ -z "${mem}" ]]; then
    echo "0"
    return 0
  fi
  echo "${mem}"
}

resolve_stage_ckpt() {
  local stage_name=$1
  local preferred="${RUN_DIR}/${stage_name}/best.pt"
  local fallback="${RUN_DIR}/${stage_name}/last.pt"

  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi
  if [[ -f "${fallback}" ]]; then
    echo "WARN: ${preferred} missing; using ${fallback}" >&2
    cp -f "${fallback}" "${preferred}"
    echo "${preferred}"
    return 0
  fi

  echo "ERROR: neither ${preferred} nor ${fallback} exists" >&2
  return 1
}

VISIBLE_GPUS=$(count_visible_devices)
if [[ -z "${VISIBLE_GPUS}" || "${VISIBLE_GPUS}" == "0" ]]; then
  echo "ERROR: no visible GPUs for this task" >&2
  exit 1
fi

MIG_MODE=$(detect_mig_mode)

BATCH_SIZE=4
NUM_WORKERS=8
PREFETCH_FACTOR=3
GPUS_TO_USE=2

if (( VISIBLE_GPUS < GPUS_TO_USE )); then
  GPUS_TO_USE=${VISIBLE_GPUS}
fi

if [[ "${MIG_MODE}" == "1" ]]; then
  # MIG slices can be unstable with DataParallel; force single-device execution.
  BATCH_SIZE=2
  NUM_WORKERS=6
  PREFETCH_FACTOR=2
  if (( GPUS_TO_USE > 1 )); then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      export CUDA_VISIBLE_DEVICES
      CUDA_VISIBLE_DEVICES=$(cut -d',' -f1 <<< "${CUDA_VISIBLE_DEVICES}")
    else
      export CUDA_VISIBLE_DEVICES=0
    fi
    GPUS_TO_USE=1
  fi

  FIRST_GPU_MEM_MB=$(first_visible_gpu_mem_mb)
  if [[ -n "${FIRST_GPU_MEM_MB}" && "${FIRST_GPU_MEM_MB}" != "0" && "${FIRST_GPU_MEM_MB}" -lt 12000 ]]; then
    echo "ERROR: MIG allocation provides ${FIRST_GPU_MEM_MB} MiB GPU memory; AR768 LoRA cannot run reliably below 12 GiB." >&2
    echo "ERROR: Requeue on full A30 nodes only (current node: $(hostname))." >&2
    exit 66
  fi
fi

echo "============================================================"
echo "FuXi A30 LoRA fast array"
echo "Host                  : $(hostname)"
echo "Partition             : ${SLURM_JOB_PARTITION:-gpu_prio}"
echo "Job/Task              : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Run directory         : ${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Visible devices       : ${VISIBLE_GPUS}"
echo "MIG mode              : ${MIG_MODE}"
echo "Using GPUs            : ${GPUS_TO_USE}"
echo "LoRA rank/alpha       : ${LORA_RANK}/${LORA_ALPHA}"
echo "LoRA dropout/targets  : ${LORA_DROPOUT} / ${LORA_TARGET_MODULES}"
echo "Noise std/coarse      : ${NOISE_STD} / ${NOISE_COARSE_FACTOR}"
echo "Teacher forcing       : ${TEACHER_FORCING}"
echo "Batch/workers/prefetch: ${BATCH_SIZE}/${NUM_WORKERS}/${PREFETCH_FACTOR}"
echo "LR / max_iters        : ${LR} / ${MAX_ITERS}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU inventory] nvidia-smi -L"
  nvidia-smi -L || true
  echo "[GPU status] nvidia-smi"
  nvidia-smi || true
fi

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    print(f"gpu[{idx}]={torch.cuda.get_device_name(idx)}")
PY

PRESSURE_VARS="temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind"
SURFACE_VARS="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,surface_pressure,total_column_water_vapour"
PRESSURE_LEVELS="850,500,250"

run_stage() {
  local stage_name=$1
  local target_offset=$2
  local resume_ckpt=$3
  local seed=$4

  local cmd=(
    "${PYTHON_BIN}" -u -m src.training.train_autoregressive
    --zarr-store "${ZARR_STORE}"
    --train-start 1979-01-01 --train-end 2015-12-31
    --val-start 2016-01-01 --val-end 2017-12-31
    --test-start 2018-01-01 --test-end 2018-12-31
    --history-steps 2
    --pressure-vars "${PRESSURE_VARS}"
    --surface-vars "${SURFACE_VARS}"
    --pressure-levels "${PRESSURE_LEVELS}"
    --preset paper
    --embed-dim 768 --window-size 8 --num-heads 8
    --depth-pre 2 --depth-mid 44 --depth-post 2
    --mlp-ratio 4.0 --drop-path-rate 0.2 --use-checkpoint
    --forecast-steps 20
    --target-offset-steps "${target_offset}"
    --teacher-forcing "${TEACHER_FORCING}"
    --curriculum-start-steps 2 --curriculum-max-steps 20
    --curriculum-step-size 4 --curriculum-epoch-interval 1
    --input-noise-std "${NOISE_STD}"
    --noise-coarse-factor "${NOISE_COARSE_FACTOR}"
    --max-epochs "${MAX_EPOCHS}"
    --max-iters "${MAX_ITERS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --status-every-steps "${STATUS_EVERY_STEPS}"
    --eval-every 1
    --patience "${PATIENCE}"
    --lr "${LR}"
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95
    --grad-clip 1.0
    --fp16
    --gpus "${GPUS_TO_USE}"
    --enable-lora
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-target-modules "${LORA_TARGET_MODULES}"
    --lora-bias none
    --output-root "${RUN_DIR}"
    --exp-name "${stage_name}"
    --resume "${resume_ckpt}"
    --resume-model-only
    --skip-final-test-eval
    --seed "${seed}"
  )

  echo ""
  echo "==> Stage: ${stage_name}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo ""
  "${cmd[@]}"
}

SEED_BASE=$(( 4200 + SLURM_ARRAY_TASK_ID ))

run_stage "stage_short" 0 "${BASE_CKPT}" "${SEED_BASE}"
SHORT_CKPT=$(resolve_stage_ckpt "stage_short")

run_stage "stage_medium" 20 "${SHORT_CKPT}" "$((SEED_BASE + 100))"
MEDIUM_CKPT=$(resolve_stage_ckpt "stage_medium")

cat > "${RUN_DIR}/run_summary.txt" <<EOF
job_id=${SLURM_JOB_ID}
task_id=${SLURM_ARRAY_TASK_ID}
rank=${LORA_RANK}
alpha=${LORA_ALPHA}
dropout=${LORA_DROPOUT}
target_modules=${LORA_TARGET_MODULES}
noise_std=${NOISE_STD}
noise_coarse_factor=${NOISE_COARSE_FACTOR}
teacher_forcing=${TEACHER_FORCING}
mig_mode=${MIG_MODE}
gpus_used=${GPUS_TO_USE}
batch_size=${BATCH_SIZE}
num_workers=${NUM_WORKERS}
prefetch_factor=${PREFETCH_FACTOR}
lr=${LR}
max_iters=${MAX_ITERS}
max_epochs=${MAX_EPOCHS}
patience=${PATIENCE}
final_checkpoint=${MEDIUM_CKPT}
EOF

echo "Done. Final checkpoint: ${MEDIUM_CKPT}"
echo "Logs: ${RUN_DIR}/logs.txt"