#!/usr/bin/env bash
set -euo pipefail

# Submit a fixed 3x3 LoRA matrix on A30 for AR768 cascade training.
# Default grid: noise={0.0,0.03,0.08} x rank={8,16,32}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"
mkdir -p logs

NOISE_VALUES_CSV=${NOISE_VALUES_CSV:-"0.0,0.03,0.08"}
RANK_VALUES_CSV=${RANK_VALUES_CSV:-"8,16,32"}
PROFILE_ARRAY_INDEX=${PROFILE_ARRAY_INDEX:-1}
NOISE_COARSE_FACTOR=${NOISE_COARSE_FACTOR:-16}
STATUS_EVERY_STEPS=${STATUS_EVERY_STEPS:-200}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-none}
LORA_TRAIN_BASE=${LORA_TRAIN_BASE:-0}
LORA_ALPHA_OVERRIDE=${LORA_ALPHA_OVERRIDE:-}

# Keep this exported in the environment to avoid SLURM --export comma parsing issues.
export AR_LORA_TARGET_MODULES=${AR_LORA_TARGET_MODULES:-qkv,proj,fc1,fc2}

RUN_TAG=${RUN_TAG:-"a30_lora_matrix_$(date +%Y%m%d_%H%M%S)"}
MATRIX_ROOT=${MATRIX_ROOT:-"results/final_runs/${RUN_TAG}"}
SUBMISSION_CSV="${MATRIX_ROOT}/submissions.csv"

case "${PROFILE_ARRAY_INDEX}" in
  0) PROFILE_NAME_DEFAULT="a30_1gpu" ;;
  1) PROFILE_NAME_DEFAULT="a30_2gpu" ;;
  *) PROFILE_NAME_DEFAULT="profile${PROFILE_ARRAY_INDEX}" ;;
esac
PROFILE_NAME=${PROFILE_NAME:-"${PROFILE_NAME_DEFAULT}"}

mkdir -p "${MATRIX_ROOT}" "${MATRIX_ROOT}/variants"

IFS=',' read -r -a RAW_NOISE_VALUES <<< "${NOISE_VALUES_CSV}"
IFS=',' read -r -a RAW_RANK_VALUES <<< "${RANK_VALUES_CSV}"

NOISE_VALUES=()
for noise_raw in "${RAW_NOISE_VALUES[@]}"; do
  noise=$(echo "${noise_raw}" | tr -d '[:space:]')
  if [[ -n "${noise}" ]]; then
    NOISE_VALUES+=("${noise}")
  fi
done

RANK_VALUES=()
for rank_raw in "${RAW_RANK_VALUES[@]}"; do
  rank=$(echo "${rank_raw}" | tr -d '[:space:]')
  if [[ -n "${rank}" ]]; then
    RANK_VALUES+=("${rank}")
  fi
done

if (( ${#NOISE_VALUES[@]} == 0 )); then
  echo "ERROR: parsed NOISE_VALUES is empty" >&2
  exit 1
fi
if (( ${#RANK_VALUES[@]} == 0 )); then
  echo "ERROR: parsed RANK_VALUES is empty" >&2
  exit 1
fi

printf '%s\n' "submitted_at,variant,noise_std,lora_rank,lora_alpha,job_id,array_task_id,profile,run_root_base,expected_output_root,slurm_out_log,slurm_err_log" > "${SUBMISSION_CSV}"

JOB_IDS=()
SUBMITTED=0

for noise in "${NOISE_VALUES[@]}"; do
  for rank in "${RANK_VALUES[@]}"; do
    if ! [[ "${rank}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: non-integer rank value: ${rank}" >&2
      exit 1
    fi

    if [[ -n "${LORA_ALPHA_OVERRIDE}" ]]; then
      lora_alpha="${LORA_ALPHA_OVERRIDE}"
    else
      lora_alpha=$(( rank * 2 ))
    fi

    noise_tag=${noise//./p}
    variant="noise${noise_tag}_rank${rank}"
    run_root_base="${MATRIX_ROOT}/variants/${variant}"

    sbatch_out=$(sbatch \
      --array="${PROFILE_ARRAY_INDEX}-${PROFILE_ARRAY_INDEX}" \
      --export="ALL,RUN_ROOT_BASE=${run_root_base},AR_ENABLE_LORA=1,AR_LORA_RANK=${rank},AR_LORA_ALPHA=${lora_alpha},AR_LORA_DROPOUT=${LORA_DROPOUT},AR_LORA_BIAS=${LORA_BIAS},AR_LORA_TRAIN_BASE=${LORA_TRAIN_BASE},AR_INPUT_NOISE_STD=${noise},AR_NOISE_COARSE_FACTOR=${NOISE_COARSE_FACTOR},AR_STATUS_EVERY_STEPS=${STATUS_EVERY_STEPS}" \
      scripts/slurm_array_a30_ar768_full.sh)

    job_id=$(awk '/Submitted batch job/{print $4}' <<< "${sbatch_out}")
    if [[ -z "${job_id}" ]]; then
      echo "ERROR: failed to parse sbatch output for variant=${variant}" >&2
      echo "sbatch output: ${sbatch_out}" >&2
      exit 1
    fi

    expected_output_root="${run_root_base}/ar768_a30_array_job${job_id}_task${PROFILE_ARRAY_INDEX}_${PROFILE_NAME}"
    slurm_out_log="logs/fuxi_a30_ar768_arr_${job_id}_${PROFILE_ARRAY_INDEX}.out"
    slurm_err_log="logs/fuxi_a30_ar768_arr_${job_id}_${PROFILE_ARRAY_INDEX}.err"

    submitted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\n' "${submitted_at},${variant},${noise},${rank},${lora_alpha},${job_id},${PROFILE_ARRAY_INDEX},${PROFILE_NAME},${run_root_base},${expected_output_root},${slurm_out_log},${slurm_err_log}" >> "${SUBMISSION_CSV}"

    JOB_IDS+=("${job_id}")
    SUBMITTED=$((SUBMITTED + 1))
    echo "[submitted] ${variant}: job_id=${job_id}, expected_output_root=${expected_output_root}"
  done
done

job_ids_csv=$(IFS=,; echo "${JOB_IDS[*]}")

cat > "${MATRIX_ROOT}/README.txt" <<EOF
run_tag=${RUN_TAG}
submitted_jobs=${SUBMITTED}
profile_array_index=${PROFILE_ARRAY_INDEX}
profile_name=${PROFILE_NAME}
noise_values=${NOISE_VALUES_CSV}
rank_values=${RANK_VALUES_CSV}
submission_csv=${SUBMISSION_CSV}
job_ids=${job_ids_csv}
monitor_squeue=squeue -j ${job_ids_csv} -o "%.18i %.8T %.10M %.20R"
monitor_sacct=sacct -j ${job_ids_csv} --format=JobID,State,Elapsed,ExitCode -P
EOF

echo ""
echo "Submitted ${SUBMITTED} jobs."
echo "Submission index: ${SUBMISSION_CSV}"
echo "Quick monitor: squeue -j ${job_ids_csv} -o '%.18i %.8T %.10M %.20R'"
echo "Final states : sacct -j ${job_ids_csv} --format=JobID,State,Elapsed,ExitCode -P"
