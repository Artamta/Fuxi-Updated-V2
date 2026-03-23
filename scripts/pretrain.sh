#!/usr/bin/env bash
set -euo pipefail

# Local pretraining launcher for the modular src/ layout.
# Usage examples:
#   bash scripts/pretrain.sh
#   ZARR_STORE=/path/to/data.zarr BATCH_SIZE=2 bash scripts/pretrain.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh" ]]; then
	source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
	conda activate weather_forecast || true
fi

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/for_model_development/weatherbench2/era5/1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr}"

if [[ ! -d "${ZARR_STORE}" ]]; then
	echo "ERROR: zarr store not found: ${ZARR_STORE}"
	exit 1
fi

EXP_NAME="${EXP_NAME:-pretrain_$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="${RUNS_DIR:-Models_paper/pretrain}"

python -m src.pretraining.pretrain \
	--zarr-store "${ZARR_STORE}" \
	--train-start "${TRAIN_START:-1979-01-01}" \
	--train-end "${TRAIN_END:-2018-12-31}" \
	--val-start "${VAL_START:-2019-01-01}" \
	--val-end "${VAL_END:-2020-12-31}" \
	--test-start "${TEST_START:-2021-01-01}" \
	--test-end "${TEST_END:-2022-12-31}" \
	--exp-name "${EXP_NAME}" \
	--runs-dir "${RUNS_DIR}" \
	--embed-dim "${EMBED_DIM:-768}" \
	--num-heads "${NUM_HEADS:-8}" \
	--window-size "${WINDOW_SIZE:-8}" \
	--depth-pre "${DEPTH_PRE:-2}" \
	--depth-mid "${DEPTH_MID:-24}" \
	--depth-post "${DEPTH_POST:-2}" \
	--drop-path-rate "${DROP_PATH:-0.1}" \
	--use-checkpoint \
	--batch-size "${BATCH_SIZE:-1}" \
	--accum-steps "${ACCUM_STEPS:-1}" \
	--max-epochs "${MAX_EPOCHS:-1}" \
	--max-iters "${MAX_ITERS:-100}" \
	--patience "${PATIENCE:-2}" \
	--num-workers "${NUM_WORKERS:-2}" \
	--lr "${LR:-2.5e-4}" \
	--weight-decay "${WEIGHT_DECAY:-0.1}" \
	--beta1 "${BETA1:-0.9}" \
	--beta2 "${BETA2:-0.95}" \
	--grad-clip "${GRAD_CLIP:-1.0}" \
	--loss "${LOSS:-l1}" \
	--device "${DEVICE:-auto}" \
	--amp "${AMP:-bf16}"
