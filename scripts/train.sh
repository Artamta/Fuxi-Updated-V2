#!/usr/bin/env bash
set -euo pipefail

# Training wrapper for the legacy-compatible training entrypoint.
# You can switch mode with MODE=pretrain|finetune.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh" ]]; then
	source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
	conda activate weather_forecast || true
fi

ZARR_STORE="${ZARR_STORE:-/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"

python -m src.training.train \
	--mode "${MODE:-pretrain}" \
	--stage "${STAGE:-short}" \
	--zarr-store "${ZARR_STORE}" \
	--train-start "${TRAIN_START:-1979-01-01}" \
	--train-end "${TRAIN_END:-1979-01-10}" \
	--val-start "${VAL_START:-2016-01-01}" \
	--val-end "${VAL_END:-2016-01-03}" \
	--test-start "${TEST_START:-2018-01-01}" \
	--test-end "${TEST_END:-2018-01-03}" \
	--history-steps "${HISTORY_STEPS:-2}" \
	--max-epochs "${MAX_EPOCHS:-1}" \
	--max-iters "${MAX_ITERS:-100}" \
	--batch-size "${BATCH_SIZE:-1}" \
	--num-workers "${NUM_WORKERS:-2}" \
	--patience "${PATIENCE:-2}" \
	--lr "${LR:-2.5e-4}" \
	--weight-decay "${WEIGHT_DECAY:-0.1}" \
	--beta1 "${BETA1:-0.9}" \
	--beta2 "${BETA2:-0.95}" \
	--embed-dim "${EMBED_DIM:-256}" \
	--window-size "${WINDOW_SIZE:-8}" \
	--drop-path-rate "${DROP_PATH:-0.1}" \
	--use-checkpoint \
	--output-root "${OUTPUT_ROOT:-Models}" \
	--exp-name "${EXP_NAME:-paper_zarr_smoke}" \
	--seed "${SEED:-42}"
