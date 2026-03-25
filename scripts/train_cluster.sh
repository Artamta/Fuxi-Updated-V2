#!/usr/bin/env bash
set -euo pipefail

# Minimal cluster entrypoint for FuXi training.
# Edit the variables below to point at your data and preferences.

ZARR_STORE=${ZARR_STORE:-"/path/to/weatherbench.zarr"}
EXP_NAME=${EXP_NAME:-"run_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results"}
MAX_EPOCHS=${MAX_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-2.5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}

# Optional config file (YAML/JSON). If set, it is loaded and CLI overrides it.
CONFIG=${CONFIG:-""}

# Model preset: paper (full) or mini (fast/debug).
PRESET=${PRESET:-"paper"}

# Optional overrides. Leave empty to use preset defaults.
EMBED_DIM=${EMBED_DIM:-""}
WINDOW_SIZE=${WINDOW_SIZE:-""}
DROP_PATH=${DROP_PATH:-""}

USE_CHECKPOINT=${USE_CHECKPOINT:-""}       # set to "--use-checkpoint" to enable
FP16=${FP16:-"--fp16"}                      # set to "" to disable
GPUS=${GPUS:-""}                           # e.g., "--gpus 2" for DataParallel
TENSORBOARD=${TENSORBOARD:-""}             # set to "--tensorboard" to enable
RESUME=${RESUME:-""}                       # set to checkpoint path if resuming

# Time ranges (edit to your split)
TRAIN_START=${TRAIN_START:-"1979-01-01"}
TRAIN_END=${TRAIN_END:-"1979-12-31"}
VAL_START=${VAL_START:-"2016-01-01"}
VAL_END=${VAL_END:-"2016-06-30"}
TEST_START=${TEST_START:-"2018-01-01"}
TEST_END=${TEST_END:-"2018-06-30"}

python -u src/training/train.py \
  ${CONFIG:+--config "$CONFIG"} \
  --preset "$PRESET" \
  --zarr-store "$ZARR_STORE" \
  --exp-name "$EXP_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --beta1 "$BETA1" \
  --beta2 "$BETA2" \
  ${EMBED_DIM:+--embed-dim "$EMBED_DIM"} \
  ${WINDOW_SIZE:+--window-size "$WINDOW_SIZE"} \
  ${DROP_PATH:+--drop-path-rate "$DROP_PATH"} \
  $USE_CHECKPOINT \
  $FP16 \
  $GPUS \
  $TENSORBOARD \
  ${RESUME:+--resume "$RESUME"} \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --val-start "$VAL_START" \
  --val-end "$VAL_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END"
