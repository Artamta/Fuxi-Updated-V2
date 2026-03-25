#!/usr/bin/env bash
set -euo pipefail

# Simple cascade fine-tuning runner.
# - Uses three stage configs (short/medium/long) under configs/
# - Starts from a base pretrained checkpoint (BASE_CKPT)
# - Writes outputs under results/cascade/<stage>/

ZARR_STORE=${ZARR_STORE:-"/path/to/weatherbench.zarr"}
BASE_CKPT=${BASE_CKPT:-"/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt"}
CONFIG_SHORT=${CONFIG_SHORT:-"configs/cascade_short.yaml"}
CONFIG_MEDIUM=${CONFIG_MEDIUM:-"configs/cascade_medium.yaml"}
CONFIG_LONG=${CONFIG_LONG:-"configs/cascade_long.yaml"}

# Override exp names if desired
EXP_SHORT=${EXP_SHORT:-"stage_short"}
EXP_MEDIUM=${EXP_MEDIUM:-"stage_medium"}
EXP_LONG=${EXP_LONG:-"stage_long"}

# Common overrides
COMMON_OPTS=(--zarr-store "$ZARR_STORE")

run_stage() {
  local cfg=$1
  local exp=$2
  local resume=$3
  echo "\n==> Stage: $exp"
  python -u src/training/train.py \
    --config "$cfg" \
    --exp-name "$exp" \
    --resume "$resume" \
    "${COMMON_OPTS[@]}"
}

run_stage "$CONFIG_SHORT" "$EXP_SHORT" "$BASE_CKPT"
run_stage "$CONFIG_MEDIUM" "$EXP_MEDIUM" "results/cascade/$EXP_SHORT/best.pt"
run_stage "$CONFIG_LONG" "$EXP_LONG" "results/cascade/$EXP_MEDIUM/best.pt"
