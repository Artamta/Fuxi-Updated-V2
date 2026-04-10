#!/usr/bin/env bash
set -euo pipefail

# Human-friendly launcher for the AR-768 poster cascade.
# Use defaults as-is, or override any variable before running this script.
# Example:
#   AR_BATCH_SIZE=6 AR_NUM_WORKERS=12 bash scripts/submit_ar768_fast.sh

AR_BATCH_SIZE=${AR_BATCH_SIZE:-12}
AR_NUM_WORKERS=${AR_NUM_WORKERS:-8}
AR_PREFETCH_FACTOR=${AR_PREFETCH_FACTOR:-3}
AR_EVAL_EVERY=${AR_EVAL_EVERY:-2}
AR_SKIP_FINAL_TEST_EVAL=${AR_SKIP_FINAL_TEST_EVAL:-1}

echo "Submitting AR-768 poster run with:"
echo "  AR_BATCH_SIZE=${AR_BATCH_SIZE}"
echo "  AR_NUM_WORKERS=${AR_NUM_WORKERS}"
echo "  AR_PREFETCH_FACTOR=${AR_PREFETCH_FACTOR}"
echo "  AR_EVAL_EVERY=${AR_EVAL_EVERY}"
echo "  AR_SKIP_FINAL_TEST_EVAL=${AR_SKIP_FINAL_TEST_EVAL}"

sbatch --export=ALL,AR_BATCH_SIZE=${AR_BATCH_SIZE},AR_NUM_WORKERS=${AR_NUM_WORKERS},AR_PREFETCH_FACTOR=${AR_PREFETCH_FACTOR},AR_EVAL_EVERY=${AR_EVAL_EVERY},AR_SKIP_FINAL_TEST_EVAL=${AR_SKIP_FINAL_TEST_EVAL} \
  scripts/slurm_prio_ar768_poster.sh
