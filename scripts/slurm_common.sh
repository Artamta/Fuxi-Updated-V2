# Shared helpers for SLURM launch scripts.
# Keeps defaults in one place and standardizes logging/validation.

set -euo pipefail

# Resolve project root from submit dir or script location.
slurm_root() {
  local submit_dir=${SLURM_SUBMIT_DIR:-}
  if [[ -n "$submit_dir" ]]; then
    echo "$submit_dir"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  fi
}

# Activate conda env if available.
slurm_activate_conda() {
  if [[ -f "/apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
    conda activate weather_forecast || true
  fi
}

# Validate Zarr store path or die.
slurm_require_zarr() {
  local zarr=${1:?}
  if [[ ! -d "$zarr" ]]; then
    echo "ERROR: zarr store not found: $zarr" >&2
    exit 1
  fi
}

# Detect GPU count on node.
slurm_detect_gpus() {
  local ngpus=${SLURM_GPUS_ON_NODE:-}
  if [[ -z "$ngpus" ]]; then
    ngpus=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
    )
  fi
  if [[ "$ngpus" -lt 1 ]]; then
    echo "ERROR: no GPUs visible on this node." >&2
    exit 1
  fi
  echo "$ngpus"
}

# Pick a random open port for torch/accelerate rendezvous.
slurm_pick_port() {
  python - <<'PY'
import random
print(random.randint(10000, 60000))
PY
}

# Standard header for logs.
slurm_log_header() {
  local title=$1; shift
  echo "============================================================"
  echo "$title"
  while [[ $# -gt 0 ]]; do
    echo "$1"; shift
  done
  echo "============================================================"
}

# Export sane NCCL/OMP defaults (can be overridden).
slurm_export_nccl_defaults() {
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8

  export NCCL_DEBUG=WARN
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_TIMEOUT=1800

  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME=^lo,docker0

  export PYTHONFAULTHANDLER=1
}