# Priority Job Array Runbook (Easy Version)

This guide gives you two ready jobs:

1. A100 priority array on `GPU-AI_prio`.
2. A30 priority multinode array on `gpu_prio`.

Both jobs save checkpoints and evaluation files in clear folders.

## 1) A100 Priority Array (`GPU-AI_prio`)

Script: `scripts/slurm_array_prio_a100_cascade.sh`

What it tests:
- Array task 0: 2 GPUs (A100 profile)
- Array task 1: 4 GPUs (A100 profile)

What each task does:
1. Tiny pretrain (multi-GPU with Accelerate)
2. Cascade short -> medium -> long
3. Evaluation on the long stage checkpoint
4. Writes a summary file

Submit:
```bash
sbatch scripts/slurm_array_prio_a100_cascade.sh
```

## 2) A30 Priority Multinode Array (`gpu_prio`)

Script: `scripts/slurm_array_prio_a30_multinode.sh`

What it tests:
- Array task 0: auto mode (detect MIG/full and choose safe NCCL)
- Array task 1: force MIG-safe NCCL

What each task does:
1. Allocates 2 nodes (`--nodes=2`, one GPU per node)
2. Logs GPU type per node (MIG or full)
3. Runs distributed NCCL check (`tests/test_multi_node.py`)
4. Runs tiny cascade on primary node
5. Runs evaluation on secondary node
6. Writes a summary file

Submit:
```bash
sbatch scripts/slurm_array_prio_a30_multinode.sh
```

## 3) Monitor Jobs

Show your jobs:
```bash
squeue -u $USER
```

Show one job log:
```bash
tail -f logs/fuxi_a100_arr_<jobid>_0.out
tail -f logs/fuxi_a30_mn_<jobid>_0.out
```

Cancel one job:
```bash
scancel <jobid>
```

Cancel whole array job:
```bash
scancel <array_jobid>
```

## 4) Where Results Are Saved

A100 array results:
- `results/prio_runs/a100/<profile>_job<id>_task<id>/`

A30 multinode array results:
- `results/prio_runs/a30_multinode/<mode>_job<id>_task<id>/`

Inside each run folder you will get:
- pretrain checkpoint
- cascade stage checkpoints
- evaluation plots and CSV
- `summary.txt`

## 5) MIG vs Full (Simple Rule)

- If a node shows MIG devices, the script uses MIG-safe NCCL settings.
- If nodes are full GPUs, it uses normal NCCL settings.
- This check is automatic in A30 array `auto` mode.

## 6) Good Practice for Submission

1. Start with A30 array to verify connectivity and pipeline.
2. Then run A100 array for stronger training quality.
3. Compare `summary.json` in evaluation folders and pick best run.
