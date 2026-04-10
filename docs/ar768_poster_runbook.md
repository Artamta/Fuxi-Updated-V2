# FuXi AR-768 Poster Runbook

This runbook is for the new paper-style-ish 768 cascade run.

## What was set up

- AR trainer now supports:
  - Stage lead-window offsets (`target_offset_steps`: 0, 20, 40)
  - Curriculum rollout schedule (2 -> 12 steps)
  - Faster sequence-window loading for AR targets (reduced per-sample Zarr I/O calls)
  - Dataloader tuning knobs (`prefetch_factor`, persistent workers)
  - Optional smooth Perlin-like input noise (disabled by default in poster configs)
  - Constant learning-rate mode
- New 768 AR stage configs:
  - `configs/cascade_ar768_short.yaml`
  - `configs/cascade_ar768_medium.yaml`
  - `configs/cascade_ar768_long.yaml`
- New AR cascade runner:
  - `scripts/finetune_cascade_ar.sh`
- New priority poster submission script:
  - `scripts/slurm_prio_ar768_poster.sh`

## Fast defaults (now active in scripts)

The Slurm/runner path now applies these runtime defaults automatically:

- `AR_BATCH_SIZE=8`
- `AR_NUM_WORKERS=16`
- `AR_PREFETCH_FACTOR=4`
- `AR_EVAL_EVERY=2`
- `AR_SKIP_FINAL_TEST_EVAL=1`

You can override any of them at submit time:

```bash
sbatch --export=ALL,AR_BATCH_SIZE=8,AR_NUM_WORKERS=20,AR_PREFETCH_FACTOR=4,AR_EVAL_EVERY=2,AR_SKIP_FINAL_TEST_EVAL=1 \
  scripts/slurm_prio_ar768_poster.sh
```

Or use the short helper wrapper:

```bash
bash scripts/submit_ar768_fast.sh
```

If you want the most conservative memory setup:

```bash
sbatch --export=ALL,AR_BATCH_SIZE=4,AR_NUM_WORKERS=12,AR_PREFETCH_FACTOR=3,AR_EVAL_EVERY=2,AR_SKIP_FINAL_TEST_EVAL=1 \
  scripts/slurm_prio_ar768_poster.sh
```

## Current submitted job

- Training Job ID: `7707`
- Name: `fuxi_ar768_poster`
- Partition: `GPU-AI_prio`

- Dependent eval job ID: `7718`
- Name: `fuxi_eval_poster`
- State: waits on `afterok:7707`
- Eval partition default: `gpu_prio` (A30-friendly, 1 GPU)

Important:

- A stage-transfer fix was added (`--resume-model-only`) so medium/long do not inherit global step counters.
- Restart from a fresh job to guarantee all 3 stages use the fixed behavior.

## Monitor

```bash
squeue -u $USER -o "%.18i %.12P %.25j %.8T %.10M %.6D %R"
```

Tail logs:

```bash
tail -f logs/fuxi_ar768_poster_<jobid>.out
```

```bash
tail -f logs/fuxi_ar768_poster_<jobid>.err
```

## Output location

The script writes to:

```bash
results/poster_runs/ar768_cascade_job<jobid>/
```

Expected outputs:

- `stage_short/best.pt`
- `stage_medium/best.pt`
- `stage_long/best.pt`
- `stage_long/evaluation_rollout60/summary.json`
- `summary.txt`

## Restart training manually (if needed)

```bash
scancel 7707
sbatch scripts/slurm_prio_ar768_poster.sh
```

## Poster comparison evaluation (shareable)

Use one command:

```bash
bash scripts/run_poster_eval_compare.sh results/poster_runs/ar768_cascade_job<jobid>
```

Or submit as a dependent Slurm job (recommended):

```bash
sbatch --dependency=afterok:<train_jobid> \
  --export=ALL,RUN_ROOT=results/poster_runs/ar768_cascade_job<train_jobid> \
  scripts/slurm_poster_eval_compare.sh
```

This eval job is intentionally lightweight and defaults to `gpu_prio` with 1 GPU,
so A100 nodes can stay reserved for training.

This generates under `results/poster_runs/ar768_cascade_job<jobid>/poster_eval_compare/`:

- `comparison_summary.csv`
- `comparison_rmse_acc.png`
- `comparison_rmse_<var>.png`
- `comparison_acc_<var>.png`
- `checkpoint_*/spectra/power_spectra_<var>.csv`
- `checkpoint_*/spectra/power_spectra_<var>.png`

If you prefer direct Python usage:

```bash
/home/raj.ayush/.conda/envs/weather_forecast/bin/python -m src.evaluation.evaluate_poster_compare \
  --checkpoints \
    base=results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt \
    short=results/poster_runs/ar768_cascade_job<jobid>/stage_short/best.pt \
    medium=results/poster_runs/ar768_cascade_job<jobid>/stage_medium/best.pt \
    long=results/poster_runs/ar768_cascade_job<jobid>/stage_long/best.pt \
  --output-root results/poster_runs/ar768_cascade_job<jobid>/poster_eval_compare
```

## Notes on paper faithfulness

This setup follows key FuXi ideas:

- Cascade stages for 0-5d, 5-10d, 10-15d windows
- AR training with curriculum schedule
- 768 embed checkpoint initialization

But one part is still approximate:

- Medium/Long stages use time-offset windows from ERA5 directly, not fully cached online outputs from previous stage models.

This tradeoff keeps the pipeline simple and robust for deadline-driven poster runs.

## GPU utilization note

This AR trainer uses `torch.nn.DataParallel`.
If `batch_size` is smaller than number of GPUs, only part of the GPUs will be active.

- Current runtime default uses `AR_BATCH_SIZE=8`, so all 4 A100s are active with better per-GPU workload.
- If you drop batch size to 2, only about 2 GPUs will show active compute.

You can still see low utilization percentages (for example 5-15%) even when all 4 GPUs are active.
This is expected for this setup because:

- The dataset is streamed from Zarr and can be I/O-bound.
- DataParallel adds scatter/gather overhead on GPU0.
- Curriculum starts with short rollout horizon (2 steps), so early-stage compute is lighter.

Additional cause on this project:

- AR samples require multiple future frames per item; poor sequence loading can make training input-bound.
- The updated trainer now loads each sample window in one sequence call to reduce that bottleneck.

How to verify health:

- Same Python PID appears on all 4 GPUs in `nvidia-smi`.
- Job log has no traceback in `logs/fuxi_ar768_poster_<jobid>.err`.
- `squeue` still shows `RUNNING`.

Only restart if the process disappears from one or more GPUs, or if an error traceback appears.

## ETA guide (realistic)

For AR-768 cascade (short+medium+long), with current defaults and healthy queue:

- Per stage: about 40 to 80 minutes
- All 3 stages: about 2.0 to 4.0 hours
- Final rollout eval (`rollout_steps=60`): about 0.5 to 1.5 hours
- End-to-end poster package: about 2.5 to 5.5 hours

These are practical ranges, not hard guarantees. Node load and filesystem throughput can shift runtime.

## Cancel commands

Cancel only poster run:

```bash
scancel <jobid>
```

Cancel old job if you decide to free cn3 resources:

```bash
scancel 7580
```
