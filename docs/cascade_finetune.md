# Cascade fine-tuning (paper-style) – quick guide

This is a pragmatic, config-driven setup to fine-tune three FuXi stages (short/medium/long) starting from a pretrained checkpoint and keep runs organized.

## Files
- Configs: `configs/cascade_short.yaml`, `configs/cascade_medium.yaml`, `configs/cascade_long.yaml`
- Tiny test configs: `configs/cascade_tiny_short.yaml`, `configs/cascade_tiny_medium.yaml`, `configs/cascade_tiny_long.yaml`
- Runner: `scripts/finetune_cascade.sh`
- Tiny submit script: `scripts/slurm_cascade_tiny.sh`
- Default base checkpoint: `/home/raj.ayush/fuxi-final/fuxi_new/results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt`
- Outputs: `results/cascade/<stage>/` (checkpoints, plots, metrics, tensorboard if enabled)

## Run
```bash
# Set your data path (required)
export ZARR_STORE=/path/to/weatherbench.zarr

# Optional: override base pretrained ckpt
export BASE_CKPT=/path/to/pretrain/best.pt

# Optional: override output location and checkpoint filename used for stage chaining
export OUTPUT_ROOT=results/cascade
export CKPT_NAME=best.pt

bash scripts/finetune_cascade.sh
```
Each stage saves `best.pt` in its output directory; the next stage resumes from the previous stage’s best.

## Tiny End-to-End Smoke Submission (single job)
This is the fastest autopilot path to validate a full short->medium->long cascade with a small model.

```bash
# Required: set data path if different from script default
export ZARR_STORE=/path/to/weatherbench.zarr

sbatch scripts/slurm_cascade_tiny.sh
```

What this job does:
- Pretrains a tiny base model checkpoint (`results/pretrain_tiny/tiny_base/best.pt`).
- Runs cascade short/medium/long using tiny configs under `configs/cascade_tiny_*.yaml`.
- Writes stage outputs under `results/cascade_tiny/<stage>/`.

## Adjustments
- Change variables/levels/dates: edit the `data` section in each YAML.
- Change model size or Swin depths/heads: edit the `model` section (embed_dim, num_heads, depth_pre/mid/post, window_size, drop_path_rate).
- Training knobs: `max_epochs`, `lr`, `patience`, `fp16`, etc. live in the `train` section.
- Output location/experiment name: `output_root` and `exp_name` in the `output` section (CLI `--exp-name` overrides exp_name).

## Notes
- Default cascade configs are paper-sized and currently use `embed_dim=1536`; set model dims to match your base checkpoint.
- Fine-tuning here still uses the single-step objective from `train.py` (one-step forecast). For full paper-accurate autoregressive curriculum, we would need to extend the training loop to ramp autoregressive steps and cache intermediate outputs; this setup is the simplified, runnable path.
- TensorBoard can be toggled via `logging.tensorboard` or `--tensorboard` on the CLI.

## AR 768 Poster Pipeline

If you want the new 768 autoregressive cascade path, use:

- Configs: `configs/cascade_ar768_short.yaml`, `configs/cascade_ar768_medium.yaml`, `configs/cascade_ar768_long.yaml`
- Runner: `scripts/finetune_cascade_ar.sh`
- Slurm submit (A100 priority): `scripts/slurm_prio_ar768_poster.sh`
- Runbook: `docs/ar768_poster_runbook.md`

Submit:

```bash
sbatch scripts/slurm_prio_ar768_poster.sh
```

Shortcut wrapper:

```bash
bash scripts/submit_ar768_fast.sh
```

Simple speed tuning (without editing YAMLs):

```bash
sbatch --export=ALL,AR_BATCH_SIZE=8,AR_NUM_WORKERS=16,AR_PREFETCH_FACTOR=4,AR_EVAL_EVERY=2,AR_SKIP_FINAL_TEST_EVAL=1 \
	scripts/slurm_prio_ar768_poster.sh
```

### AR LoRA quick start (parameter-efficient fine-tuning)

The AR trainer now supports optional LoRA adapters. Full fine-tuning remains the default.

To run LoRA across short/medium/long stages with adapter-only training:

```bash
sbatch --export=ALL,AR_ENABLE_LORA=1,AR_LORA_RANK=16,AR_LORA_ALPHA=32,AR_LORA_DROPOUT=0.05,AR_LORA_TARGET_MODULES=qkv,proj,fc1,fc2,AR_LORA_BIAS=none,AR_LORA_TRAIN_BASE=0 \
  scripts/slurm_prio_ar768_poster.sh
```

Recommended first settings for time-constrained runs:

- `AR_LORA_RANK=16` (use `32` if underfitting)
- `AR_LORA_ALPHA=32` (or `64` for rank 32)
- `AR_LORA_DROPOUT=0.05`
- Keep `AR_LORA_TRAIN_BASE=0` for adapter-only updates

You can still run full fine-tuning by leaving `AR_ENABLE_LORA=0` (default).

### Training artifacts (AR trainer)

Each stage directory now includes:

- `epoch_metrics.csv`: per-epoch train/val loss and MAE
- `loss_curve.png`: train/val loss and MAE curves

Example paths:

- `results/.../stage_short/epoch_metrics.csv`
- `results/.../stage_short/loss_curve.png`

LoRA defaults are also declared in AR stage configs under a `finetuning` section:

- `configs/cascade_ar768_short.yaml`
- `configs/cascade_ar768_medium.yaml`
- `configs/cascade_ar768_long.yaml`

This keeps the pipeline readable while letting you tune throughput and stability from one command.

### A30 LoRA matrix sweep (noise x rank)

To submit the default 9-variant A30 sweep (noise `{0.0,0.03,0.08}` x rank `{8,16,32}` on `a30_2gpu`):

```bash
bash scripts/submit_a30_ar768_lora_matrix.sh
```

This creates a submission index under:

- `results/final_runs/a30_lora_matrix_<timestamp>/submissions.csv`

Monitor all submitted jobs with the command printed by the submit script.

Generate an audit snapshot at any time:

```bash
/home/raj.ayush/.conda/envs/weather_forecast/bin/python scripts/audit_a30_ar768_lora_matrix.py \
	--submissions-csv results/final_runs/a30_lora_matrix_<timestamp>/submissions.csv
```

Audit outputs are written to:

- `results/final_runs/a30_lora_matrix_<timestamp>/audit/matrix_summary.csv`
- `results/final_runs/a30_lora_matrix_<timestamp>/audit/matrix_summary.json`
- `results/final_runs/a30_lora_matrix_<timestamp>/audit/matrix_report.md`
