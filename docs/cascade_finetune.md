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

This keeps the pipeline readable while letting you tune throughput and stability from one command.
