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
