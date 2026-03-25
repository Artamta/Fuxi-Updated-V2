# Archive

Older or alternate files moved out of the main pipeline to avoid confusion.

## Legacy training
- `archive/legacy_training/train_solid_prev.py`: older training entrypoint kept for reference; not used by the current pretraining/finetuning pipeline.
- `archive/legacy_training/finetune_empty.py`: placeholder from old finetuning path (empty).

## Legacy configs
- `archive/legacy_configs/finetuning.yaml`: empty stub moved out of configs.

## Current main pipeline (kept in place)
- Pretraining: `src/pretraining/pretrain.py`
- Training (Zarr single-step): `src/training/train.py`
- Dataset wrapper: `src/training/fuxi_train.py`
- Model: `src/models/fuxi_model.py` (+ presets) and `src/models/u_tranformer.py`
- Cluster runner: `scripts/train_cluster.sh`
- Config examples: `configs/*.yaml`

If you need to restore a file, move it back from `archive/` to its original path.
