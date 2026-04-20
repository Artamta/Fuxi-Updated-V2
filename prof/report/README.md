# FuXi Finetuning Share Pack

This folder was auto-generated for advisor sharing.

## Included
- LoRA vs original embed-768 comparison plots
- Train/validation loss plots across all detected finetuning runs
- CSV tables backing each plot
- Raw copied logs and metric artifacts for traceability

## Quick Stats
- Finetuning log files parsed: 30
- Train-step points parsed: 950
- Epoch-level points parsed (deduped): 22
- LoRA runs detected: 18
- Embed-768 finetune runs detected: 12
- Best validation loss seen in parsed finetuning: 0.064500

## Evaluation Comparison Source
- Source CSV: results/poster_runs/ar768_cascade_job7608/poster_eval_compare_quick/comparison_summary.csv
- Checkpoints included: base, stage_medium, stage_short

## Key Files
- report/HIGHLIGHTS.md
- data/finetune_step_losses.csv
- data/finetune_epoch_losses.csv
- data/finetune_run_summary.csv
- data/lora_vs_embed768_summary.csv
- data/eval_embed768_compare_all.csv
- plots/finetune_all_train_loss_curves_stage_short.png
- plots/finetune_all_train_loss_curves_stage_medium.png
- plots/finetune_all_val_loss_curves_stage_short.png
- plots/finetune_all_val_loss_curves_stage_medium.png
- plots/lora_vs_embed768_best_val_loss.png
- plots/original_embed768_eval_comparison.png
