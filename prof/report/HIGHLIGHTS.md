# Highlights For Prof

## Best LoRA Runs By Validation Loss
- 1. job_8211_task_5_r32_a64_n0p02_c8_t5 (stage_medium): val_loss=0.104356
- 2. job_8152_task_6_r32_a64_n0p02_c16_t6 (stage_medium): val_loss=0.105530
- 3. job_8186_task_3_r16_a32_n0p02_c8_t3 (stage_medium): val_loss=0.114689
- 4. job_8206_task_4_r16_a32_n0p02_c16_t4 (stage_medium): val_loss=0.115081
- 5. job_8211_task_5_r32_a64_n0p02_c8_t5 (stage_short): val_loss=0.115289

## Best Embed-768 Finetune Run
- ar768_cascade_job7707 (stage_short): val_loss=0.064500

## Eval Delta Vs Original Embed-768 Base
- Base RMSE=67.172675, Base ACC=0.879443
- stage_medium: ΔRMSE=+0.214267, ΔACC=+0.008104
- stage_short: ΔRMSE=+0.217874, ΔACC=+0.008093

## Data Files
- data/top_lora_runs.csv
- data/eval_deltas_vs_base.csv
