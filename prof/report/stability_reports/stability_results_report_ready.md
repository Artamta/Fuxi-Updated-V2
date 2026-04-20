# Stability Results For Report

## Source
- Main metrics: /home/raj.ayush/fuxi-final/fuxi_new/results_new/checkpoint_emb768/eval_90d/metrics_per_lead.csv

## Horizon Summary
- 60 days (240 steps): ACC at horizon = 0.1178, RMSE at horizon = 396.408, last-7d ACC mean = 0.1378, ACC slope last-7d (per day) = -0.003923
- 90 days (360 steps): ACC at horizon = -0.0447, RMSE at horizon = 517.542, last-7d ACC mean = -0.0354, ACC slope last-7d (per day) = -0.005728
- Step 360 checkpoint: ACC = -0.0447, RMSE = 517.542

## Notes
- Step 360 corresponds to 90 days at 6-hour lead spacing.
- 60-day all-variable mean ACC remains positive but decreasing.
- 90-day all-variable mean ACC becomes slightly negative near tail, indicating near-climatology skill at very long horizon.
