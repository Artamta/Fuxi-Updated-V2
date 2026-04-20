# Report Asset Pipeline (Embed-768)

This runbook generates the new report deliverables:
- 6 GIFs total (3 t2m horizons + 3 richer-variable horizons)
- explicit init date/time in GIF titles
- ACC overlays from metrics CSV
- 90-day ACC drift plots and threshold table
- all-variable surface/upper-air static plots from existing evaluator

## Files Added For This Pipeline
- prof/report/generate_forecast_gifs.py
- prof/report/plot_long_horizon_acc_drift.py
- src/checkpoint_study/forecast_to_nc.py
	- new flags: --init-random-sampling, --init-seed

## Recommended Environment
- Use weather_forecast conda env on A100 node.
- Use bf16 mixed precision for inference/evaluation.

## 1) Generate 15-day Forecast NetCDF (3 Random Init Times)

```bash
python -m src.checkpoint_study.forecast_to_nc \
	--checkpoints emb768=results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt \
	--rollout-steps 60 \
	--max-inits 3 \
	--init-random-sampling \
	--init-seed 42 \
	--batch-size 1 \
	--num-workers 4 \
	--device cuda \
	--amp bf16 \
	--overwrite
```

Main output:
- results_new/checkpoint_emb768/forecast/forecast.nc
- results_new/checkpoint_emb768/forecast/forecast_metadata.json

The metadata now includes random sampling settings and init times used.

## 2) Evaluate Forecast (RMSE/ACC + All-Variable Static Plots)

```bash
python -m src.checkpoint_study.evaluate_forecast_nc \
	--forecast-files emb768=results_new/checkpoint_emb768/forecast/forecast.nc \
	--horizon-days 5,10,15 \
	--overwrite
```

Main outputs:
- results_new/checkpoint_emb768/metrics/metrics_per_lead.csv
- results_new/checkpoint_emb768/metrics/metrics_per_day.csv
- results_new/checkpoint_emb768/metrics/surface_variable_metrics.png
- results_new/checkpoint_emb768/metrics/pressure_15_variable_metrics.png
- results_new/checkpoint_emb768/metrics/average_rmse_acc_by_group.png
- results_new/checkpoint_emb768/metrics/poster_all20_rmse_per_variable.png
- results_new/checkpoint_emb768/metrics/poster_all20_acc_per_variable.png

## 3) Build The 6 Report GIFs

Default richer-variable set:
- geopotential_plev500
- total_column_water_vapour
- u_component_of_wind_plev850

```bash
python prof/report/generate_forecast_gifs.py \
	--forecast-file results_new/checkpoint_emb768/forecast/forecast.nc \
	--metrics-file results_new/checkpoint_emb768/metrics/metrics_per_lead.csv \
	--horizon-days 5,10,15 \
	--horizon-init-indices 0,1,2 \
	--fps 3 \
	--frame-step 1 \
	--output-dir prof/report/gifs
```

Expected outputs:
- prof/report/gifs/t2m_rollout_05d_*.gif
- prof/report/gifs/t2m_rollout_10d_*.gif
- prof/report/gifs/t2m_rollout_15d_*.gif
- prof/report/gifs/rich_rollout_05d_*.gif
- prof/report/gifs/rich_rollout_10d_*.gif
- prof/report/gifs/rich_rollout_15d_*.gif
- prof/report/gifs/gif_manifest.json

## 3b) Optional: One 15-Day All-Variable GIF (Paged)

This mode keeps one GIF file while paging through all variables per lead time.

```bash
python prof/report/generate_forecast_gifs.py \
	--forecast-file results_new/checkpoint_emb768/forecast/forecast.nc \
	--metrics-file results_new/checkpoint_emb768/metrics/metrics_per_lead.csv \
	--horizon-days 15 \
	--horizon-init-indices 2 \
	--make-all-vars-gif \
	--all-vars-per-page 6 \
	--fps 3 \
	--frame-step 1 \
	--output-dir prof/report/gifs_allvars_15d
```

Expected additional output:
- prof/report/gifs_allvars_15d/allvars_rollout_15d_*.gif
- prof/report/gifs_allvars_15d/gif_manifest.json

## 4) Run 90-Day Autoregressive Evaluation

```bash
python -m src.evaluation.evaluate_checkpoint \
	--checkpoint results/checkpoints_archive/fuxi_paper_prev/Models_paper/pretrain/emb_768/best.pt \
	--rollout-steps 360 \
	--batch-size 1 \
	--num-workers 4 \
	--max-samples 64 \
	--stats-samples 256 \
	--device cuda \
	--amp bf16 \
	--output-dir results_new/checkpoint_emb768/eval_90d
```

Main metric file produced by this step:
- results_new/checkpoint_emb768/eval_90d/metrics_per_lead.csv

## 5) Plot Long-Horizon ACC Drift + Thresholds

```bash
python prof/report/plot_long_horizon_acc_drift.py \
	--metrics-file results_new/checkpoint_emb768/eval_90d/metrics_per_lead.csv \
	--key-vars 2m_temperature,geopotential_plev500,total_column_water_vapour,u_component_of_wind_plev850 \
	--thresholds 0.8,0.6,0.4 \
	--smoothing-window-steps 1 \
	--output-dir prof/report/plots
```

Expected outputs:
- prof/report/plots/acc_drift_overview.png
- prof/report/plots/acc_drift_key_variables.png
- prof/report/plots/acc_threshold_summary.csv
- prof/report/plots/acc_drift_summary.md

## Submission Checklist
- 6 GIF files exist and open correctly.
- gif_manifest.json includes horizon and init timestamps.
- 90-day ACC plots generated.
- acc_threshold_summary.csv generated.
- surface and upper-air static metric plots generated.
