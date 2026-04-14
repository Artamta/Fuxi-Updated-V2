# Shared benchmark (forecast-only)

This folder is a simple, shared evaluation pipeline for weather-model teams.

The workflow is forecast-only:

- each team runs its own model however it wants
- each team exports one forecast NetCDF file
- this benchmark script evaluates all files with the same metric code

No model training or checkpoint loading is done in this folder.

## Who this is for

Anyone who wants to compare model quality on the same data split:

- FuXi users
- Pangu users
- GraphCast users
- any custom weather model users

As long as the exported forecast file format matches the schema below, the model type does not matter.

## Required forecast file format

Each model must provide one NetCDF with:

- data variable `forecast`
- data variable `truth`
- coordinate `channel_name`
- coordinate `lead_step`
- coordinate `lat`
- coordinate `lon`

Required dimension order for both `forecast` and `truth`:

- `init_time`
- `lead_step`
- `channel`
- `lat`
- `lon`

Expected shape:

- `(n_init, n_lead, n_channel, n_lat, n_lon)`

## Metric formulas used in this benchmark

For each channel and lead time, the evaluation uses latitude weights and averages over initialization times.

Latitude weight:

$$
a_i = \frac{\cos(\phi_i)}{\mathrm{mean}_i[\cos(\phi_i)]}
$$

Weighted RMSE:

$$
\mathrm{RMSE}(c,\tau) = \frac{1}{|D|} \sum_{t_0 \in D}
\sqrt{\frac{1}{HW} \sum_{i,j} a_i\,\left(\hat{X}_{c,\tau,t_0}(i,j)-X_{c,\tau,t_0}(i,j)\right)^2}
$$

Unweighted RMSE (also exported):

$$
\mathrm{RMSE}_{\text{unweighted}}(c,\tau) = \frac{1}{|D|} \sum_{t_0 \in D}
\sqrt{\frac{1}{HW} \sum_{i,j}\left(\hat{X}_{c,\tau,t_0}(i,j)-X_{c,\tau,t_0}(i,j)\right)^2}
$$

ACC (anomaly correlation):

$$
\mathrm{ACC}(c,\tau) = \frac{1}{|D|}\sum_{t_0 \in D}
\frac{\sum_{i,j} a_i\,(\hat{X}-M)(X-M)}
{\sqrt{\left(\sum_{i,j} a_i\,(\hat{X}-M)^2\right)\left(\sum_{i,j} a_i\,(X-M)^2\right)}}
$$

where $M$ is climatology selected by valid time (day-of-year, hour).

Optional L1 (if enabled):

$$
\mathrm{L1}(\tau)=\mathrm{mean}_{\text{init,channel,lat,lon}}\left|\hat{X}-X\right|
$$

## How model comparison is ranked

The script writes `comparison/model_summary.csv` and sorts models by:

1. lower `mean_rmse` (latitude-weighted) is better
2. if two models have very similar RMSE, higher `mean_acc` is better

It also writes `comparison/horizon_comparison.csv` using your `horizon_days` windows (for example 5-day, 10-day, 15-day).

## Example plots

These example images are generated from real benchmark runs in this repo:

- [Comparison figure](../../docs/shared_benchmark_examples/model_comparison.png)
- [RMSE per-variable panels (all 20 variables)](../../docs/shared_benchmark_examples/poster_all20_rmse_per_variable_emb768.png)
- [ACC per-variable panels (all 20 variables)](../../docs/shared_benchmark_examples/poster_all20_acc_per_variable_emb768.png)

Note:

- `poster_all20_rmse_overlay_emb768.png` is no longer produced.
- The overlay version was removed because it was hard to read for poster use.

## Expected results after running benchmark

After one benchmark run, you should expect these files.

Inside `results_root/checkpoint_<model_name>/metrics/` (per model):

- `summary.json`
- `metrics_per_lead.csv`
- `mean_metrics_per_lead.csv`
- `horizon_window_summary.csv`
- `poster_all20_rmse_per_variable.png`
- `poster_all20_acc_per_variable.png`
- optional `l1_per_lead.csv` when L1 is enabled

Inside `results_root/comparison/` (shared across all models):

- `model_comparison.png`
- `model_summary.csv`
- `horizon_comparison.csv`

Inside `docs/shared_benchmark_examples/` (repo examples for GitHub README):

- `model_comparison.png`
- `poster_all20_rmse_per_variable_emb768.png`
- `poster_all20_acc_per_variable_emb768.png`

## Step-by-step usage

1. Ask each team to export forecast files in the required schema.
2. Copy and edit [src/shared_benchmark/example_config.json](src/shared_benchmark/example_config.json).
3. Add one entry per model under `models`:
   - `name`
   - `forecast_file`
4. Run benchmark:

```bash
python -m src.shared_benchmark.run_benchmark --config src/shared_benchmark/example_config.json
```

Optional CLI switches:

```bash
python -m src.shared_benchmark.run_benchmark \
  --config src/shared_benchmark/example_config.json \
  --compute-l1
```

Use `--no-compute-l1` to force-disable L1 from CLI.

## Config keys

- `results_root`: output folder
- `horizon_days`: horizon windows, for example `[5, 10, 15]`
- `compute_l1`: true/false
- `strict_consistency`: true/false
- `eval_no_heatmaps`: true/false
- `overwrite_eval`: true/false
- `climatology_store`: optional ACC climatology path override
- `models`: list of model entries

Each model entry:

- `name`: model label in plots/tables
- `forecast_file`: path to the model forecast NetCDF

## What gets generated

Per model (inside `results_root/checkpoint_<name>/metrics/`):

- RMSE/ACC CSVs and plots
- optional `l1_per_lead.csv` if L1 is enabled

Shared outputs (inside `results_root/comparison/`):

- `model_comparison.png`
- `model_summary.csv`
- `horizon_comparison.csv`

## Common errors

`Forecast file not found`

- check path in config
- relative paths are resolved from repo root

`Consistency check failed`

- channel list/order, lead steps, or grid shape differ across models
- fix exported files or set `strict_consistency: false` only when mismatch is intentional

`Missing variable/coordinate`

- required fields are missing in NetCDF

`Dimension order error`

- make sure both `forecast` and `truth` are exactly `(init_time, lead_step, channel, lat, lon)`

## Minimal exporter checklist for each team

Before sharing a forecast file, each team should check:

1. predictions stored in `forecast`
2. targets stored in `truth`
3. channel names in `channel_name`
4. lead indices in `lead_step`
5. coordinates in `lat`, `lon`
6. correct dim order for both arrays
