import xarray as xr
import eval_function

# 1. Load data (WeatherBench 2 format)
fc = xr.open_dataset('model_output.nc')
tr = xr.open_dataset('era5_truth.nc')
cl = xr.open_dataset('era5_climatology.nc')

# 2. One-liner evaluation
results = eval_function.evaluate(
    forecast=fc, 
    truth=tr, 
    climatology=cl, 
    variables=['z500', 't2m', 'u10'], 
    weighted=True  # Set False for simple RMSE/ACC
)

# 3. Plot & Save
eval_function.plot(results, save_path='metrics_plot.png')
eval_function.save(results, 'lab_results')

# 4. Get 15-day mean ACC (if needed)
acc_15d = eval_function.mean_acc_15d(results)
print(acc_15d)