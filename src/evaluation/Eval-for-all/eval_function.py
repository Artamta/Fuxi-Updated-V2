"""
eval_function.py
Standardized RMSE & ACC evaluation for AI weather models (FuXi, GraphCast, Pangu, etc.)
Follows Eq. 3 & 4 from standard weather bench evaluation papers.

Usage:
    import eval_function
    
    results = eval_function.evaluate(
        forecast=forecast_ds,
        truth=truth_ds,
        climatology=clim_ds,
        variables=['z500', 't2m', 'u10'],
        weighted=True  # Set False for unweighted
    )
    
    eval_function.plot(results)
    eval_function.save(results, 'metrics.csv')
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
import warnings

# ---------------------------------------------------------------------------
# CORE METRICS (Implements Eq. 3 & 4 exactly)
# ---------------------------------------------------------------------------

def _get_lead_time_dim(ds: xr.Dataset) -> str:
    """Auto-detect lead time dimension name."""
    for dim in ['lead_time', 'step', 'prediction_timedelta']:
        if dim in ds.dims:
            return dim
    raise ValueError("Could not find lead time dimension. Expected 'lead_time' or 'step'.")

def _compute_lat_weights(lat: xr.DataArray) -> xr.DataArray:
    """Compute normalized cosine latitude weights."""
    lat_rad = np.deg2rad(lat)
    weights = np.cos(lat_rad)
    # Normalize so global mean equals arithmetic mean
    return weights / weights.mean()

def _spatial_rmse(fc: xr.DataArray, tr: xr.DataArray, weights: Optional[xr.DataArray]) -> xr.DataArray:
    """Compute spatial RMSE for a single time step (inner sqrt of Eq. 3)."""
    err_sq = (fc - tr) ** 2
    if weights is not None:
        # Weighted spatial mean
        mse = (err_sq * weights).sum(dim=['lat', 'lon']) / weights.sum()
    else:
        mse = err_sq.mean(dim=['lat', 'lon'])
    return np.sqrt(mse)

def _spatial_acc(fc: xr.DataArray, tr: xr.DataArray, cl: xr.DataArray, weights: Optional[xr.DataArray]) -> xr.DataArray:
    """Compute spatial ACC for a single time step (inner fraction of Eq. 4)."""
    fc_anom = fc - cl
    tr_anom = tr - cl
    
    if weights is not None:
        cov = (fc_anom * tr_anom * weights).sum(dim=['lat', 'lon']) / weights.sum()
        var_fc = ((fc_anom ** 2) * weights).sum(dim=['lat', 'lon']) / weights.sum()
        var_tr = ((tr_anom ** 2) * weights).sum(dim=['lat', 'lon']) / weights.sum()
    else:
        cov = (fc_anom * tr_anom).mean(dim=['lat', 'lon'])
        var_fc = (fc_anom ** 2).mean(dim=['lat', 'lon'])
        var_tr = (tr_anom ** 2).mean(dim=['lat', 'lon'])
        
    acc = cov / (np.sqrt(var_fc) * np.sqrt(var_tr))
    return np.clip(acc, -1.0, 1.0)

# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    climatology: xr.Dataset,
    variables: List[str],
    weighted: bool = True,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compute latitude-weighted (or unweighted) RMSE and ACC per lead time.
    Matches Eq. 3 & 4: spatial metric per initialization time, then averaged over time.
    
    Parameters
    ----------
    forecast : xr.Dataset
        Model forecast data with dimensions (time, lead_time/step, lat, lon, [level])
    truth : xr.Dataset
        ERA5/analysis truth data aligned with forecast
    climatology : xr.Dataset
        Climatological mean (same structure as truth)
    variables : list of str
        Variables to evaluate, e.g., ['z500', 't2m', 'u10']
    weighted : bool, default True
        Apply cosine(latitude) weighting
    verbose : bool, default True
        Print progress
        
    Returns
    -------
    dict
        Nested dictionary: {var: DataFrame with columns ['lead_time_hours', 'rmse', 'acc']}
    """
    results = {}
    lead_dim = _get_lead_time_dim(forecast)
    lat_weights = _compute_lat_weights(forecast.lat) if weighted else None
    
    for var in variables:
        if verbose:
            print(f"⏳ Evaluating {var}...")
            
        # Extract and align
        fc = forecast[var]
        tr = truth[var]
        cl = climatology[var]
        
        # Broadcast climatology to match forecast time/lead structure if needed
        if 'time' in fc.dims and 'time' not in cl.dims:
            cl = cl.expand_dims(time=fc.time)
        if lead_dim not in cl.dims and lead_dim in fc.dims:
            cl = cl.expand_dims({lead_dim: fc.sizes[lead_dim]})
            
        fc, tr, cl = xr.align(fc, tr, cl, join='inner')
        
        rmse_list, acc_list, lt_hours = [], [], []
        
        # Loop over lead times (Eq 3 & 4 structure: compute per step, average over time)
        for lt in fc[lead_dim].values:
            fc_lt = fc.sel({lead_dim: lt})
            tr_lt = tr.sel({lead_dim: lt})
            cl_lt = cl.sel({lead_dim: lt}, method='nearest')
            
            # Compute spatial metrics for each initialization time
            rmse_per_time = _spatial_rmse(fc_lt, tr_lt, lat_weights)
            acc_per_time  = _spatial_acc(fc_lt, tr_lt, cl_lt, lat_weights)
            
            # Average over initialization times (1/|D| Σ ...)
            rmse_mean = rmse_per_time.mean(dim='time').values
            acc_mean  = acc_per_time.mean(dim='time').values
            
            # Convert lead time to hours
            if np.issubdtype(lt.dtype, np.timedelta64):
                hours = lt / np.timedelta64(1, 'h')
            else:
                hours = float(lt)
                
            rmse_list.append(rmse_mean)
            acc_list.append(acc_mean)
            lt_hours.append(hours)
            
        results[var] = pd.DataFrame({
            'lead_time_hours': lt_hours,
            'rmse': rmse_list,
            'acc': acc_list
        })
        
    if verbose:
        print("✅ Evaluation complete.")
    return results

# ---------------------------------------------------------------------------
# PLOTTING & SAVING UTILITIES
# ---------------------------------------------------------------------------

def plot(results: Dict[str, pd.DataFrame], 
         variables: Optional[List[str]] = None,
         save_path: Optional[str] = None,
         figsize: tuple = (14, 8)):
    """Plot RMSE and ACC vs lead time for all variables."""
    vars_to_plot = variables or list(results.keys())
    n_vars = len(vars_to_plot)
    
    fig, axes = plt.subplots(2, n_vars, figsize=figsize, constrained_layout=True)
    if n_vars == 1:
        axes = axes.reshape(-1, 1)
        
    for i, var in enumerate(vars_to_plot):
        df = results[var]
        
        # RMSE
        axes[0, i].plot(df['lead_time_hours'], df['rmse'], 'b-o', linewidth=2, markersize=5)
        axes[0, i].set_xlabel('Lead Time (hours)')
        axes[0, i].set_ylabel('RMSE')
        axes[0, i].set_title(f'{var.upper()} RMSE')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(0, df['lead_time_hours'].max())
        
        # ACC
        axes[1, i].plot(df['lead_time_hours'], df['acc'], 'r-s', linewidth=2, markersize=5)
        axes[1, i].axhline(0.6, color='gray', linestyle='--', alpha=0.7, label='ACC=0.6')
        axes[1, i].set_xlabel('Lead Time (hours)')
        axes[1, i].set_ylabel('ACC')
        axes[1, i].set_title(f'{var.upper()} ACC')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlim(0, df['lead_time_hours'].max())
        axes[1, i].set_ylim(-0.2, 1.05)
        axes[1, i].legend()
        
    plt.suptitle('Standardized Model Evaluation (RMSE & ACC)', fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    plt.show()

def save(results: Dict[str, pd.DataFrame], path: str):
    """Save results to CSV (one file per variable)."""
    for var, df in results.items():
        fname = f"{path}_{var}.csv" if path.endswith('.csv') else f"{path}_{var}.csv"
        df.to_csv(fname, index=False)
    print(f"💾 Results saved to {path}_*.csv")

def mean_acc_15d(results: Dict[str, pd.DataFrame], variables: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute mean ACC over 0-360 hours (15 days)."""
    vars_to_use = variables or list(results.keys())
    data = []
    for var in vars_to_use:
        df = results[var]
        df_15 = df[df['lead_time_hours'] <= 360]
        data.append({'variable': var, 'mean_acc_15d': df_15['acc'].mean()})
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# QUICK TEST / USAGE EXAMPLE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("📦 eval_function.py loaded successfully.")
    print("Usage in your script:")
    print("  import eval_function")
    print("  res = eval_function.evaluate(forecast, truth, clim, variables=['z500'], weighted=True)")
    print("  eval_function.plot(res)")