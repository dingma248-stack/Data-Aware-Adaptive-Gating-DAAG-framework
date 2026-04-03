import numpy as np

def interpolate_data(arr, target_len):
    """Linear interpolation to fixed length."""
    if len(arr) == 0: return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)

def normalize_time_series(cycle_data):
    """
    Normalizes time series data locally per cycle or globally if fitted scaler provided.
    Here we assume MinMax [0,1] normalization per feature across the whole dataset is handled externally,
    or we do local normalization (MinMax per cycle) which is robust for shape features but loses absolute magnitude.
    Current Transfer Learning best practice for Battery:
    Fit Scaler on Source Domain, Transform Target Domain.
    """
    pass # Managed in the main script via sklearn
