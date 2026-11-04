import sys; sys.path.insert(0, "/home/claude/agrishift_v2")
import numpy as np
from agrishift.models import Climate

def make_climate(lat, t_mean, t_amp, diurnal, p_month, p_seasonal=None, wet=0.4, n_years=30, seed=1, warming=0.0, peak_month=6.5, noise=1.0):
    """Synthetic monthly climate. p_seasonal: 12 multipliers on p_month (rain seasonality)."""
    rng = np.random.default_rng(seed)
    months = np.arange(12)
    sign = 1.0 if lat >= 0 else -1.0
    seas = np.cos(2*np.pi*(months - (peak_month if lat >= 0 else (peak_month+6) % 12))/12)
    years = np.arange(1996, 1996+n_years)
    tm = np.empty((n_years, 12)); pm = np.empty((n_years, 12))
    ps = np.ones(12) if p_seasonal is None else np.asarray(p_seasonal, float)
    for i, y in enumerate(years):
        tm[i] = t_mean + t_amp*seas + rng.normal(0, 0.8, 12)*noise + warming*i/10
        pm[i] = np.maximum(0, p_month*ps*rng.lognormal(0, 0.45*noise, 12))
    return Climate(lat=lat, lon=0.0, years=years, tmean=tm, tmax=tm+diurnal/2, tmin=tm-diurnal/2,
                   precip_mm=pm, root_wet=np.full((n_years, 12), wet))
