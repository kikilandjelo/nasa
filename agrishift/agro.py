"""Pure agro-climatic helper functions (no I/O, no Streamlit).

Everything here is standard, citable agronomy so that every number the app
shows can be traced to an equation:

* Extraterrestrial radiation ........ FAO-56 (Allen et al., 1998), eq. 21
* Reference evapotranspiration ...... Hargreaves & Samani (1985), FAO-56 eq. 52
* Effective rainfall ................ USDA-SCS monthly method (as in FAO CROPWAT)
* Yield response to water stress .... FAO-33 (Doorenbos & Kassam, 1979)
"""
from __future__ import annotations

import math

import numpy as np

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Average days per month (Feb = 28.25) – used for ET0 totals.
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
# Day-of-year of the middle of each month.
MID_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
AVG_MONTH_DAYS = 30.4375


def extraterrestrial_radiation(lat_deg: float, doy: float) -> float:
    """Daily extraterrestrial radiation Ra in MJ m-2 day-1 (FAO-56 eq. 21)."""
    gsc = 0.0820  # solar constant, MJ m-2 min-1
    phi = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    x = -math.tan(phi) * math.tan(delta)
    x = max(-1.0, min(1.0, x))  # polar day / polar night
    ws = math.acos(x)
    ra = (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return max(ra, 0.0)


def hargreaves_et0(tmean, tmax, tmin, ra_mj):
    """Reference evapotranspiration ET0 in mm/day (Hargreaves & Samani, 1985).

    ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * Ra(mm/day equivalent)
    Works element-wise on numpy arrays.
    """
    trange = np.maximum(np.asarray(tmax, float) - np.asarray(tmin, float), 0.0)
    et0 = 0.0023 * (np.asarray(tmean, float) + 17.8) * np.sqrt(trange) * 0.408 * ra_mj
    return np.maximum(et0, 0.0)


def effective_rain(p_mm):
    """USDA-SCS effective rainfall (mm/month) from monthly rainfall (mm/month)."""
    p = np.asarray(p_mm, float)
    low = p * (125.0 - 0.2 * p) / 125.0
    high = 125.0 + 0.1 * p
    return np.maximum(np.where(p <= 250.0, low, high), 0.0)


def trapezoid(x, lo, opt_lo, opt_hi, hi):
    """Trapezoidal suitability function: 0 outside (lo, hi), 1 on [opt_lo, opt_hi]."""
    x = np.asarray(x, float)
    rise = (x - lo) / max(opt_lo - lo, 1e-9)
    fall = (hi - x) / max(hi - opt_hi, 1e-9)
    out = np.where(x < opt_lo, rise, np.where(x > opt_hi, fall, 1.0))
    return np.clip(out, 0.0, 1.0)


def kc_shape(u):
    """Relative crop coefficient over the season; ``u`` in [0, 1] is development progress.

    Rises from 0.35 to 1.0 over the first 35 % of development, stays at 1.0 until 75 %,
    then declines to 0.6 (generic FAO-56-style initial / mid / end stages).
    """
    u = np.asarray(u, float)
    return np.where(u <= 0.35, 0.35 + 0.65 * (u / 0.35),
                    np.where(u <= 0.75, 1.0, 1.0 - 0.4 * (u - 0.75) / 0.25))


def excess_rain_factor(p_month_mm):
    """Waterlogging penalty for crops that are not flood-tolerant (1.0 up to 250 mm/month)."""
    over = np.maximum(np.asarray(p_month_mm, float) - 250.0, 0.0)
    return np.clip(1.0 - over / 500.0, 0.2, 1.0)


def heat_factor(tmax_month_mean, tolerance_c: float):
    """Heat-stress factor from the monthly mean of daily maxima.

    1.0 up to ``tolerance_c``; falls linearly to 0 six degrees above it.
    """
    over = np.maximum(np.asarray(tmax_month_mean, float) - tolerance_c, 0.0)
    return np.clip(1.0 - over / 6.0, 0.0, 1.0)


def texture_class(clay_pct: float, sand_pct: float) -> str:
    """Very coarse texture class from clay and sand percentages."""
    if clay_pct >= 40.0:
        return "clayey"
    if sand_pct >= 70.0:
        return "sandy"
    if sand_pct >= 50.0:
        return "sandy loam"
    return "loam / silty"


# Rough plant-available water capacity (mm of water per metre of soil).
_PAWC = {"sandy": 70.0, "sandy loam": 110.0, "loam / silty": 160.0, "clayey": 130.0}


def pawc_mm_per_m(clay_pct: float, sand_pct: float) -> float:
    """Approximate plant-available water capacity for a texture class."""
    return _PAWC[texture_class(clay_pct, sand_pct)]
