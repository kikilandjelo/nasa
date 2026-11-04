"""Data access layer: NASA POWER, ISRIC SoilGrids and MODIS NDVI (ORNL DAAC).

Design rules (the opposite of the v1 prototype):
  * every network call has a timeout and raises ``DataError`` with a readable message;
  * nothing is ever fabricated or silently replaced by a default;
  * parsing is separate from fetching, so it is unit-tested offline.
"""
from __future__ import annotations

import calendar
import datetime as dt

import numpy as np
import requests

from .models import Climate, DataError, Soil

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT"]
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
MODIS_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
USER_AGENT = "AgriShift/2.0 (NASA Space Apps 2026 prototype)"

SOIL_DEPTHS = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}   # label -> thickness (cm)
_SOIL_D_FACTOR = {"phh2o": 10.0, "soc": 10.0, "clay": 10.0, "sand": 10.0}
SOIL_SOURCE = "ISRIC SoilGrids 2.0, 0-30 cm weighted mean (250 m model, not a field survey)"


def default_year_range(today: dt.date | None = None, n_years: int = 30) -> tuple[int, int]:
    """Last ``n_years`` complete calendar years."""
    today = today or dt.date.today()
    end = today.year - 1
    return end - n_years + 1, end


# --------------------------------------------------------------------------- NASA POWER
def _is_month_key(k) -> bool:
    return isinstance(k, str) and len(k) == 6 and k.isdigit() and 1 <= int(k[4:]) <= 12


def parse_power_monthly(payload: dict, lat: float, lon: float) -> Climate:
    """Turn a POWER monthly-point JSON response into a ``Climate``.

    POWER monthly responses hold one value per "YYYYMM" key plus an annual
    aggregate under month "13" (ignored here). Fill value is -999. PRECTOTCORR is a
    monthly *mean in mm/day*, so it is multiplied by the days in the month.
    """
    try:
        par = payload["properties"]["parameter"]
    except (KeyError, TypeError):
        raise DataError("Unexpected NASA POWER response (no 'properties.parameter').") from None
    for name in POWER_PARAMS:
        if not isinstance(par.get(name), dict):
            raise DataError(f"NASA POWER response is missing '{name}'.")

    keys = [k for k in par["T2M"] if _is_month_key(k)]
    if not keys:
        raise DataError("NASA POWER returned no monthly values.")
    years = sorted({int(k[:4]) for k in keys})
    yidx = {y: i for i, y in enumerate(years)}

    def grid(name: str, per_day_to_month: bool = False) -> np.ndarray:
        a = np.full((len(years), 12), np.nan)
        for k, v in par[name].items():
            if not _is_month_key(k) or v is None:
                continue
            v = float(v)
            if v <= -990.0:          # POWER fill value (-999)
                continue
            y, m = int(k[:4]), int(k[4:6])
            if y not in yidx:
                continue
            a[yidx[y], m - 1] = v * calendar.monthrange(y, m)[1] if per_day_to_month else v
        return a

    tmean, tmax, tmin = grid("T2M"), grid("T2M_MAX"), grid("T2M_MIN")
    precip, wet = grid("PRECTOTCORR", per_day_to_month=True), grid("GWETROOT")

    if np.isnan(tmean).all() or np.isnan(precip).all():
        raise DataError("NASA POWER has no land data for this point (ocean or unsupported location).")
    keep = ~np.isnan(tmean).all(axis=1)          # drop years with no data at all
    if keep.sum() < 8:
        raise DataError("Fewer than 8 usable years of NASA POWER data for this point.")
    if np.isnan(wet).all():
        wet = np.full_like(wet, 0.5)             # soil wetness unavailable: neutral, flagged below
    clim = Climate(lat=lat, lon=lon, years=np.array(years)[keep], tmean=tmean[keep], tmax=tmax[keep],
                   tmin=tmin[keep], precip_mm=precip[keep], root_wet=wet[keep])
    return clim.filled()


def _error_detail(r: requests.Response) -> str:
    try:
        j = r.json()
        msgs = j.get("messages") or j.get("message") or j.get("detail")
        if msgs:
            return str(msgs)[:200]
    except ValueError:
        pass
    return r.text[:160].replace("\n", " ")


def fetch_power_monthly(lat: float, lon: float, start_year: int, end_year: int,
                        timeout: float = 45.0) -> Climate:
    """Monthly NASA POWER (MERRA-2) record for one point; raises DataError on any problem."""
    params = {"parameters": ",".join(POWER_PARAMS), "community": "AG",
              "longitude": f"{lon:.4f}", "latitude": f"{lat:.4f}",
              "start": int(start_year), "end": int(end_year), "format": "JSON"}
    try:
        r = requests.get(POWER_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        raise DataError(f"Could not reach NASA POWER ({e.__class__.__name__}).") from e
    if r.status_code != 200:
        raise DataError(f"NASA POWER returned HTTP {r.status_code}: {_error_detail(r)}")
    try:
        payload = r.json()
    except ValueError:
        raise DataError("NASA POWER returned a response that is not JSON.") from None
    return parse_power_monthly(payload, lat, lon)


# --------------------------------------------------------------------------- SoilGrids
def parse_soilgrids(payload: dict) -> Soil:
    """Depth-weighted 0-30 cm means from a SoilGrids ``properties/query`` response.

    The API returns integers in "mapped" units; ``unit_measure.d_factor`` converts
    them to conventional units (pH*10 -> pH, dg/kg -> g/kg, g/kg -> %).
    """
    try:
        layers = payload["properties"]["layers"]
    except (KeyError, TypeError):
        raise DataError("Unexpected SoilGrids response format.") from None
    out: dict[str, float] = {}
    for layer in layers:
        name = layer.get("name")
        d = (layer.get("unit_measure") or {}).get("d_factor") or _SOIL_D_FACTOR.get(name, 1.0)
        num = den = 0.0
        for depth in layer.get("depths", []):
            mean = (depth.get("values") or {}).get("mean")
            if mean is None:
                continue
            w = SOIL_DEPTHS.get(depth.get("label"), 10.0)
            num += w * float(mean) / float(d)
            den += w
        if den > 0:
            out[name] = num / den
    if not {"phh2o", "soc", "clay", "sand"} <= set(out):
        raise DataError("SoilGrids has no soil data for this point (water, ice or built-up pixel?).")
    soc_pct = out["soc"] / 10.0                      # g/kg -> % organic carbon
    return Soil(ph=out["phh2o"], om_pct=soc_pct * 1.724, clay_pct=out["clay"],
                sand_pct=out["sand"], source=SOIL_SOURCE)


def fetch_soilgrids(lat: float, lon: float, timeout: float = 30.0) -> Soil:
    params = [("lon", f"{lon:.5f}"), ("lat", f"{lat:.5f}")]
    params += [("property", p) for p in ("phh2o", "soc", "clay", "sand")]
    params += [("depth", d) for d in SOIL_DEPTHS]
    params += [("value", "mean")]
    try:
        r = requests.get(SOILGRIDS_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        raise DataError(f"Could not reach ISRIC SoilGrids ({e.__class__.__name__}).") from e
    if r.status_code != 200:
        raise DataError(f"SoilGrids returned HTTP {r.status_code}: {_error_detail(r)}")
    try:
        payload = r.json()
    except ValueError:
        raise DataError("SoilGrids returned a response that is not JSON.") from None
    return parse_soilgrids(payload)


# --------------------------------------------------------------------------- MODIS NDVI
def modis_composite_dates(year: int) -> list[str]:
    """MOD13Q1 16-day composite start dates for a year, as 'AYYYYDDD'."""
    return [f"A{year}{doy:03d}" for doy in range(1, 366, 16)]


def parse_modis_subset(payload: dict) -> list[tuple[str, float]]:
    """(ISO date, NDVI) pairs from an ORNL DAAC MODIS subset response (scale 0.0001)."""
    try:
        scale = float(payload.get("scale") or 0.0001)
    except (TypeError, ValueError):
        scale = 0.0001
    rows: list[tuple[str, float]] = []
    for item in payload.get("subset", []) or []:
        vals = [v for v in (item.get("data") or []) if v is not None and -2000 <= v <= 10000]
        if not vals:
            continue                                  # fill value (-3000) / cloud: skip
        date = item.get("calendar_date")
        if not date and item.get("modis_date"):
            date = dt.datetime.strptime(item["modis_date"][1:], "%Y%j").date().isoformat()
        if date:
            rows.append((date, float(np.mean(vals)) * scale))
    return rows


def fetch_modis_ndvi(lat: float, lon: float, year: int, timeout: float = 60.0,
                     chunk: int = 10) -> list[tuple[str, float]]:
    """MODIS Terra MOD13Q1 250 m NDVI at one pixel for a whole year (requests of <=10 composites)."""
    dates = modis_composite_dates(year)
    rows: list[tuple[str, float]] = []
    for i in range(0, len(dates), chunk):
        part = dates[i:i + chunk]
        params = {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "band": "250m_16_days_NDVI",
                  "startDate": part[0], "endDate": part[-1], "kmAboveBelow": 0, "kmLeftRight": 0}
        try:
            r = requests.get(MODIS_URL, params=params, timeout=timeout,
                             headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        except requests.RequestException as e:
            raise DataError(f"Could not reach ORNL DAAC MODIS service ({e.__class__.__name__}).") from e
        if r.status_code != 200:
            raise DataError(f"ORNL DAAC MODIS service returned HTTP {r.status_code}: {_error_detail(r)}")
        try:
            rows.extend(parse_modis_subset(r.json()))
        except ValueError:
            raise DataError("ORNL DAAC MODIS service returned a response that is not JSON.") from None
    if not rows:
        raise DataError("No valid MODIS NDVI pixels returned (cloud, water or fill values).")
    return sorted(rows)
