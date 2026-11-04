"""Plain data containers shared by the data layer and the engine."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import agro


class DataError(Exception):
    """A data source returned nothing usable (ocean, outage, unexpected format)."""


@dataclass
class Climate:
    """Monthly climate record for one point; every array has shape (n_years, 12)."""

    lat: float
    lon: float
    years: np.ndarray        # (n_years,) calendar years
    tmean: np.ndarray        # deg C, monthly mean of daily mean temperature
    tmax: np.ndarray         # deg C, monthly mean of daily maximum
    tmin: np.ndarray         # deg C, monthly mean of daily minimum
    precip_mm: np.ndarray    # mm per month (total)
    root_wet: np.ndarray     # root-zone soil wetness, 0..1 (modelled, MERRA-2)
    source: str = "NASA POWER monthly API (MERRA-2 reanalysis)"

    @property
    def n_years(self) -> int:
        return int(len(self.years))

    def filled(self) -> "Climate":
        """Copy with NaNs replaced by the multi-year mean of that calendar month."""
        def fill(a: np.ndarray) -> np.ndarray:
            a = np.array(a, dtype=float, copy=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                month_mean = np.nanmean(a, axis=0)
            if np.isnan(month_mean).any():
                raise DataError("A climate variable has no valid values for at least one month.")
            rows, cols = np.where(np.isnan(a))
            a[rows, cols] = month_mean[cols]
            return a

        return Climate(
            lat=self.lat, lon=self.lon, years=np.asarray(self.years, dtype=int),
            tmean=fill(self.tmean), tmax=fill(self.tmax), tmin=fill(self.tmin),
            precip_mm=np.maximum(fill(self.precip_mm), 0.0),
            root_wet=np.clip(fill(self.root_wet), 0.0, 1.0),
            source=self.source,
        )


@dataclass
class Soil:
    """Topsoil (0-30 cm) properties used by the model."""

    ph: float
    om_pct: float        # soil organic matter, %
    clay_pct: float
    sand_pct: float
    source: str = "Assumed typical loam (no soil data supplied)"

    @property
    def texture(self) -> str:
        return agro.texture_class(self.clay_pct, self.sand_pct)

    @property
    def pawc_mm_per_m(self) -> float:
        return agro.pawc_mm_per_m(self.clay_pct, self.sand_pct)


DEFAULT_SOIL = Soil(ph=6.5, om_pct=2.0, clay_pct=25.0, sand_pct=40.0)
