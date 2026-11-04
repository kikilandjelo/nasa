"""Where every number comes from - shown in the app and mirrored in the README."""
from __future__ import annotations

import pandas as pd

LINEAGE = [
    ("Air temperature (mean / max / min, 2 m)", "NASA POWER monthly API: T2M, T2M_MAX, T2M_MIN (MERRA-2 reanalysis, NASA GMAO)",
     "~0.5° x 0.625°", "Growth efficiency, heat and cold stress, thermal time (cycle length)", "Live NASA data (modelled reanalysis)"),
    ("Precipitation", "NASA POWER: PRECTOTCORR (MERRA-2 precipitation with NASA POWER bias correction)",
     "~0.5° x 0.625°", "Effective rain in the soil-water balance", "Live NASA data (modelled reanalysis)"),
    ("Root-zone soil wetness", "NASA POWER: GWETROOT (MERRA-2 land model - this is NOT SMAP)",
     "~0.5° x 0.625°", "Soil water available at sowing", "Live NASA data (modelled)"),
    ("Reference evapotranspiration (ET0)", "Hargreaves-Samani equation on POWER temperatures + latitude (FAO-56 radiation)",
     "point", "Crop water demand (ET0 x Kc)", "Derived by equation"),
    ("Vegetation greenness (NDVI)", "MODIS Terra MOD13Q1 via ORNL DAAC point service",
     "250 m, 16-day", "Satellite reality check of the growing season (not used in scoring)", "Live NASA data, on demand"),
    ("Map imagery", "NASA GIBS: MODIS NDVI, MODIS land surface temperature, GPM IMERG precipitation, MODIS true colour",
     "250 m - 10 km", "Visual context only", "Live NASA imagery"),
    ("Soil pH, organic carbon, clay, sand", "ISRIC SoilGrids 2.0, 0-30 cm weighted mean",
     "250 m (model)", "Crop pH suitability, water-holding capacity, soil-health baseline", "Live, NOT NASA - editable with your soil test"),
    ("Crop characteristics", "agrishift/data/crops.csv - approximate FAO-style reference values (EcoCrop, FAO-33, FAO-56)",
     "-", "Temperature, cold, water and pH limits; nitrogen fixation; residue", "Assumption - verify with local agronomists"),
    ("Profitability", "Relative margin index you can edit in the app",
     "-", "Profit criterion", "User assumption - not market data"),
]


def lineage_frame() -> pd.DataFrame:
    return pd.DataFrame(LINEAGE, columns=["Variable", "Dataset / product", "Resolution", "Used for", "Status"])
