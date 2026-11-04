"""Crop characteristics table (loaded from data/crops.csv).

The numbers in the CSV are *approximate, generic reference values* in the spirit of
FAO EcoCrop, FAO-33 (Ky) and FAO-56 (Kc). They are NOT site-calibrated: replace
them with values from your local agronomists before relying on a recommendation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CROPS_CSV = Path(__file__).parent / "data" / "crops.csv"

REQUIRED = [
    "crop", "type", "family", "role", "n_fixer", "season_days", "t_base", "gdd_req", "t_lo", "t_opt_lo",
    "t_opt_hi", "t_hi", "t_kill", "waterlog_tolerant", "kc_mid", "ky", "root_depth_m", "ph_lo",
    "ph_opt_lo", "ph_opt_hi", "ph_hi", "residue_om", "margin_index", "paddy_extra_mm_month",
]
# Field-crop types offered by default in the app (vegetables are opt-in).
DEFAULT_TYPES = ["cereal", "legume", "oilseed", "fiber", "root"]


def load_crops(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the crop table; returns a DataFrame indexed 0..n-1."""
    df = pd.read_csv(path or CROPS_CSV)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"crops table is missing columns: {missing}")
    for c in ("n_fixer", "waterlog_tolerant"):
        df[c] = df[c].astype(bool)
    bad = df[~((df.t_lo < df.t_opt_lo) & (df.t_opt_lo <= df.t_opt_hi) & (df.t_opt_hi < df.t_hi))]
    if len(bad):
        raise ValueError(f"temperature thresholds not ordered for: {list(bad.crop)}")
    bad = df[~((df.ph_lo < df.ph_opt_lo) & (df.ph_opt_lo <= df.ph_opt_hi) & (df.ph_opt_hi < df.ph_hi))]
    if len(bad):
        raise ValueError(f"pH thresholds not ordered for: {list(bad.crop)}")
    if (df.gdd_req <= 0).any() or (df.t_base >= df.t_opt_hi).any():
        raise ValueError("gdd_req must be > 0 and t_base < t_opt_hi")
    if (df.t_kill >= df.t_opt_lo).any():
        raise ValueError("t_kill (lethal monthly-mean minimum, deg C) must be below t_opt_lo")
    if df.crop.duplicated().any():
        raise ValueError("duplicate crop names in crops table")
    if not set(df.role) <= {"cash", "cover", "both"}:
        raise ValueError("role must be one of: cash, cover, both")
    return df.reset_index(drop=True)


def main_crop_names(df: pd.DataFrame) -> list[str]:
    """Crops that can be the main crop of a rotation year."""
    return list(df.loc[df.role.isin(["cash", "both"]), "crop"])
