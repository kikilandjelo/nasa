import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrishift import sources  # noqa: E402
from agrishift.models import DataError  # noqa: E402


def fake_power(years=range(2001, 2011), fill_all=False, drop=None, holes=()):
    """A synthetic response that follows the documented POWER monthly structure."""
    par = {k: {} for k in sources.POWER_PARAMS}
    for y in years:
        for m in range(1, 13):
            key = f"{y}{m:02d}"
            t = 12 + 10 * np.sin((m - 4) * np.pi / 6)
            vals = {"T2M": t, "T2M_MAX": t + 5, "T2M_MIN": t - 5, "PRECTOTCORR": 2.0, "GWETROOT": 0.4}
            for name, v in vals.items():
                par[name][key] = -999 if fill_all or (y, m, name) in holes else round(float(v), 2)
        for name in par:
            par[name][f"{y}13"] = 99.0           # annual aggregate that must be ignored
    if drop:
        par.pop(drop)
    return {"properties": {"parameter": par}}


class PowerParsingTests(unittest.TestCase):
    def test_shape_years_and_annual_key_ignored(self):
        c = sources.parse_power_monthly(fake_power(), 40.0, 47.0)
        self.assertEqual(c.tmean.shape, (10, 12))
        self.assertEqual((int(c.years[0]), int(c.years[-1])), (2001, 2010))
        self.assertTrue((c.tmean < 90).all())

    def test_precipitation_converted_from_mm_per_day_to_monthly_total(self):
        c = sources.parse_power_monthly(fake_power(), 40.0, 47.0)
        i2004 = list(c.years).index(2004)
        self.assertAlmostEqual(c.precip_mm[i2004, 1], 2.0 * 29)       # leap-year February
        self.assertAlmostEqual(c.precip_mm[i2004 + 1, 1], 2.0 * 28)
        self.assertAlmostEqual(c.precip_mm[0, 0], 62.0)

    def test_fill_values_are_replaced_by_the_month_mean(self):
        c = sources.parse_power_monthly(fake_power(holes={(2005, 7, "T2M")}), 40.0, 47.0)
        self.assertFalse(np.isnan(c.tmean).any())
        i = list(c.years).index(2005)
        others = np.delete(c.tmean[:, 6], i)
        self.assertAlmostEqual(c.tmean[i, 6], others.mean(), places=6)

    def test_ocean_or_missing_data_raises_clear_error(self):
        with self.assertRaises(DataError) as cm:
            sources.parse_power_monthly(fake_power(fill_all=True), 0.0, -30.0)
        self.assertIn("no land data", str(cm.exception))

    def test_missing_parameter_and_bad_payload(self):
        with self.assertRaises(DataError):
            sources.parse_power_monthly(fake_power(drop="GWETROOT"), 40.0, 47.0)
        with self.assertRaises(DataError):
            sources.parse_power_monthly({"unexpected": 1}, 40.0, 47.0)

    def test_too_few_years_is_rejected(self):
        with self.assertRaises(DataError):
            sources.parse_power_monthly(fake_power(years=range(2001, 2005)), 40.0, 47.0)

    def test_default_year_range_is_thirty_complete_years(self):
        self.assertEqual(sources.default_year_range(dt.date(2026, 11, 3)), (1996, 2025))


def soil_layer(name, vals, d=10):
    return {"name": name, "unit_measure": {"d_factor": d},
            "depths": [{"label": lab, "values": {"mean": v}} for lab, v in zip(["0-5cm", "5-15cm", "15-30cm"], vals)]}


class SoilGridsParsingTests(unittest.TestCase):
    def test_depth_weighted_means_and_unit_conversion(self):
        payload = {"properties": {"layers": [soil_layer("phh2o", [65, 66, 68]), soil_layer("soc", [200, 150, 100]),
                                             soil_layer("clay", [250, 260, 280]), soil_layer("sand", [400, 390, 380])]}}
        s = sources.parse_soilgrids(payload)
        self.assertAlmostEqual(s.ph, (5 * 6.5 + 10 * 6.6 + 15 * 6.8) / 30, places=6)
        soc_g_kg = (5 * 20 + 10 * 15 + 15 * 10) / 30
        self.assertAlmostEqual(s.om_pct, soc_g_kg / 10 * 1.724, places=6)
        self.assertAlmostEqual(s.clay_pct, (5 * 25 + 10 * 26 + 15 * 28) / 30, places=6)
        self.assertEqual(s.texture, "loam / silty")

    def test_null_values_mean_no_data(self):
        payload = {"properties": {"layers": [soil_layer(n, [None, None, None]) for n in ("phh2o", "soc", "clay", "sand")]}}
        with self.assertRaises(DataError):
            sources.parse_soilgrids(payload)
        with self.assertRaises(DataError):
            sources.parse_soilgrids({})


class ModisParsingTests(unittest.TestCase):
    def test_scale_fill_values_and_date_fallback(self):
        payload = {"scale": "0.0001", "subset": [
            {"calendar_date": "2025-01-01", "data": [3500]},
            {"modis_date": "A2025017", "data": [-3000]},             # fill value -> skipped
            {"modis_date": "A2025033", "data": [6200]}]}             # no calendar_date -> derived
        rows = sources.parse_modis_subset(payload)
        self.assertEqual([r[0] for r in rows], ["2025-01-01", "2025-02-02"])
        self.assertAlmostEqual(rows[0][1], 0.35)
        self.assertAlmostEqual(rows[1][1], 0.62)

    def test_composite_dates(self):
        d = sources.modis_composite_dates(2025)
        self.assertEqual((len(d), d[0], d[-1]), (23, "A2025001", "A2025353"))


if __name__ == "__main__":
    unittest.main()
