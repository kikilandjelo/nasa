import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrishift import agro  # noqa: E402


class AgroTests(unittest.TestCase):
    def test_extraterrestrial_radiation_matches_fao56_example_8(self):
        # FAO-56 Example 8: 20 deg S on 3 September (day 246) -> Ra = 32.2 MJ m-2 day-1
        self.assertAlmostEqual(agro.extraterrestrial_radiation(-20, 246), 32.2, delta=0.2)

    def test_radiation_polar_night_is_zero_and_never_negative(self):
        self.assertEqual(agro.extraterrestrial_radiation(85, 355), 0.0)
        for lat in (-80, -45, 0, 45, 80):
            for doy in (15, 105, 196, 288):
                self.assertGreaterEqual(agro.extraterrestrial_radiation(lat, doy), 0.0)

    def test_effective_rain_is_continuous_at_250_and_bounded(self):
        self.assertAlmostEqual(float(agro.effective_rain(250.0)), 150.0)
        self.assertAlmostEqual(float(agro.effective_rain(250.0001)), float(agro.effective_rain(249.9999)), places=2)
        self.assertEqual(float(agro.effective_rain(0.0)), 0.0)
        p = np.linspace(0, 600, 61)
        eff = agro.effective_rain(p)
        self.assertTrue(np.all(eff <= p + 1e-9) and np.all(np.diff(eff) >= 0))

    def test_hargreaves_grows_with_temperature_range_and_is_non_negative(self):
        low = float(agro.hargreaves_et0(20, 24, 16, 35.0))
        high = float(agro.hargreaves_et0(20, 30, 10, 35.0))
        self.assertGreater(high, low)
        self.assertAlmostEqual(float(agro.hargreaves_et0(25, 30, 20, 35.0)), 4.45, delta=0.05)
        self.assertEqual(float(agro.hargreaves_et0(-30, -25, -35, 5.0)), 0.0)  # (T+17.8) < 0 -> clipped

    def test_trapezoid_shape(self):
        f = lambda x: float(agro.trapezoid(x, 5, 15, 25, 35))  # noqa: E731
        self.assertEqual(f(0), 0.0)
        self.assertEqual(f(40), 0.0)
        self.assertEqual(f(20), 1.0)
        self.assertAlmostEqual(f(10), 0.5)
        self.assertAlmostEqual(f(30), 0.5)
        v = agro.trapezoid(np.array([0, 10, 20, 30, 40]), 5, 15, 25, 35)
        self.assertEqual(v.shape, (5,))

    def test_kc_shape_stages(self):
        self.assertAlmostEqual(float(agro.kc_shape(0.0)), 0.35)
        self.assertAlmostEqual(float(agro.kc_shape(0.5)), 1.0)
        self.assertAlmostEqual(float(agro.kc_shape(1.0)), 0.6)

    def test_heat_and_waterlogging_factors(self):
        self.assertEqual(float(agro.heat_factor(30, 35)), 1.0)
        self.assertAlmostEqual(float(agro.heat_factor(38, 35)), 0.5)
        self.assertEqual(float(agro.heat_factor(45, 35)), 0.0)
        self.assertEqual(float(agro.excess_rain_factor(100)), 1.0)
        self.assertAlmostEqual(float(agro.excess_rain_factor(500)), 0.5)
        self.assertEqual(float(agro.excess_rain_factor(5000)), 0.2)

    def test_texture_and_water_holding(self):
        self.assertEqual(agro.texture_class(10, 85), "sandy")
        self.assertEqual(agro.texture_class(50, 20), "clayey")
        self.assertEqual(agro.texture_class(15, 60), "sandy loam")
        self.assertEqual(agro.texture_class(25, 40), "loam / silty")
        self.assertGreater(agro.pawc_mm_per_m(25, 40), agro.pawc_mm_per_m(10, 85))


if __name__ == "__main__":
    unittest.main()
