import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agrishift import crops as crop_db  # noqa: E402
from agrishift import engine  # noqa: E402
from agrishift.models import DataError, Soil  # noqa: E402
from synth import make_climate  # noqa: E402

SOIL = Soil(6.8, 2.5, 25, 40)
SEMI_ARID = dict(lat=40.1, t_mean=15, t_amp=11.5, diurnal=11, p_month=25, wet=0.35,
                 p_seasonal=[1.1, 1.1, 1.2, 1.4, 1.4, 0.8, 0.1, 0.1, 0.6, 1.5, 1.3, 1.1])
TEMPERATE = dict(lat=50.4, t_mean=8.5, t_amp=12, diurnal=9, p_month=50, wet=0.55)
MONSOON = dict(lat=23.8, t_mean=26, t_amp=4.5, diurnal=8, p_month=165, wet=0.6, peak_month=5,
               p_seasonal=[0.05, 0.15, 0.3, 0.8, 1.6, 2.2, 2.4, 2.2, 1.8, 0.9, 0.15, 0.03])
DESERT = dict(lat=26, t_mean=24, t_amp=11, diurnal=14, p_month=2, wet=0.1)


def table(climate_kwargs, irrigation=0.0, soil=SOIL):
    crops = crop_db.load_crops()
    cd = engine.prepare(make_climate(**climate_kwargs))
    return crops, cd, engine.evaluate_all(cd, soil, crops, irrigation)


def idx(crops, name):
    return int(crops.index[crops.crop == name][0])


class CropSuitabilityTests(unittest.TestCase):
    def test_humid_temperate_climate(self):
        crops, _, tab = table(TEMPERATE)
        y = dict(zip(crops.crop, tab.mean_yield))
        self.assertGreater(y["Wheat (winter)"], engine.SUITABLE_MIN)
        self.assertLess(y["Rice"], engine.SUITABLE_MIN)
        self.assertLess(y["Cotton"], engine.SUITABLE_MIN)
        self.assertLess(y["Watermelon"], engine.SUITABLE_MIN)

    def test_monsoon_climate_favours_rice_and_rejects_winter_wheat(self):
        crops, _, tab = table(MONSOON)
        y = dict(zip(crops.crop, tab.mean_yield))
        self.assertGreater(y["Rice"], engine.SUITABLE_MIN)
        self.assertLess(y["Wheat (winter)"], engine.SUITABLE_MIN)
        self.assertLess(y["Potato"], engine.SUITABLE_MIN)

    def test_hyper_arid_site_is_only_viable_with_irrigation(self):
        crops, _, dry = table(DESERT, irrigation=0)
        _, _, wet = table(DESERT, irrigation=800)
        main = crops.role.isin(["cash", "both"]).to_numpy()
        self.assertEqual(int((dry.mean_yield[main] >= engine.SUITABLE_MIN).sum()), 0)
        self.assertGreater(int((wet.mean_yield[main] >= engine.SUITABLE_MIN).sum()), 3)

    def test_irrigation_never_hurts_any_crop(self):
        crops, _, a = table(SEMI_ARID, irrigation=0)
        _, _, b = table(SEMI_ARID, irrigation=300)
        self.assertTrue(np.all(b.mean_yield >= a.mean_yield - 1e-9))

    def test_less_rain_lowers_rainfed_suitability(self):
        crops, _, normal = table(SEMI_ARID)
        drier = dict(SEMI_ARID, p_month=SEMI_ARID["p_month"] * 0.4)
        _, _, dry = table(drier)
        main = crops.role.isin(["cash", "both"]).to_numpy()
        self.assertLess(dry.mean_yield[main].mean(), normal.mean_yield[main].mean())

    def test_spring_crops_cannot_overwinter_in_a_cold_continental_climate(self):
        crops, _, tab = table(TEMPERATE)
        row = tab.df.set_index("crop")
        self.assertIn(row.loc["Chickpea", "best_sow"], ("Mar", "Apr", "May"))
        self.assertIn(row.loc["Wheat (winter)", "best_sow"], ("Aug", "Sep", "Oct"))

    def test_southern_hemisphere_warm_season_crop_is_sown_in_the_southern_spring(self):
        _, _, north = table(dict(SEMI_ARID, p_seasonal=None, p_month=60), irrigation=400)
        south_kw = dict(lat=-34.0, t_mean=15, t_amp=11.5, diurnal=11, p_month=60, wet=0.35)
        crops, _, south = table(south_kw, irrigation=400)
        i = idx(crops, "Maize (corn)")
        self.assertIn(int(north.sow[i]), (2, 3, 4, 5))
        self.assertIn(int(south.sow[i]), (8, 9, 10, 11))

    def test_outputs_are_finite_and_bounded(self):
        crops, _, tab = table(MONSOON, irrigation=200)
        for arr in (tab.Y, tab.IRR):
            self.assertTrue(np.isfinite(arr).all())
        self.assertTrue(((tab.Y >= 0) & (tab.Y <= 1.0 + 1e-9)).all())
        self.assertTrue((tab.IRR >= 0).all())


class RotationEffectTests(unittest.TestCase):
    """Noise-free climate: every year is identical, so rotation effects can be checked exactly."""

    @classmethod
    def setUpClass(cls):
        cls.crops = crop_db.load_crops()
        cd = engine.prepare(make_climate(**dict(TEMPERATE, noise=0.0)))
        cls.tab = engine.evaluate_all(cd, SOIL, cls.crops, 300.0)
        cls.cur = idx(cls.crops, "Maize (corn)")           # neutral predecessor (Poaceae, not a legume)
        cls.ctx = engine.build_context(cls.tab, cls.crops, True, cls.cur)

    def slot(self, seq):
        G = np.array([[idx(self.crops, c) for c in seq]], dtype=np.int32)
        m = engine.rotation_metrics(G, self.ctx, n_start=10)
        return m["slot_y"][0, :, 0], m

    def test_legume_gives_the_next_non_legume_a_nitrogen_bonus(self):
        after_legume, _ = self.slot(["Chickpea", "Sunflower"])
        after_other, _ = self.slot(["Canola (rapeseed)", "Sunflower"])
        raw = self.tab.Y[idx(self.crops, "Sunflower"), 0]
        # precondition (fails loudly if the crop table changes): the bonus must not hit the 1.0 cap
        self.assertLessEqual(raw * (1 + engine.LEGUME_CARRYOVER), 1.0)
        self.assertAlmostEqual(after_legume[1] / after_other[1], 1 + engine.LEGUME_CARRYOVER, places=6)

    def test_repeating_a_botanical_family_is_penalised(self):
        same, _ = self.slot(["Barley", "Wheat (spring)"])
        diff, _ = self.slot(["Barley", "Sunflower"])
        raw_w = self.tab.Y[idx(self.crops, "Wheat (spring)"), 0]
        raw_s = self.tab.Y[idx(self.crops, "Sunflower"), 0]
        self.assertAlmostEqual(same[1], min(1.0, raw_w * engine.SAME_FAMILY_PENALTY), places=6)
        self.assertAlmostEqual(diff[1], min(1.0, raw_s), places=6)

    def test_legume_rotation_beats_cereal_monoculture_on_soil_health(self):
        mono = self.slot(["Barley", "Wheat (spring)", "Barley"])[1]["soil"][0]
        mixed = self.slot(["Barley", "Chickpea", "Sunflower"])[1]["soil"][0]
        self.assertGreater(mixed, mono + 20)

    def test_calendar_chain_invariants(self):
        pool = [idx(self.crops, c) for c in ("Wheat (winter)", "Chickpea", "Sunflower", "Barley", "Potato")]
        G = engine.enumerate_rotations(pool, 3, self.cur)
        starts = engine.chain_offsets(G, self.tab)
        for i in range(2):
            nxt = G[:, i + 1]
            earliest = starts[:, i] + self.tab.nmon[G[:, i]] - engine.OVERLAP_TOLERANCE
            self.assertTrue((starts[:, i + 1] >= earliest).all())                     # never sown before harvest
            self.assertTrue((starts[:, i + 1] % 12 == self.tab.sow[nxt]).all())       # always in its own best month
        cyc = engine.cycle_years(G, self.tab, starts)
        self.assertTrue((cyc >= 1).all())
        self.assertTrue(np.allclose(cyc, np.round(cyc)))                              # whole years

    def test_enumeration_rules(self):
        pool = [idx(self.crops, c) for c in ("Chickpea", "Sunflower", "Barley")]
        G = engine.enumerate_rotations(pool, 3, idx(self.crops, "Barley"))
        self.assertTrue((G[:, 0] != idx(self.crops, "Barley")).all())
        self.assertTrue((G[:, 0] != G[:, 1]).all() and (G[:, 1] != G[:, 2]).all())
        self.assertEqual(len(engine.enumerate_rotations([pool[0]], 3, 0)), 0)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crops = crop_db.load_crops()
        cls.field_crops = list(cls.crops.loc[cls.crops.type.isin(crop_db.DEFAULT_TYPES)
                                             & cls.crops.role.isin(["cash", "both"]), "crop"])

    def run_plan(self, kw, current="Wheat (winter)", **opts):
        opts.setdefault("allowed_crops", self.field_crops)
        return engine.run_planner(make_climate(**kw), SOIL, self.crops, current, **opts)

    def test_result_structure_for_two_three_and_four_crop_plans(self):
        for n in (2, 3, 4):
            r = self.run_plan(TEMPERATE, irrigation_cap_mm=150, rotation_years=n)
            self.assertGreater(r.n_candidates, 0)
            self.assertTrue(r.plans)
            for p in r.plans + [r.baseline]:
                self.assertEqual(len(p.crops), n)
                self.assertEqual(len(p.timeline), n)
                self.assertEqual(p.slot_yield.shape, (n, len(r.start_years)))
                self.assertEqual(p.yield_by_start.shape, r.start_years.shape)
                self.assertTrue(all(np.isfinite(v) for v in p.metrics.values()))
                self.assertGreaterEqual(p.cycle_years, 1.0)
            for p in r.plans:
                self.assertNotEqual(p.crops[0], "Wheat (winter)")            # must change what is grown next
                self.assertTrue(all(a != b for a, b in zip(p.crops, p.crops[1:])))
                self.assertTrue(p.why)

    def test_baseline_is_the_current_crop_repeated(self):
        r = self.run_plan(TEMPERATE, current="Barley", irrigation_cap_mm=100)
        self.assertEqual(set(r.baseline.crops), {"Barley"})

    def test_rotation_beats_monoculture_where_diversity_matters(self):
        r = self.run_plan(TEMPERATE, irrigation_cap_mm=150)
        self.assertGreater(r.plans[0].metrics["soil_index"], r.baseline.metrics["soil_index"] + 20)

    def test_priorities_change_the_recommendation(self):
        base = dict(SEMI_ARID, p_seasonal=None, p_month=30)
        water = self.run_plan(base, irrigation_cap_mm=400, weights=dict(profit=0, water=10, soil=0, risk=0))
        profit = self.run_plan(base, irrigation_cap_mm=400, weights=dict(profit=10, water=0, soil=0, risk=0))
        self.assertLessEqual(water.plans[0].metrics["irrigation_mm"], profit.plans[0].metrics["irrigation_mm"] + 1e-6)
        self.assertGreaterEqual(profit.plans[0].metrics["profit_index"], water.plans[0].metrics["profit_index"] - 1e-6)
        soil = self.run_plan(base, irrigation_cap_mm=400, weights=dict(profit=0, water=0, soil=10, risk=0))
        self.assertGreaterEqual(soil.plans[0].metrics["soil_index"], profit.plans[0].metrics["soil_index"] - 1e-6)

    def test_margin_overrides_change_profit(self):
        a = self.run_plan(TEMPERATE, irrigation_cap_mm=150)
        b = self.run_plan(TEMPERATE, irrigation_cap_mm=150, margin_overrides={"Wheat (winter)": 10.0})
        self.assertGreater(b.baseline.metrics["profit_index"], a.baseline.metrics["profit_index"])

    def test_cover_crops_can_be_switched_off(self):
        on = self.run_plan(TEMPERATE, irrigation_cap_mm=150, use_cover=True)
        off = self.run_plan(TEMPERATE, irrigation_cap_mm=150, use_cover=False)
        self.assertTrue(any(c for p in on.plans for c in p.cover))
        self.assertFalse(any(c for p in off.plans for c in p.cover))

    def test_hyper_arid_rainfed_gives_warnings_not_plans(self):
        r = self.run_plan(DESERT, irrigation_cap_mm=0)
        self.assertEqual(r.plans, [])
        self.assertTrue(r.warnings)
        r2 = self.run_plan(DESERT, irrigation_cap_mm=800)
        self.assertTrue(r2.plans)

    def test_invalid_inputs(self):
        with self.assertRaises(DataError):
            self.run_plan(TEMPERATE, current="Hairy vetch (cover)")
        with self.assertRaises(ValueError):
            self.run_plan(TEMPERATE, rotation_years=7)

    def test_climate_summary_trend_detects_warming(self):
        warm = make_climate(**dict(TEMPERATE, warming=0.5, noise=0.2))       # +0.5 C per decade
        cs = engine.climate_summary(engine.prepare(warm))
        self.assertAlmostEqual(cs["t_trend_per_decade"], 0.5, delta=0.1)

    def test_fuzz_random_climates_never_crash_or_return_non_finite_numbers(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            kw = dict(lat=float(rng.uniform(-60, 65)), t_mean=float(rng.uniform(-2, 28)), t_amp=float(rng.uniform(1, 16)),
                      diurnal=float(rng.uniform(4, 15)), p_month=float(rng.uniform(1, 200)), wet=float(rng.uniform(0.05, 0.8)),
                      seed=int(rng.integers(1, 1000)))
            r = self.run_plan(kw, irrigation_cap_mm=float(rng.choice([0, 150, 500])), rotation_years=int(rng.integers(2, 5)),
                              use_cover=bool(rng.integers(0, 2)))
            self.assertTrue(np.isfinite(r.crop_table[["mean_yield_idx", "irrigation_need_mm"]].to_numpy()).all())
            for p in [r.baseline] + r.plans:
                self.assertTrue(all(np.isfinite(v) for v in p.metrics.values()), kw)
                self.assertTrue(0 <= p.metrics["failure_pct"] <= 100)
                self.assertTrue(0 <= p.metrics["soil_index"] <= 100.0001)


if __name__ == "__main__":
    unittest.main()
