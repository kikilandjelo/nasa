import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrishift import crops  # noqa: E402


class CropTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = crops.load_crops()

    def test_loads_and_has_expected_structure(self):
        self.assertGreaterEqual(len(self.df), 30)
        self.assertTrue(self.df.crop.is_unique)
        self.assertEqual(set(self.df.role), {"cash", "cover", "both"})
        self.assertGreaterEqual(len(crops.main_crop_names(self.df)), 25)

    def test_nitrogen_fixers_are_legumes_and_main_crops_have_margins(self):
        fixers = self.df[self.df.n_fixer]
        self.assertTrue((fixers.family == "Fabaceae").all())
        self.assertGreaterEqual(len(fixers), 6)
        main = self.df[self.df.role.isin(["cash", "both"])]
        self.assertTrue((main.margin_index > 0).all())
        self.assertTrue((self.df[self.df.role == "cover"].margin_index == 0).all())

    def test_thermal_time_is_consistent_with_nominal_season_length(self):
        """gdd_req / season_days + t_base is the mean temperature of a nominal cycle.

        It must be a believable temperature for the crop. This guards against typos in the table.
        """
        t_mean = self.df.t_base + self.df.gdd_req / self.df.season_days
        too_cold = t_mean < self.df.t_lo + 3
        too_hot = t_mean > self.df.t_opt_hi + 2
        self.assertFalse((too_cold | too_hot).any(), list(self.df.crop[too_cold | too_hot]))

    def test_loader_rejects_bad_thresholds(self):
        import tempfile

        bad = self.df.copy()
        bad.loc[0, "t_opt_lo"] = bad.loc[0, "t_hi"] + 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                crops.load_crops(path)


if __name__ == "__main__":
    unittest.main()
