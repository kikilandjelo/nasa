import datetime as dt
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agrishift import crops as crop_db  # noqa: E402
from agrishift import engine, mapview  # noqa: E402
from agrishift.lineage import LINEAGE, lineage_frame  # noqa: E402
from agrishift.models import Soil  # noqa: E402
from agrishift.report import plan_to_markdown  # noqa: E402
from synth import make_climate  # noqa: E402


class MapTests(unittest.TestCase):
    def test_imagery_date_is_lagged_and_snapped_to_composite_start(self):
        self.assertEqual(mapview.imagery_date(dt.date(2026, 11, 3), 8), "2026-10-26")
        snapped = dt.date.fromisoformat(mapview.imagery_date(dt.date(2026, 11, 3), 20, 8))
        self.assertEqual((snapped.timetuple().tm_yday - 1) % 8, 0)
        self.assertLessEqual(snapped, dt.date(2026, 10, 14))

    def test_html_embeds_real_gibs_layers_marker_and_is_script_safe(self):
        html = mapview.build_map_html(40.1438, 47.5769, today=dt.date(2026, 11, 3))
        self.assertNotIn("__CFG__", html)
        for layer in ("MODIS_Terra_NDVI_8Day", "MODIS_Terra_Land_Surface_Temp_Day", "IMERG_Precipitation_Rate",
                      "MODIS_Terra_CorrectedReflectance_TrueColor"):
            self.assertIn(layer, html)
        self.assertIn("gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi", html)
        cfg = json.loads(re.search(r"const CFG = (\{.*?\});\nconst map", html, re.S).group(1).replace("<\\/", "</"))
        self.assertAlmostEqual(cfg["lat"], 40.1438)
        self.assertEqual(len(cfg["overlays"]), 3)
        self.assertEqual(html.count("</script>"), 2)        # Leaflet + our inline script only


class ReportAndLineageTests(unittest.TestCase):
    def test_markdown_report_contains_the_decision_and_the_caveats(self):
        crops = crop_db.load_crops()
        allowed = list(crops.loc[crops.type.isin(crop_db.DEFAULT_TYPES) & crops.role.isin(["cash", "both"]), "crop"])
        soil = Soil(6.8, 2.5, 25, 40, "ISRIC SoilGrids 2.0")
        r = engine.run_planner(make_climate(lat=50.4, t_mean=8.5, t_amp=12, diurnal=9, p_month=50, wet=0.55),
                               soil, crops, "Wheat (winter)", irrigation_cap_mm=150, allowed_crops=allowed)
        w = {"profit": 5, "water": 5, "soil": 5, "risk": 5}
        md = plan_to_markdown(r, r.plans[0], 50.4, 30.5, soil, w, 150.0, today=dt.date(2026, 11, 14))
        for token in ("# AgriShift rotation plan", "2026-11-14", "50.4000", "## Compared with continuing Wheat (winter)",
                      "## Why", "Read this before acting", "not a yield forecast", "ISRIC SoilGrids 2.0"):
            self.assertIn(token, md)
        for crop in r.plans[0].crops:
            self.assertIn(crop, md)

    def test_lineage_is_explicit_about_what_is_not_nasa_or_not_data(self):
        df = lineage_frame()
        self.assertEqual(len(df), len(LINEAGE))
        status = dict(zip(df["Variable"], df["Status"]))
        soil = next(v for k, v in status.items() if k.startswith("Soil pH"))
        self.assertIn("NOT NASA", soil)
        self.assertTrue(any("Assumption" in v for v in status.values()))
        self.assertTrue(any("not SMAP".lower() in d.lower() or "NOT SMAP" in d for d in df["Dataset / product"]))


if __name__ == "__main__":
    unittest.main()
