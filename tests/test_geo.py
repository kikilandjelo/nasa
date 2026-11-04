import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrishift import geo  # noqa: E402

NO_NET = lambda url: None  # noqa: E731


class GeoTests(unittest.TestCase):
    def test_place_pin_beats_viewport_centre(self):
        url = ("https://www.google.com/maps/place/Baku/@40.4093,49.8671,12z/data=!3m1!4b1!4m6!3m5!1s0x40307d"
               "!8m2!3d40.4092617!4d49.8671")
        self.assertEqual(geo.extract_lat_lon_from_url(url, NO_NET), (40.4092617, 49.8671))

    def test_viewport_query_and_encoded_forms(self):
        self.assertEqual(geo.extract_lat_lon_from_url("https://www.google.com/maps/@41.3625,48.5133,15z", NO_NET),
                         (41.3625, 48.5133))
        self.assertEqual(geo.extract_lat_lon_from_url("https://maps.google.com/?q=40.1438,47.5769", NO_NET),
                         (40.1438, 47.5769))
        self.assertEqual(geo.extract_lat_lon_from_url(
            "https://www.google.com/maps/search/?api=1&query=40.1438%2C47.5769", NO_NET), (40.1438, 47.5769))

    def test_plain_text_coordinates(self):
        for text in ("40.1438, 47.5769", "40.1438 47.5769", "40.1438°, 47.5769°", "-33.86; 151.21"):
            self.assertIsNotNone(geo.parse_lat_lon_text(text), text)
        self.assertEqual(geo.parse_lat_lon_text("-33.86, 151.21"), (-33.86, 151.21))

    def test_no_silent_default_for_garbage_or_out_of_range(self):
        self.assertIsNone(geo.extract_lat_lon_from_url("hello world", NO_NET))
        self.assertIsNone(geo.extract_lat_lon_from_url("", NO_NET))
        self.assertIsNone(geo.extract_lat_lon_from_url("https://x.com/@123.4,47.5", NO_NET))
        self.assertIsNone(geo.parse_lat_lon_text("95, 10"))

    def test_short_links_are_resolved_through_the_injected_resolver(self):
        final = "https://www.google.com/maps/@41.3625,48.5133,15z"
        self.assertTrue(geo.is_short_url("https://maps.app.goo.gl/abc123"))
        self.assertFalse(geo.is_short_url("https://www.google.com/maps/@1,2,3z"))
        self.assertEqual(geo.extract_lat_lon_from_url("https://maps.app.goo.gl/abc123", lambda u: final),
                         (41.3625, 48.5133))
        self.assertIsNone(geo.extract_lat_lon_from_url("https://maps.app.goo.gl/abc123", NO_NET))


if __name__ == "__main__":
    unittest.main()
