"""Leaflet map with real NASA GIBS imagery layers (pure HTML builder, no Streamlit import).

The GIBS layers are *visual context only* - they do not feed the crop model. We use GIBS's WMS
endpoint (no tile-matrix guessing) and a date that is old enough for the imagery to exist.
"""
from __future__ import annotations

import datetime as dt
import json

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"

OVERLAYS = [
    {"title": "NDVI - vegetation greenness (MODIS Terra, 8-day)", "layer": "MODIS_Terra_NDVI_8Day",
     "format": "image/png", "lag_days": 20, "snap_days": 8, "opacity": 0.75, "default": True},
    {"title": "Land surface temperature, day (MODIS Terra)", "layer": "MODIS_Terra_Land_Surface_Temp_Day",
     "format": "image/png", "lag_days": 8, "snap_days": 0, "opacity": 0.7, "default": False},
    {"title": "Precipitation rate (GPM IMERG)", "layer": "IMERG_Precipitation_Rate",
     "format": "image/png", "lag_days": 8, "snap_days": 0, "opacity": 0.7, "default": False},
]
BASE_TRUE_COLOR = {"title": "NASA true colour (MODIS Terra)", "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
                   "format": "image/jpeg", "lag_days": 3, "snap_days": 0}

# NASA POWER meteorology comes from MERRA-2, a global model grid of roughly 0.5 deg x 0.625 deg.
MERRA2_HALF_LAT, MERRA2_HALF_LON = 0.25, 0.3125


def imagery_date(today: dt.date, lag_days: int, snap_days: int = 0) -> str:
    """ISO date ``lag_days`` before ``today``; composite layers are snapped to their period start."""
    d = today - dt.timedelta(days=lag_days)
    if snap_days:
        doy = (d.timetuple().tm_yday - 1) // snap_days * snap_days
        d = dt.date(d.year, 1, 1) + dt.timedelta(days=doy)
    return d.isoformat()


def build_config(lat: float, lon: float, today: dt.date | None = None, zoom: int = 8) -> dict:
    today = today or dt.date.today()

    def entry(spec: dict) -> dict:
        return {"title": spec["title"], "layer": spec["layer"], "format": spec["format"],
                "time": imagery_date(today, spec["lag_days"], spec.get("snap_days", 0)),
                "opacity": spec.get("opacity", 1.0), "default": bool(spec.get("default", False))}

    return {"lat": float(lat), "lon": float(lon), "zoom": int(zoom), "wms": GIBS_WMS,
            "base": entry(BASE_TRUE_COLOR), "overlays": [entry(o) for o in OVERLAYS],
            "halfLat": MERRA2_HALF_LAT, "halfLon": MERRA2_HALF_LON}


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<style>html,body{height:100%;margin:0;font-family:sans-serif}#map{height:100%}
.note{position:absolute;z-index:1000;left:8px;bottom:8px;background:rgba(255,255,255,.88);padding:4px 8px;
font-size:11px;border-radius:4px;max-width:70%}</style></head><body><div id="map"></div>
<div class="note">NASA GIBS imagery is shown for context only and does not feed the model. The dashed box is
roughly the resolution of the MERRA-2 climate grid behind NASA POWER (0.5&deg; &times; 0.625&deg;).</div>
<script>
const CFG = __CFG__;
const map = L.map('map').setView([CFG.lat, CFG.lon], CFG.zoom);
const carto = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {subdomains: 'abcd', maxZoom: 18, attribution: '&copy; OpenStreetMap contributors &copy; CARTO'}).addTo(map);
function wms(e, extra) {
  return L.tileLayer.wms(CFG.wms, Object.assign({layers: e.layer, format: e.format, version: '1.1.1',
    transparent: e.format === 'image/png', time: e.time, opacity: e.opacity === undefined ? 1 : e.opacity,
    attribution: 'NASA GIBS'}, extra || {}));
}
const bases = {'Street map': carto};
bases[CFG.base.title + ' [' + CFG.base.time + ']'] = wms(CFG.base);
const overlays = {};
CFG.overlays.forEach(function (o) {
  const layer = wms(o);
  overlays[o.title + ' [' + o.time + ']'] = layer;
  if (o.default) { layer.addTo(map); }
});
L.control.layers(bases, overlays, {collapsed: false}).addTo(map);
L.marker([CFG.lat, CFG.lon]).addTo(map).bindPopup('Your field');
L.rectangle([[CFG.lat - CFG.halfLat, CFG.lon - CFG.halfLon], [CFG.lat + CFG.halfLat, CFG.lon + CFG.halfLon]],
  {color: '#d9480f', weight: 1, dashArray: '5', fill: false}).addTo(map)
  .bindTooltip('~ resolution scale of the MERRA-2 climate grid');
</script></body></html>"""


def build_map_html(lat: float, lon: float, today: dt.date | None = None, zoom: int = 8) -> str:
    cfg = json.dumps(build_config(lat, lon, today, zoom)).replace("</", "<\\/")
    return _TEMPLATE.replace("__CFG__", cfg)
