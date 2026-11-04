"""Coordinate helpers: parse Google Maps links and pasted 'lat, lon' text.

Unlike the v1 prototype this never falls back silently to a default location and never
contains hard-coded shortcuts: it returns ``None`` when it cannot find valid coordinates.
"""
from __future__ import annotations

import re
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import requests

_NUM = r"(-?\d{1,3}(?:\.\d+)?)"
_URL_PATTERNS = [
    re.compile(r"!3d" + _NUM + r"!4d" + _NUM),                                    # place pin (most precise)
    re.compile(r"@" + _NUM + r"," + _NUM),                                         # map viewport centre
    re.compile(r"[?&](?:q|ll|query|center|destination|daddr)=" + _NUM + r"\s*,\s*" + _NUM, re.I),
]
_TEXT_PATTERN = re.compile(r"^\s*" + _NUM + r"\s*°?\s*[,;\s]\s*" + _NUM + r"\s*°?\s*$")
SHORT_HOSTS = ("maps.app.goo.gl", "goo.gl", "g.co")

Coords = tuple[float, float]


def is_valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def parse_lat_lon_text(text: str) -> Optional[Coords]:
    """'40.1438, 47.5769' / '40.1438 47.5769' / '40.1438°, 47.5769°' -> (lat, lon)."""
    m = _TEXT_PATTERN.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return (lat, lon) if is_valid(lat, lon) else None


def _from_url_text(url: str) -> Optional[Coords]:
    decoded = unquote(url)
    for pat in _URL_PATTERNS:
        m = pat.search(decoded)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if is_valid(lat, lon):
                return lat, lon
    return None


def is_short_url(url: str) -> bool:
    host = (urlparse(url if "//" in url else "//" + url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in SHORT_HOSTS)


def resolve_short_url(url: str, timeout: float = 6.0) -> Optional[str]:
    """Follow redirects of a short map link and return the final URL (None on failure)."""
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 (AgriShift)"})
        final = r.url
        r.close()
        return final
    except requests.RequestException:
        return None


def extract_lat_lon_from_url(url: str, resolver: Callable[[str], Optional[str]] = resolve_short_url) -> Optional[Coords]:
    """Coordinates from plain 'lat, lon' text or from a Google Maps link (short links are resolved)."""
    url = (url or "").strip()
    if not url:
        return None
    direct = parse_lat_lon_text(url) or _from_url_text(url)
    if direct:
        return direct
    if is_short_url(url):
        final = resolver(url)
        if final:
            return _from_url_text(final)
    return None
