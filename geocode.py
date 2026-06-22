# Montpellier live open-data dashboard
# Copyright (C) 2026 Damien Huzard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Lightweight geocoding for the Montpellier open-data Streamlit app.

The MVP relies on a built-in landmark dictionary so a citizen can type
"near Place de la Comédie" and get a (lat, lon). Manual coordinates and
map-clicks are handled elsewhere; an external geocoder (Nominatim) is an
OPTIONAL fallback that only runs when explicitly requested.

No network calls happen at import time.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """A resolved place."""

    label: str  # human name, e.g. "Place de la Comédie"
    latitude: float
    longitude: float
    source: str  # "landmark" | "manual" | "nominatim"


# Montpellier city centre, used as a sanity reference (lat 43.6109, lon 3.8772).
# Approximate centroids are acceptable for an MVP.
def _make_landmarks() -> dict[str, Location]:
    """Build the landmark dictionary, including aliases pointing to the same Location.

    Keys are normalized (lowercase, accent-stripped, space-collapsed). Several
    keys may map to the same Location to support common aliases.
    """

    # Canonical landmarks: human label -> (lat, lon, [aliases])
    catalogue: list[tuple[str, float, float, list[str]]] = [
        ("Place de la Comédie", 43.6085, 3.8797, ["comedie", "la comedie", "place comedie"]),
        ("Gare Saint-Roch", 43.6045, 3.8806, ["gare", "saint roch", "st roch", "gare st roch", "gare saint roch", "train station", "station"]),
        ("Le Corum", 43.6135, 3.8820, ["corum", "opera berlioz", "opera berlioz le corum"]),
        ("Antigone", 43.6082, 3.8869, ["quartier antigone"]),
        ("Port Marianne", 43.5990, 3.8975, ["quartier port marianne"]),
        ("Beaux-Arts", 43.6155, 3.8810, ["beaux arts", "quartier beaux arts"]),
        ("Hôpitaux-Facultés", 43.6320, 3.8480, ["hopitaux facultes", "hopitaux", "facultes", "quartier hopitaux facultes"]),
        ("Odysseum", 43.6047, 3.9210, ["odyseum", "quartier odysseum"]),
        ("Place Jean Jaurès", 43.6105, 3.8770, ["jean jaures", "place jean jaures"]),
        ("Les Arceaux", 43.6130, 3.8680, ["arceaux", "les arceaux"]),
        ("Boutonnet", 43.6210, 3.8700, ["quartier boutonnet"]),
        ("Université de Montpellier", 43.6310, 3.8640, ["universite", "universite de montpellier", "fac", "campus", "universite montpellier", "university"]),
        ("Aéroport Montpellier", 43.5762, 4.0017, ["aeroport", "aeroport montpellier", "airport", "mediterranee", "aeroport montpellier mediterranee"]),
        ("Palavas-les-Flots", 43.5290, 3.9290, ["palavas", "plage palavas", "palavas les flots", "plage", "beach"]),
        # A few more well-known ones.
        ("Écusson", 43.6109, 3.8772, ["ecusson", "centre ville", "centre historique", "vieille ville", "downtown", "old town", "city center", "city centre"]),
        ("Place de la Préfecture", 43.6112, 3.8758, ["prefecture", "place prefecture"]),
        ("Promenade du Peyrou", 43.6116, 3.8704, ["peyrou", "le peyrou", "promenade peyrou"]),
        ("Jardin des Plantes", 43.6149, 3.8714, ["jardin des plantes"]),
        ("Place du Nombre d'Or", 43.6066, 3.8862, ["nombre d or", "place nombre d or"]),
        ("Stade de la Mosson", 43.6225, 3.8120, ["mosson", "stade mosson", "stade de la mosson"]),
        ("Polygone", 43.6086, 3.8825, ["le polygone", "centre commercial polygone"]),
    ]

    landmarks: dict[str, Location] = {}
    for label, lat, lon, aliases in catalogue:
        loc = Location(label=label, latitude=lat, longitude=lon, source="landmark")
        # Key by the normalized label and every normalized alias.
        keys = [normalize(label)] + [normalize(a) for a in aliases]
        for key in keys:
            if key and key not in landmarks:
                landmarks[key] = loc
    return landmarks


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not text:
        return ""
    # Decompose accents and drop combining marks.
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Lowercase and collapse all runs of whitespace into single spaces.
    return " ".join(stripped.lower().split())


LANDMARKS: dict[str, Location] = _make_landmarks()


def landmark_names() -> list[str]:
    """Return the human labels of all landmarks (de-duplicated, in stable order)."""
    seen: set[str] = set()
    names: list[str] = []
    for loc in LANDMARKS.values():
        if loc.label not in seen:
            seen.add(loc.label)
            names.append(loc.label)
    return names


# Leading articles/determiners stripped before matching ("the university" ->
# "university"). Safe because every landmark also has an article-free alias.
_LEADING_ARTICLES = {"the", "a", "an", "le", "la", "les", "l", "du", "de", "des", "au", "aux"}


def _strip_leading_articles(query: str) -> str:
    tokens = query.split()
    while len(tokens) > 1 and tokens[0] in _LEADING_ARTICLES:
        tokens.pop(0)
    return " ".join(tokens)


def _best_substring_match(query: str) -> Location | None:
    best_loc: Location | None = None
    best_key_len: int | None = None
    for key, loc in LANDMARKS.items():
        norm_label = normalize(loc.label)
        # Match if a landmark key/label is contained in the query, or the query
        # is contained in a key/label.
        candidates = (key, norm_label)
        matched = any(cand and (cand in query or query in cand) for cand in candidates)
        if matched:
            # Prefer the shortest matching key (most specific landmark name).
            key_len = len(key)
            if best_key_len is None or key_len < best_key_len:
                best_key_len = key_len
                best_loc = loc
    return best_loc


def resolve_location(text: str, allow_nominatim: bool = False) -> Location | None:
    """Resolve free text to a Location.

    Strategy: normalize, then for both the full query and an article-stripped
    variant, try an exact key match followed by substring/contains matching
    against landmark keys and labels (preferring the best/shortest match). If
    nothing matches and ``allow_nominatim`` is True, fall back to the Nominatim
    geocoder. Returns None if unresolved or input is blank.
    """
    query = normalize(text)
    if not query:
        return None

    tried: list[str] = []
    for candidate in (query, _strip_leading_articles(query)):
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        exact = LANDMARKS.get(candidate)
        if exact is not None:
            return exact
        match = _best_substring_match(candidate)
        if match is not None:
            return match

    # Optional external fallback.
    if allow_nominatim:
        return geocode_nominatim(text)

    return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in metres."""
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def walk_estimate_min(distance_m: float, speed_m_per_min: float = 80.0) -> float:
    """Straight-line walking-time ESTIMATE in minutes (default 80 m/min)."""
    if speed_m_per_min <= 0:
        raise ValueError("speed_m_per_min must be positive")
    return distance_m / speed_m_per_min


def geocode_nominatim(text: str, timeout: int = 8) -> Location | None:
    """Geocode free text via OpenStreetMap Nominatim, biased to Montpellier.

    Only runs when explicitly called. Imports ``requests`` lazily so the module
    stays import-safe even if ``requests`` is unavailable. Returns None on any
    error or empty result.
    """
    if not text or not text.strip():
        return None

    query = text.strip()
    bias = ", montpellier, france"
    if bias not in normalize(query):
        query = f"{query}, Montpellier, France"

    try:
        import requests

        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "montpellier-live-streamlit/0.1"},
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        first = results[0]
        label = first.get("display_name") or text.strip()
        return Location(
            label=label,
            latitude=float(first["lat"]),
            longitude=float(first["lon"]),
            source="nominatim",
        )
    except Exception:
        return None
