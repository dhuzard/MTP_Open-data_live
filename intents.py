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

"""Approved, deterministic intents over Montpellier live open data.

The LLM layer may only classify a citizen question into one of INTENTS (or
"unsupported"). The functions here do the actual fetching/filtering/ranking and
return a structured, JSON-serializable result. The LLM then *explains* that
result; it never produces the underlying numbers.

Every result follows the same schema so the UI and the provider's explain()
step can rely on it:

    {
      "ok": bool,
      "intent": str,
      "title": str,
      "summary": str,            # deterministic, citizen-readable fallback sentence
      "items": list[dict],       # the rows actually used (with distance/timestamp)
      "data_used": list[dict],   # {resource, id, timestamp} provenance
      "freshness": dict,         # oldest/newest/age_seconds/stale
      "location": dict | None,   # resolved location, if any
      "confidence": float,
      "notes": list[str],        # caveats (e.g. distances are straight-line)
    }
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from geocode import (
    Location,
    haversine_m,
    landmark_names,
    resolve_location,
    walk_estimate_min,
)

# The LLM may classify ONLY into these. Anything else -> "unsupported".
INTENTS: List[str] = [
    "find_nearest_bike_station",
    "find_nearest_parking",
    "plan_bike_trip",
    "summarize_city_now",
    "summarize_ecological_signal",
    "explain_data_quality",
    "find_glass_container",
]

FALLBACK_MESSAGE = (
    "I cannot answer that from the available Montpellier open data. "
    "I can help with finding a bike, finding car parking, planning a bike trip, "
    "finding a glass container, a live city summary, an ecological summary, "
    "or a data-quality report."
)

STRAIGHT_LINE_NOTE = (
    "Distances are straight-line (as the crow flies), not routed walking distance; "
    "walk times are rough estimates at 80 m/min."
)

STALE_AFTER_SECONDS = 15 * 60  # live readings older than this are flagged stale


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_int(value: Any) -> Optional[int]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(round(f))


def _has_coords(df: pd.DataFrame) -> bool:
    return not df.empty and "latitude" in df.columns and "longitude" in df.columns


def _row_timestamp(row: Mapping[str, Any]) -> Optional[str]:
    for key in row.keys():
        if key.endswith("_timestamp") or key in ("TimeInstant", "dateObserved", "observedAt"):
            value = row.get(key)
            if pd.notna(value):
                return str(value)
    return None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _freshness(timestamps: List[Optional[str]]) -> Dict[str, Any]:
    parsed = [d for d in (_parse_dt(t) for t in timestamps) if d is not None]
    if not parsed:
        return {"oldest_timestamp": None, "newest_timestamp": None, "age_seconds": None, "stale": None}
    now = datetime.now(timezone.utc)
    oldest, newest = min(parsed), max(parsed)
    age = (now - oldest).total_seconds()
    return {
        "oldest_timestamp": oldest.isoformat(),
        "newest_timestamp": newest.isoformat(),
        "age_seconds": age,
        "stale": age > STALE_AFTER_SECONDS,
    }


def _with_distance(df: pd.DataFrame, location: Location) -> pd.DataFrame:
    out = df.copy()
    out["_lat"] = _num(out["latitude"])
    out["_lon"] = _num(out["longitude"])
    out = out.dropna(subset=["_lat", "_lon"])
    if out.empty:
        return out
    out["distance_m"] = out.apply(
        lambda r: haversine_m(location.latitude, location.longitude, r["_lat"], r["_lon"]),
        axis=1,
    )
    return out.sort_values("distance_m")


def _label_for(row: Mapping[str, Any], label_cols: List[str]) -> str:
    for col in label_cols:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return str(row.get("id", "unknown"))


def _item(row: Mapping[str, Any], value_col: str, label_cols: List[str], role: Optional[str] = None,
          extra_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    distance = float(row["distance_m"])
    item: Dict[str, Any] = {
        "id": row.get("id"),
        "label": _label_for(row, label_cols),
        "value": _safe_int(row.get(value_col)) if value_col else None,
        "distance_m": round(distance),
        "walk_min_est": round(walk_estimate_min(distance)),
        "latitude": float(row["_lat"]),
        "longitude": float(row["_lon"]),
        "timestamp": _row_timestamp(row),
    }
    if role:
        item["role"] = role
    if extra_fields:
        item["fields"] = {
            f: (None if pd.isna(row.get(f)) else row.get(f)) for f in extra_fields if f in row
        }
    return item


def _result(intent: str, ok: bool, title: str, summary: str, items: List[Dict[str, Any]],
            resource: str, freshness: Dict[str, Any], location: Optional[Location] = None,
            notes: Optional[List[str]] = None, confidence: Optional[float] = None) -> Dict[str, Any]:
    data_used = [{"resource": resource, "id": it.get("id"), "timestamp": it.get("timestamp")} for it in items]
    return {
        "ok": ok,
        "intent": intent,
        "title": title,
        "summary": summary,
        "items": items,
        "data_used": data_used,
        "freshness": freshness,
        "location": (
            {
                "label": location.label,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "source": location.source,
            }
            if location is not None
            else None
        ),
        "confidence": confidence if confidence is not None else (0.9 if ok else 0.3),
        "notes": notes or [],
    }


def _empty(intent: str, message: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "intent": intent,
        "title": "No result",
        "summary": message,
        "items": [],
        "data_used": [],
        "freshness": {"oldest_timestamp": None, "newest_timestamp": None, "age_seconds": None, "stale": None},
        "location": None,
        "confidence": 0.3,
        "notes": [],
    }


def _need_location(intent: str, what: str = "location") -> Dict[str, Any]:
    suggestions = ", ".join(sorted(landmark_names())[:12])
    message = (
        f"I need a {what} to answer this. Try a known landmark "
        f"(e.g. {suggestions}), drop a pin on the map, or enter coordinates."
    )
    return _empty(intent, message)


# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #
def find_nearest_bike_station(datasets: Mapping[str, pd.DataFrame], location: Location, n: int = 3) -> Dict[str, Any]:
    df = datasets.get("bikestation", pd.DataFrame())
    if not _has_coords(df) or "availableBikeNumber" not in df.columns:
        return _empty("find_nearest_bike_station", "VéloMagg bike-station data is unavailable right now.")
    avail = df.copy()
    avail["availableBikeNumber"] = _num(avail["availableBikeNumber"])
    avail = avail[avail["availableBikeNumber"] > 0]
    if "status" in avail.columns:
        closed = avail["status"].astype(str).str.contains("closed|out|noPlan", case=False, na=False)
        avail = avail[~closed]
    ranked = _with_distance(avail, location)
    if ranked.empty:
        return _empty("find_nearest_bike_station", f"No VéloMagg station with available bikes found near {location.label}.")
    items = [_item(r, "availableBikeNumber", ["streetAddress", "address", "name"]) for _, r in ranked.head(n).iterrows()]
    nearest = items[0]
    summary = (
        f"Nearest VéloMagg station with bikes near {location.label}: {nearest['label']} — "
        f"{nearest['value']} bike(s) available, ≈{nearest['distance_m']} m away "
        f"(~{nearest['walk_min_est']} min walk, straight-line estimate)."
    )
    return _result("find_nearest_bike_station", True, f"Bikes near {location.label}", summary, items,
                   "bikestation", _freshness([it["timestamp"] for it in items]), location,
                   notes=[STRAIGHT_LINE_NOTE])


def find_nearest_parking(datasets: Mapping[str, pd.DataFrame], location: Location, n: int = 3) -> Dict[str, Any]:
    df = datasets.get("offstreetparking", pd.DataFrame())
    if not _has_coords(df) or "availableSpotNumber" not in df.columns:
        return _empty("find_nearest_parking", "Off-street parking data is unavailable right now.")
    avail = df.copy()
    avail["availableSpotNumber"] = _num(avail["availableSpotNumber"])
    avail = avail[avail["availableSpotNumber"] > 0]
    if "status" in avail.columns:
        closed = avail["status"].astype(str).str.contains("closed|full|out", case=False, na=False)
        avail = avail[~closed]
    ranked = _with_distance(avail, location)
    if ranked.empty:
        return _empty("find_nearest_parking", f"No parking with free spaces found near {location.label}.")
    items = [_item(r, "availableSpotNumber", ["name", "address", "streetAddress"]) for _, r in ranked.head(n).iterrows()]
    nearest = items[0]
    summary = (
        f"Nearest car park with space near {location.label}: {nearest['label']} — "
        f"{nearest['value']} free space(s), ≈{nearest['distance_m']} m away "
        f"(~{nearest['walk_min_est']} min walk, straight-line estimate)."
    )
    return _result("find_nearest_parking", True, f"Parking near {location.label}", summary, items,
                   "offstreetparking", _freshness([it["timestamp"] for it in items]), location,
                   notes=[STRAIGHT_LINE_NOTE])


def plan_bike_trip(datasets: Mapping[str, pd.DataFrame], origin: Location, dest: Location, n: int = 3) -> Dict[str, Any]:
    df = datasets.get("bikestation", pd.DataFrame())
    needed = {"availableBikeNumber", "freeSlotNumber"}
    if not _has_coords(df) or not needed.issubset(df.columns):
        return _empty("plan_bike_trip", "VéloMagg bike-station data is unavailable right now.")
    work = df.copy()
    work["availableBikeNumber"] = _num(work["availableBikeNumber"])
    work["freeSlotNumber"] = _num(work["freeSlotNumber"])

    pickup = _with_distance(work[work["availableBikeNumber"] > 0], origin)
    dropoff = _with_distance(work[work["freeSlotNumber"] > 0], dest)
    if pickup.empty:
        return _empty("plan_bike_trip", f"No VéloMagg station with available bikes found near your start ({origin.label}).")
    if dropoff.empty:
        return _empty("plan_bike_trip", f"No VéloMagg station with a free dock found near your destination ({dest.label}).")

    pickup_items = [_item(r, "availableBikeNumber", ["streetAddress", "address", "name"], role="pickup")
                    for _, r in pickup.head(n).iterrows()]
    dropoff_items = [_item(r, "freeSlotNumber", ["streetAddress", "address", "name"], role="dropoff")
                     for _, r in dropoff.head(n).iterrows()]
    items = pickup_items + dropoff_items
    p0, d0 = pickup_items[0], dropoff_items[0]
    summary = (
        f"Bike trip {origin.label} → {dest.label}: pick up at {p0['label']} "
        f"({p0['value']} bike(s), ≈{p0['distance_m']} m from start), then return at {d0['label']} "
        f"({d0['value']} free dock(s), ≈{d0['distance_m']} m from your destination). "
        f"Both constraints are checked: a bike to take AND a free slot to return it."
    )
    return _result("plan_bike_trip", True, f"Bike trip: {origin.label} → {dest.label}", summary, items,
                   "bikestation", _freshness([it["timestamp"] for it in items]),
                   notes=[STRAIGHT_LINE_NOTE,
                          "Availability can change between now and your arrival; check again before returning the bike."])


def find_glass_container(datasets: Mapping[str, pd.DataFrame], location: Location, n: int = 3) -> Dict[str, Any]:
    df = datasets.get("wastecontainer", pd.DataFrame())
    if not _has_coords(df):
        return _empty("find_glass_container", "Glass waste-container data is unavailable right now.")
    status_fields = [c for c in ["fillingLevel", "alertLevel", "status", "containerVolume", "wasteType", "dateObserved"]
                     if c in df.columns]
    ranked = _with_distance(df, location)
    if ranked.empty:
        return _empty("find_glass_container", f"No glass container found near {location.label}.")

    has_fill = "fillingLevel" in df.columns and _num(df["fillingLevel"]).notna().any()
    items = [_item(r, "", ["name", "address", "streetAddress", "wasteType"], extra_fields=status_fields)
             for _, r in ranked.head(n).iterrows()]
    nearest = items[0]

    fields_text = ", ".join(status_fields) if status_fields else "none"
    summary = (
        f"The API reports the following container status fields: {fields_text}. "
        f"Nearest glass container to {location.label}: {nearest['label']} — "
        f"≈{nearest['distance_m']} m away (~{nearest['walk_min_est']} min walk, straight-line estimate)."
    )
    notes = [STRAIGHT_LINE_NOTE]
    if has_fill:
        fill = nearest["fields"].get("fillingLevel")
        summary += f" Reported filling level there is {fill} (lower means more space)."
    else:
        notes.append("This dataset does not expose a usable filling level, so available space cannot be confirmed.")
    return _result("find_glass_container", True, f"Glass containers near {location.label}", summary, items,
                   "wastecontainer", _freshness([it["timestamp"] for it in items]), location, notes=notes)


def summarize_city_now(datasets: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    bike = datasets.get("bikestation", pd.DataFrame())
    parking = datasets.get("offstreetparking", pd.DataFrame())
    eco = datasets.get("ecocounter", pd.DataFrame())

    parts: List[str] = []
    items: List[Dict[str, Any]] = []
    timestamps: List[Optional[str]] = []

    if not bike.empty and "availableBikeNumber" in bike.columns:
        bikes = _safe_int(_num(bike["availableBikeNumber"]).sum(min_count=1))
        free = _safe_int(_num(bike["freeSlotNumber"]).sum(min_count=1)) if "freeSlotNumber" in bike.columns else None
        parts.append(f"{bikes if bikes is not None else 'n/a'} bikes available across {len(bike)} VéloMagg stations"
                     + (f" ({free} free docks)" if free is not None else ""))
        items.append({"resource": "bikestation", "stations": len(bike), "available_bikes": bikes, "free_docks": free})
        if "availableBikeNumber_timestamp" in bike.columns:
            timestamps += bike["availableBikeNumber_timestamp"].dropna().astype(str).tolist()

    if not parking.empty and "availableSpotNumber" in parking.columns:
        spots = _safe_int(_num(parking["availableSpotNumber"]).sum(min_count=1))
        parts.append(f"{spots if spots is not None else 'n/a'} free car-park spaces across {len(parking)} parkings")
        items.append({"resource": "offstreetparking", "parkings": len(parking), "available_spaces": spots})
        if "availableSpotNumber_timestamp" in parking.columns:
            timestamps += parking["availableSpotNumber_timestamp"].dropna().astype(str).tolist()

    if not eco.empty and "intensity" in eco.columns:
        intensity = _num(eco["intensity"]).sum(min_count=1)
        intensity_val = round(float(intensity)) if pd.notna(intensity) else None
        parts.append(f"eco-counter activity sum {intensity_val if intensity_val is not None else 'n/a'}")
        items.append({"resource": "ecocounter", "counters": len(eco), "intensity_sum": intensity_val})
        if "TimeInstant" in eco.columns:
            timestamps += eco["TimeInstant"].dropna().astype(str).tolist()

    if not parts:
        return _empty("summarize_city_now", "No live data is available to summarize right now.")
    summary = "Live Montpellier snapshot: " + "; ".join(parts) + "."
    return _result("summarize_city_now", True, "City snapshot now", summary, items, "multiple",
                   _freshness(timestamps), notes=["Aggregated across all live stations/parkings/counters."])


def summarize_ecological_signal(datasets: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    bike = datasets.get("bikestation", pd.DataFrame())
    eco = datasets.get("ecocounter", pd.DataFrame())

    items: List[Dict[str, Any]] = []
    timestamps: List[Optional[str]] = []
    parts: List[str] = []

    if not bike.empty and "availableBikeNumber" in bike.columns:
        avail = _num(bike["availableBikeNumber"])
        with_bikes = int((avail > 0).sum())
        free = _num(bike["freeSlotNumber"]) if "freeSlotNumber" in bike.columns else None
        with_dock = int((free > 0).sum()) if free is not None else None
        parts.append(f"{with_bikes}/{len(bike)} stations currently have a bike to take"
                     + (f" and {with_dock} have a free dock to return one" if with_dock is not None else ""))
        items.append({"resource": "bikestation", "stations_with_bikes": with_bikes,
                      "stations_with_free_dock": with_dock, "stations": len(bike)})
        if "availableBikeNumber_timestamp" in bike.columns:
            timestamps += bike["availableBikeNumber_timestamp"].dropna().astype(str).tolist()

    if not eco.empty and "intensity" in eco.columns:
        intensity = _num(eco["intensity"]).sum(min_count=1)
        intensity_val = round(float(intensity)) if pd.notna(intensity) else None
        parts.append(f"current eco-counter activity sum is {intensity_val if intensity_val is not None else 'n/a'}")
        items.append({"resource": "ecocounter", "counters": len(eco), "intensity_sum": intensity_val})
        if "TimeInstant" in eco.columns:
            timestamps += eco["TimeInstant"].dropna().astype(str).tolist()

    if not parts:
        return _empty("summarize_ecological_signal", "No data is available to assess the ecological signal right now.")
    summary = (
        "Ecological signal: " + "; ".join(parts) + ". "
        "Cycling appears viable where both available bikes and free docks are sufficient — a low-carbon "
        "mobility opportunity. This summary does not estimate avoided car trips or CO2 saved; those require "
        "explicit assumptions you would need to enable."
    )
    return _result("summarize_ecological_signal", True, "Ecological signal", summary, items, "multiple",
                   _freshness(timestamps),
                   notes=["No avoided-car-trip or CO2 figure is implied; the app reports observed availability only."])


def explain_data_quality(datasets: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    all_timestamps: List[Optional[str]] = []
    warnings: List[str] = []

    for resource, df in datasets.items():
        if df.empty:
            items.append({"resource": resource, "rows": 0})
            warnings.append(f"{resource}: no rows returned.")
            continue
        ts_cols = [c for c in df.columns if c.endswith("_timestamp") or c in ("TimeInstant", "dateObserved")]
        ts_values: List[Optional[str]] = []
        for col in ts_cols:
            ts_values += df[col].dropna().astype(str).tolist()
        all_timestamps += ts_values
        fresh = _freshness(ts_values)
        missing_coords = 0
        if "latitude" in df.columns and "longitude" in df.columns:
            missing_coords = int(_num(df["latitude"]).isna().sum() + _num(df["longitude"]).isna().sum())
        entry = {
            "resource": resource,
            "rows": len(df),
            "missing_coordinates": missing_coords,
            "oldest": fresh["oldest_timestamp"],
            "stale": fresh["stale"],
        }
        items.append(entry)
        if missing_coords:
            warnings.append(f"{resource}: {missing_coords} missing coordinate value(s).")
        if fresh["stale"]:
            warnings.append(f"{resource}: readings look stale (oldest > 15 min).")

    summary = (
        "Data-quality report across "
        f"{len(items)} resource(s). "
        + (" ".join(warnings) if warnings else "No major freshness or coordinate issues detected.")
    )
    return _result("explain_data_quality", True, "Data quality", summary, items, "multiple",
                   _freshness(all_timestamps), notes=warnings)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def _resolve_param(params: Mapping[str, Any], text_key: str, lat_key: str, lon_key: str,
                   allow_nominatim: bool) -> Optional[Location]:
    lat, lon = params.get(lat_key), params.get(lon_key)
    if lat is not None and lon is not None:
        try:
            return Location(label=str(params.get(text_key) or "selected point"),
                            latitude=float(lat), longitude=float(lon), source="manual")
        except (TypeError, ValueError):
            pass
    text = (params.get(text_key) or "").strip()
    if not text:
        return None
    return resolve_location(text, allow_nominatim=allow_nominatim)


def run_intent(intent: str, params: Optional[Mapping[str, Any]], datasets: Mapping[str, pd.DataFrame],
               allow_nominatim: bool = False) -> Dict[str, Any]:
    """Resolve params and dispatch to the matching deterministic intent."""
    params = dict(params or {})

    if intent == "summarize_city_now":
        return summarize_city_now(datasets)
    if intent == "summarize_ecological_signal":
        return summarize_ecological_signal(datasets)
    if intent == "explain_data_quality":
        return explain_data_quality(datasets)

    if intent == "plan_bike_trip":
        origin = _resolve_param(params, "origin", "origin_latitude", "origin_longitude", allow_nominatim)
        dest = _resolve_param(params, "destination", "destination_latitude", "destination_longitude", allow_nominatim)
        if origin is None:
            return _need_location("plan_bike_trip", "start location")
        if dest is None:
            return _need_location("plan_bike_trip", "destination")
        return plan_bike_trip(datasets, origin, dest)

    if intent in {"find_nearest_bike_station", "find_nearest_parking", "find_glass_container"}:
        location = _resolve_param(params, "location", "latitude", "longitude", allow_nominatim)
        if location is None:
            return _need_location(intent)
        if intent == "find_nearest_bike_station":
            return find_nearest_bike_station(datasets, location)
        if intent == "find_nearest_parking":
            return find_nearest_parking(datasets, location)
        return find_glass_container(datasets, location)

    return _empty("unsupported", FALLBACK_MESSAGE)
