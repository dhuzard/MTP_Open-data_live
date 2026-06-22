"""Deterministic "Public & Ecological Insights" for the Montpellier live open-data app.

This module is intentionally LLM-free and network-free. It derives simple,
defensible signals from whatever live DataFrames are passed in and renders them
with cautious, honest wording.

Design rules honoured here:
- Never assume a column exists. Inspect ``df.columns`` and only interpret what
  is actually present (this matters most for waste containers).
- Never crash. Every signal degrades to ``None`` (or an empty structure) when
  the underlying data is missing, empty, or malformed.
- Never fabricate ecological "impact" numbers. Avoided car trips / CO2 saved are
  shown only as a clearly-labelled hypothetical, and only when the user opts in.

Public interface:
    compute_signals(datasets: dict) -> dict      # pure, testable, no streamlit
    render_insights_tab(datasets: dict) -> None   # streamlit rendering
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

# Reading older than this many seconds => we flag the resource as "stale".
STALENESS_THRESHOLD_SECONDS: int = 15 * 60  # 15 minutes

# Columns that may carry a timestamp, per resource. We try each in order and use
# whichever one is actually present in the DataFrame. The generic fallbacks at
# the end are tried for any resource.
TIMESTAMP_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "bikestation": ["availableBikeNumber_timestamp"],
    "offstreetparking": ["availableSpotNumber_timestamp"],
    "ecocounter": ["TimeInstant"],
    "wastecontainer": ["dateObserved"],
}
GENERIC_TIMESTAMP_CANDIDATES: List[str] = [
    "TimeInstant",
    "dateObserved",
    "dateModified",
    "observedAt",
]


# ---------------------------------------------------------------------------
# Small, defensive helpers (no streamlit).
# ---------------------------------------------------------------------------
def _as_dataframe(datasets: Mapping[str, Any], key: str) -> pd.DataFrame:
    """Return ``datasets[key]`` as a DataFrame, or an empty one if absent/invalid."""
    value = datasets.get(key) if isinstance(datasets, Mapping) else None
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _numeric_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    """Return a numeric series for ``column`` with non-numeric coerced to NaN.

    Returns ``None`` when the column is missing or has no numeric values at all.
    """
    if df.empty or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return None
    return series


def _safe_sum(df: pd.DataFrame, column: str) -> Optional[float]:
    series = _numeric_series(df, column)
    if series is None:
        return None
    return float(series.sum())


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Bounded ratio numerator/denominator, or ``None`` when not computable."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _timestamp_columns(df: pd.DataFrame, resource: str) -> List[str]:
    """Which candidate timestamp columns are actually present in ``df``."""
    present: List[str] = []
    for candidate in TIMESTAMP_COLUMN_CANDIDATES.get(resource, []) + GENERIC_TIMESTAMP_CANDIDATES:
        if candidate in df.columns and candidate not in present:
            present.append(candidate)
    return present


def _parse_timestamps(df: pd.DataFrame, resource: str) -> pd.Series:
    """Return a UTC-aware datetime series gathered from any timestamp column.

    All present timestamp columns are stacked together so the oldest/newest
    reflect the whole resource. Unparseable values become NaT and are dropped.
    """
    columns = _timestamp_columns(df, resource)
    if not columns:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    frames: List[pd.Series] = []
    for column in columns:
        parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
        frames.append(parsed)
    stacked = pd.concat(frames, ignore_index=True)
    return stacked.dropna()


# ---------------------------------------------------------------------------
# Individual signal computations.
# ---------------------------------------------------------------------------
def _bike_availability(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "stations": 0,
            "stations_with_bikes": 0,
            "total_available_bikes": None,
            "total_capacity": None,
            "fill_ratio": None,
        }

    stations = int(len(df))

    available = _numeric_series(df, "availableBikeNumber")
    if available is not None:
        stations_with_bikes = int((available.fillna(0) > 0).sum())
        total_available = float(available.sum())
    else:
        stations_with_bikes = 0
        total_available = None

    total_capacity = _safe_sum(df, "totalSlotNumber")
    fill_ratio = _ratio(total_available, total_capacity)

    return {
        "stations": stations,
        "stations_with_bikes": stations_with_bikes,
        "total_available_bikes": total_available,
        "total_capacity": total_capacity,
        "fill_ratio": fill_ratio,
    }


def _free_docks(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"total_free_slots": None, "stations_with_free_dock": 0}

    free = _numeric_series(df, "freeSlotNumber")
    if free is None:
        return {"total_free_slots": None, "stations_with_free_dock": 0}

    return {
        "total_free_slots": float(free.sum()),
        "stations_with_free_dock": int((free.fillna(0) > 0).sum()),
    }


def _parking_pressure(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "parkings": 0,
            "total_available": None,
            "total_capacity": None,
            "occupancy_ratio": None,
        }

    total_available = _safe_sum(df, "availableSpotNumber")
    total_capacity = _safe_sum(df, "totalSpotNumber")

    free_ratio = _ratio(total_available, total_capacity)
    occupancy_ratio = None if free_ratio is None else max(0.0, min(1.0, 1.0 - free_ratio))

    return {
        "parkings": int(len(df)),
        "total_available": total_available,
        "total_capacity": total_capacity,
        "occupancy_ratio": occupancy_ratio,
    }


def _eco_activity(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"counters": 0, "total_intensity": None, "by_vehicle_type": {}}

    total_intensity = _safe_sum(df, "intensity")

    by_vehicle_type: Dict[str, float] = {}
    if "vehicleType" in df.columns and "intensity" in df.columns:
        intensity = pd.to_numeric(df["intensity"], errors="coerce")
        grouped = (
            pd.DataFrame({"vehicleType": df["vehicleType"].astype("string"), "intensity": intensity})
            .dropna(subset=["intensity"])
            .groupby("vehicleType", dropna=True)["intensity"]
            .sum()
        )
        by_vehicle_type = {str(key): float(value) for key, value in grouped.items()}

    return {
        "counters": int(len(df)),
        "total_intensity": total_intensity,
        "by_vehicle_type": by_vehicle_type,
    }


def _freshness(datasets: Mapping[str, Any], now_utc: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
    freshness: Dict[str, Dict[str, Any]] = {}
    for resource in datasets:
        df = _as_dataframe(datasets, resource)
        entry: Dict[str, Any] = {
            "oldest": None,
            "newest": None,
            "age_seconds": None,
            "stale": None,
        }
        if not df.empty:
            timestamps = _parse_timestamps(df, resource)
            if not timestamps.empty:
                oldest = timestamps.min()
                newest = timestamps.max()
                age_seconds = float((now_utc - oldest).total_seconds())
                entry = {
                    "oldest": oldest.isoformat(),
                    "newest": newest.isoformat(),
                    "age_seconds": age_seconds,
                    "stale": bool(age_seconds > STALENESS_THRESHOLD_SECONDS),
                }
        freshness[resource] = entry
    return freshness


def _count_missing_coordinates(df: pd.DataFrame) -> int:
    """How many rows lack a usable latitude/longitude pair."""
    if df.empty:
        return 0
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return int(len(df))
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    return int((lat.isna() | lon.isna()).sum())


def _warnings(
    datasets: Mapping[str, Any],
    freshness: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    warnings: List[str] = []

    for resource in datasets:
        df = _as_dataframe(datasets, resource)
        if df.empty:
            continue

        missing_coords = _count_missing_coordinates(df)
        if missing_coords > 0:
            warnings.append(f"{resource}: {missing_coords} rows missing coordinates")

        info = freshness.get(resource, {})
        if info.get("stale"):
            warnings.append(f"{resource}: data appears stale (oldest reading > 15 min)")

    # Waste containers: be explicit when the fill signal is not actually exposed.
    waste = _as_dataframe(datasets, "wastecontainer")
    if not waste.empty:
        if "fillingLevel" not in waste.columns:
            warnings.append("wastecontainer: no fillingLevel column exposed; cannot assess fill")
        elif _numeric_series(waste, "fillingLevel") is None:
            warnings.append("wastecontainer: fillingLevel column present but has no numeric values; cannot assess fill")

    return warnings


# ---------------------------------------------------------------------------
# Public, pure signal computation.
# ---------------------------------------------------------------------------
def compute_signals(datasets: dict) -> dict:
    """Compute deterministic public/ecological signals from live DataFrames.

    Pure and side-effect-free: no streamlit, no network, never raises. Missing
    data yields ``None`` (or empty collections) rather than an exception.
    """
    safe_datasets: Mapping[str, Any] = datasets if isinstance(datasets, Mapping) else {}

    now_utc = pd.Timestamp.now(tz="UTC")

    bike = _as_dataframe(safe_datasets, "bikestation")
    parking = _as_dataframe(safe_datasets, "offstreetparking")
    eco = _as_dataframe(safe_datasets, "ecocounter")

    freshness = _freshness(safe_datasets, now_utc)

    return {
        "bike_availability": _bike_availability(bike),
        "free_docks": _free_docks(bike),
        "parking_pressure": _parking_pressure(parking),
        "eco_activity": _eco_activity(eco),
        "freshness": freshness,
        "warnings": _warnings(safe_datasets, freshness),
    }


# ---------------------------------------------------------------------------
# Streamlit rendering helpers.
# ---------------------------------------------------------------------------
def _fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value)):,}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _has_any_data(datasets: Mapping[str, Any]) -> bool:
    for resource in datasets:
        df = _as_dataframe(datasets, resource)
        if not df.empty:
            return True
    return False


def render_insights_tab(datasets: dict) -> None:
    """Render the deterministic insights tab using streamlit.

    Safe to call with empty/partial data: it renders a friendly notice rather
    than raising.
    """
    import streamlit as st

    st.subheader("Public & Ecological Insights")
    st.caption(
        "Deterministic signals computed directly from the live data. "
        "No language model, no external API, no fabricated estimates."
    )

    safe_datasets: Mapping[str, Any] = datasets if isinstance(datasets, Mapping) else {}

    if not _has_any_data(safe_datasets):
        st.info("No live data available right now.")
        return

    signals = compute_signals(dict(safe_datasets))

    bike = signals["bike_availability"]
    free_docks = signals["free_docks"]
    parking = signals["parking_pressure"]
    eco = signals["eco_activity"]

    # --- Core signal 1: bike availability + free docks ---------------------
    st.markdown("### Bike availability (VéloMagg)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Stations", _fmt_int(bike["stations"]))
    b2.metric("Stations with bikes", _fmt_int(bike["stations_with_bikes"]))
    b3.metric("Available bikes", _fmt_int(bike["total_available_bikes"]))
    b4.metric("Free docks (returns)", _fmt_int(free_docks["total_free_slots"]))

    if bike["fill_ratio"] is not None:
        st.caption(
            f"Network fill ratio (available bikes / total slots): {_fmt_pct(bike['fill_ratio'])} "
            f"of {_fmt_int(bike['total_capacity'])} slots."
        )
        st.progress(min(1.0, max(0.0, bike["fill_ratio"])))
    else:
        st.caption("Fill ratio unavailable: total slot capacity not exposed for these stations.")

    if free_docks["total_free_slots"] is not None:
        st.caption(
            f"{_fmt_int(free_docks['stations_with_free_dock'])} stations currently have at least one "
            "free dock, so a bike can be returned there."
        )
    else:
        st.caption("Free-dock counts unavailable: no freeSlotNumber field exposed.")

    # --- Core signal 2: parking pressure -----------------------------------
    st.markdown("### Parking pressure (off-street)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Parkings", _fmt_int(parking["parkings"]))
    p2.metric("Available spaces", _fmt_int(parking["total_available"]))
    p3.metric("Occupancy", _fmt_pct(parking["occupancy_ratio"]))

    if parking["occupancy_ratio"] is not None:
        st.caption(
            f"Occupancy estimated as 1 - available/total over {_fmt_int(parking['total_capacity'])} "
            "known spaces."
        )
        st.progress(min(1.0, max(0.0, parking["occupancy_ratio"])))
    else:
        st.caption("Occupancy unavailable: available and/or total spaces not exposed.")

    # --- Core signal 3: eco activity ---------------------------------------
    st.markdown("### Eco-counter activity")
    e1, e2 = st.columns(2)
    e1.metric("Counters", _fmt_int(eco["counters"]))
    e2.metric("Total intensity (sum of latest readings)", _fmt_int(eco["total_intensity"]))

    if eco["by_vehicle_type"]:
        by_type_df = (
            pd.DataFrame(
                {
                    "vehicleType": list(eco["by_vehicle_type"].keys()),
                    "summed_intensity": list(eco["by_vehicle_type"].values()),
                }
            )
            .sort_values("summed_intensity", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(by_type_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Per-vehicle-type breakdown unavailable: no vehicleType/intensity fields exposed.")
    st.caption(
        "Intensity is the counter's reported flow for its latest interval; it is a momentary "
        "reading, not a cumulative daily total."
    )

    # --- Waste containers: only interpret what is actually exposed ---------
    _render_waste_section(st, safe_datasets)

    # --- Data freshness ----------------------------------------------------
    _render_freshness_section(st, signals["freshness"])

    # --- Data-quality warnings --------------------------------------------
    st.markdown("### Data-quality warnings")
    if signals["warnings"]:
        for message in signals["warnings"]:
            st.warning(message)
    else:
        st.caption("No data-quality warnings detected.")

    # --- Cautious ecological interpretation + opt-in hypothetical ----------
    _render_ecological_interpretation(st, bike, free_docks)


def _render_waste_section(st: Any, datasets: Mapping[str, Any]) -> None:
    waste = _as_dataframe(datasets, "wastecontainer")
    if waste.empty:
        return

    st.markdown("### Waste containers")

    fill_series = _numeric_series(waste, "fillingLevel")
    if fill_series is not None:
        # fillingLevel may be a 0-1 ratio or a 0-100 percentage; normalise to %.
        scaled = fill_series * 100.0 if fill_series.max() <= 1.0 else fill_series
        mean_fill = float(scaled.mean())
        st.metric("Mean filling level", f"{mean_fill:.0f}%")
        st.caption(
            f"Computed from {int(fill_series.notna().sum())} containers reporting a numeric "
            "fillingLevel. Containers without a reading are excluded."
        )
    else:
        # Be explicit about what *is* exposed instead of guessing about fill.
        status_like = [
            col
            for col in ("wasteType", "alertLevel", "containerVolume", "status", "dateObserved")
            if col in waste.columns
        ]
        if status_like:
            st.caption(
                "The API reports the following container status fields: "
                + ", ".join(status_like)
                + ". Filling level is not exposed, so no fill assessment is made."
            )
            for col in ("status", "alertLevel", "wasteType"):
                if col in waste.columns:
                    counts = waste[col].astype("string").value_counts(dropna=True)
                    if not counts.empty:
                        st.caption(
                            f"{col}: "
                            + ", ".join(f"{idx} ({int(val)})" for idx, val in counts.items())
                        )
        else:
            st.caption(
                "No recognised container status fields are exposed for these containers; "
                "nothing to interpret."
            )


def _render_freshness_section(st: Any, freshness: Mapping[str, Mapping[str, Any]]) -> None:
    st.markdown("### Data freshness")
    rows: List[Dict[str, Any]] = []
    for resource, info in freshness.items():
        age = info.get("age_seconds")
        rows.append(
            {
                "resource": resource,
                "oldest": info.get("oldest") or "n/a",
                "newest": info.get("newest") or "n/a",
                "age (min)": "n/a" if age is None else round(age / 60.0, 1),
                "stale": "unknown" if info.get("stale") is None else bool(info.get("stale")),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("\"Stale\" means the oldest reading in a resource is older than 15 minutes.")


def _render_ecological_interpretation(
    st: Any,
    bike: Mapping[str, Any],
    free_docks: Mapping[str, Any],
) -> None:
    st.markdown("### Ecological interpretation")
    st.write(
        "Cycling appears viable in areas where both available bikes and free docks are "
        "sufficient — a low-carbon mobility opportunity. This app does NOT estimate avoided "
        "car trips or CO2 saved unless you enable assumptions below."
    )

    with st.expander("Enable hypothetical assumptions (off by default)"):
        st.caption(
            "Anything below is a what-if illustration based on a number YOU choose. "
            "It is not measured, and it is not a claim about real-world impact."
        )
        enabled = st.checkbox("Enable hypothetical assumptions", value=False)
        grams_per_km = st.number_input(
            "Assumed CO2 grams per car-km avoided (your assumption)",
            min_value=0.0,
            value=120.0,
            step=10.0,
        )
        assumed_km = st.number_input(
            "Assumed km per bike trip (your assumption)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )

        if not enabled:
            st.caption("Assumptions are disabled. No hypothetical figures are shown.")
            return

        available = bike.get("total_available_bikes")
        if available is None:
            st.caption(
                "Cannot build a hypothetical: the number of available bikes is not exposed "
                "by the live data right now."
            )
            return

        # Deliberately conservative framing: this is purely the user's arithmetic,
        # using the currently available bikes as a stand-in for "possible trips".
        hypothetical_grams = float(available) * assumed_km * grams_per_km
        st.warning(
            "HYPOTHETICAL ONLY — not a measurement. If each of the "
            f"{_fmt_int(available)} currently available bikes were ridden once for "
            f"{assumed_km:g} km in place of a car emitting {grams_per_km:g} g CO2/km, "
            f"that would correspond to roughly {hypothetical_grams / 1000.0:,.1f} kg CO2 "
            "under YOUR assumptions. This figure depends entirely on the inputs above and "
            "does not represent observed behaviour or avoided trips."
        )
