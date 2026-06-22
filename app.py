from __future__ import annotations

import json
from datetime import datetime, timedelta, time as dtime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_BASE_URL = "https://portail-api-data.montpellier3m.fr"
API_PORTAL_URL = "https://portail-api.montpellier3m.fr/"
OPENAPI_URL = "https://portail-api.montpellier3m.fr/doc/fiware3m.yaml"

RESOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "bikestation": {
        "label": "VéloMagg stations — live bike availability",
        "path": "/bikestation",
        "params": {"limit": 1000},
        "id_example": "urn:ngsi-ld:station:001",
        "main_columns": [
            "id",
            "streetAddress",
            "availableBikeNumber",
            "freeSlotNumber",
            "totalSlotNumber",
            "status",
            "latitude",
            "longitude",
            "availableBikeNumber_timestamp",
        ],
    },
    "offstreetparking": {
        "label": "Off-street parking — live available spaces",
        "path": "/offstreetparking",
        "params": {"limit": 1000},
        "id_example": "urn:ngsi-ld:parking:001",
        "main_columns": [
            "id",
            "name",
            "availableSpotNumber",
            "totalSpotNumber",
            "status",
            "latitude",
            "longitude",
            "availableSpotNumber_timestamp",
        ],
    },
    "parkingspaces": {
        "label": "Parking spaces — static/detail inventory",
        "path": "/parkingspaces",
        "params": {"limit": 1000, "offset": 0},
        "id_example": "urn:ngsi-ld:ParkingSpace:34172_ARCTRI",
        "main_columns": [
            "id",
            "name",
            "address",
            "typeOfUse",
            "isFree",
            "parkingSpaceNumber",
            "publicSpaces",
            "remainingSpaces",
            "latitude",
            "longitude",
        ],
    },
    "ecocounter": {
        "label": "Eco-counters — live bike/pedestrian counts",
        "path": "/ecocounter",
        "params": {"limit": 1000},
        "id_example": "urn:ngsi-ld:EcoCounter:XTH19101158",
        "main_columns": [
            "id",
            "deviceType",
            "vehicleType",
            "intensity",
            "laneId",
            "latitude",
            "longitude",
            "TimeInstant",
        ],
    },
    "wastecontainer": {
        "label": "Glass waste containers — filling level",
        "path": "/wastecontainer",
        "params": {},
        "id_example": "urn:ngsi-ld:WasteContainer:V_0510",
        "main_columns": [
            "id",
            "wasteType",
            "fillingLevel",
            "alertLevel",
            "containerVolume",
            "installationType",
            "status",
            "latitude",
            "longitude",
            "dateObserved",
        ],
    },
}

TIMESERIES_CONFIG: Dict[str, Dict[str, str]] = {
    "bikestation": {
        "label": "VéloMagg available bikes",
        "path_template": "/bikestation_timeseries/{entity_id}/attrs/availableBikeNumber",
        "value_name": "availableBikeNumber",
    },
    "offstreetparking": {
        "label": "Parking available spaces",
        "path_template": "/parking_timeseries/{entity_id}/attrs/availableSpotNumber",
        "value_name": "availableSpotNumber",
    },
    "ecocounter": {
        "label": "Eco-counter intensity",
        "path_template": "/ecocounter_timeseries/{entity_id}/attrs/intensity",
        "value_name": "intensity",
    },
}


def _clean_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _params_to_tuple(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in params.items() if v is not None and v != ""))


@st.cache_data(ttl=20, show_spinner=False)
def cached_get_json(
    base_url: str,
    path: str,
    params_tuple: Tuple[Tuple[str, Any], ...],
    timeout_seconds: int,
) -> Any:
    """Fetch JSON with a short TTL so the app remains close to live data."""
    url = f"{_clean_base_url(base_url)}/{path.lstrip('/')}"
    params = dict(params_tuple)
    response = requests.get(
        url,
        params=params,
        headers={"Accept": "application/json", "User-Agent": "montpellier-live-streamlit/0.1"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"The endpoint did not return valid JSON. URL: {response.url}") from exc


def api_get(base_url: str, path: str, params: Optional[Mapping[str, Any]], timeout_seconds: int) -> Any:
    return cached_get_json(base_url, path, _params_to_tuple(params or {}), timeout_seconds)


def ensure_entity_list(payload: Any) -> List[Dict[str, Any]]:
    """Normalize common API response envelopes into a list of entities."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "entities"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        return [payload]
    return []


def ngsi_value(obj: Any) -> Any:
    """Extract the value from NGSI-v2 normalized attributes when present."""
    if isinstance(obj, dict) and "value" in obj:
        return obj.get("value")
    return obj


def ngsi_timestamp(obj: Any) -> Optional[str]:
    """Extract common Fiware timestamp metadata patterns."""
    if not isinstance(obj, dict):
        return None
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("timestamp", "TimeInstant", "dateObserved"):
        candidate = metadata.get(key)
        if isinstance(candidate, dict) and "value" in candidate:
            return str(candidate.get("value"))
    return None


def _flatten_nested(prefix: str, value: Mapping[str, Any], out: Dict[str, Any]) -> None:
    """Flatten a shallow nested dict; store complex values as JSON strings."""
    for key, val in value.items():
        column = key if prefix in {"address", "name"} else f"{prefix}_{key}"
        if isinstance(val, (dict, list)):
            out[column] = json.dumps(val, ensure_ascii=False)
        else:
            out[column] = val


def flatten_entity(entity: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one Fiware entity into a tabular row."""
    row: Dict[str, Any] = {
        "id": entity.get("id"),
        "type": entity.get("type"),
    }

    for attr, wrapped_value in entity.items():
        if attr in {"id", "type"}:
            continue

        value = ngsi_value(wrapped_value)
        timestamp = ngsi_timestamp(wrapped_value)
        if timestamp:
            row[f"{attr}_timestamp"] = timestamp

        if attr == "location" and isinstance(value, dict):
            coords = value.get("coordinates")
            if isinstance(coords, list) and len(coords) >= 2:
                try:
                    row["longitude"] = float(coords[0])
                    row["latitude"] = float(coords[1])
                except (TypeError, ValueError):
                    row["longitude"] = coords[0]
                    row["latitude"] = coords[1]
            row["location_type"] = value.get("type")
        elif isinstance(value, dict):
            _flatten_nested(attr, value, row)
        elif isinstance(value, list):
            row[attr] = ", ".join(map(str, value))
        else:
            row[attr] = value

    return row


def entities_to_dataframe(payload: Any) -> pd.DataFrame:
    rows = [flatten_entity(entity) for entity in ensure_entity_list(payload)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.lower() in {"latitude", "longitude"} or col.endswith("Number") or col in {
            "intensity",
            "fillingLevel",
            "parkingSpaceNumber",
            "publicSpaces",
            "remainingSpaces",
        }:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def preferred_columns(df: pd.DataFrame, resource_key: str) -> List[str]:
    configured = RESOURCE_CONFIG[resource_key].get("main_columns", [])
    visible = [col for col in configured if col in df.columns]
    leftovers = [col for col in df.columns if col not in visible]
    return visible + leftovers


def make_csv_download(df: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def safe_sum(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return None
    return float(series.sum())


def safe_mean(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return None
    return float(series.mean())


def metric_value(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}{suffix}"
    return f"{value:.1f}{suffix}"


def fetch_resource(base_url: str, resource_key: str, limit: int, timeout_seconds: int) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[str]]:
    config = RESOURCE_CONFIG[resource_key]
    params = dict(config.get("params", {}))
    if "limit" in params:
        params["limit"] = limit
    try:
        payload = api_get(base_url, config["path"], params, timeout_seconds)
        return entities_to_dataframe(payload), payload, None
    except Exception as exc:  # deliberately visible in Streamlit UI
        return None, None, str(exc)


def build_map_dataframe(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for key, df in datasets.items():
        if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            continue
        map_df = df[["latitude", "longitude", "id"]].copy()
        map_df["resource"] = key
        map_df["label"] = RESOURCE_CONFIG[key]["label"]
        rows.append(map_df)
    if not rows:
        return pd.DataFrame(columns=["latitude", "longitude", "id", "resource", "label"])
    out = pd.concat(rows, ignore_index=True)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    return out.dropna(subset=["latitude", "longitude"])


def timeseries_to_dataframe(payload: Any, default_value_name: str) -> pd.DataFrame:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        # Some Fiware time-series APIs return a list with one object per attribute.
        frames = [timeseries_to_dataframe(item, default_value_name) for item in payload]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not isinstance(payload, dict):
        return pd.DataFrame()

    index = payload.get("index") or payload.get("timestamps") or payload.get("time")
    values = payload.get("values") or payload.get("value")
    value_name = payload.get("attrName") or default_value_name or "value"

    if isinstance(index, list) and isinstance(values, list):
        df = pd.DataFrame({"timestamp": pd.to_datetime(index, errors="coerce"), value_name: values})
        df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
        return df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Fallback for unusual dict shapes: show as key/value table.
    return pd.DataFrame([flatten_entity(payload)])


def format_entity_label(df: pd.DataFrame, entity_id: str) -> str:
    if df.empty or "id" not in df.columns:
        return entity_id
    row = df.loc[df["id"] == entity_id]
    if row.empty:
        return entity_id
    candidate_cols = ["name", "streetAddress", "address", "deviceType", "vehicleType"]
    label_parts = [str(row.iloc[0][col]) for col in candidate_cols if col in row.columns and pd.notna(row.iloc[0][col])]
    return f"{' — '.join(label_parts)} ({entity_id})" if label_parts else entity_id


def render_status_cards(datasets: Mapping[str, pd.DataFrame]) -> None:
    bike = datasets.get("bikestation", pd.DataFrame())
    parking = datasets.get("offstreetparking", pd.DataFrame())
    eco = datasets.get("ecocounter", pd.DataFrame())
    waste = datasets.get("wastecontainer", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VéloMagg stations", len(bike) if not bike.empty else "n/a")
    c1.metric("Available bikes", metric_value(safe_sum(bike, "availableBikeNumber")))

    c2.metric("Parkings", len(parking) if not parking.empty else "n/a")
    c2.metric("Available parking spots", metric_value(safe_sum(parking, "availableSpotNumber")))

    c3.metric("Eco-counter rows", len(eco) if not eco.empty else "n/a")
    c3.metric("Current intensity sum", metric_value(safe_sum(eco, "intensity")))

    c4.metric("Waste containers", len(waste) if not waste.empty else "n/a")
    c4.metric("Mean filling level", metric_value(safe_mean(waste, "fillingLevel"), "%"))


def render_dashboard(datasets: Mapping[str, pd.DataFrame], errors: Mapping[str, str]) -> None:
    st.subheader("Live overview")
    render_status_cards(datasets)

    if errors:
        with st.expander("Endpoint errors", expanded=True):
            for key, message in errors.items():
                st.error(f"{RESOURCE_CONFIG[key]['label']}: {message}")

    map_df = build_map_dataframe(datasets)
    if not map_df.empty:
        st.subheader("Geographic coverage")
        st.map(map_df[["latitude", "longitude"]], zoom=12)
        with st.expander("Map source rows"):
            st.dataframe(map_df, use_container_width=True)


def render_data_explorer(datasets: Mapping[str, pd.DataFrame]) -> None:
    st.subheader("Data explorer")
    resource_key = st.selectbox(
        "Resource",
        list(RESOURCE_CONFIG.keys()),
        format_func=lambda key: RESOURCE_CONFIG[key]["label"],
    )
    df = datasets.get(resource_key, pd.DataFrame())
    if df.empty:
        st.warning("No tabular data available for this resource.")
        return

    search_text = st.text_input("Filter rows by text", "")
    shown = df.copy()
    if search_text.strip():
        mask = shown.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False)).any(axis=1)
        shown = shown.loc[mask]

    shown = shown[preferred_columns(shown, resource_key)]
    st.caption(f"Rows: {len(shown)} | Columns: {len(shown.columns)}")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    make_csv_download(shown, "Download visible table as CSV", f"montpellier_{resource_key}.csv")


def render_timeseries(base_url: str, timeout_seconds: int, limit: int) -> None:
    st.subheader("Historical time series")
    st.caption("The OpenAPI file documents historical endpoints for VéloMagg, off-street parking, and eco-counter intensity.")

    resource_key = st.selectbox(
        "Historical resource",
        list(TIMESERIES_CONFIG.keys()),
        format_func=lambda key: TIMESERIES_CONFIG[key]["label"],
    )

    df, _, error = fetch_resource(base_url, resource_key, limit, timeout_seconds)
    if error or df is None or df.empty or "id" not in df.columns:
        st.error(f"Cannot load entity list for this resource: {error or 'empty response'}")
        return

    entity_id = st.selectbox(
        "Entity ID",
        sorted(df["id"].dropna().astype(str).unique()),
        format_func=lambda eid: format_entity_label(df, eid),
    )

    today = datetime.now().date()
    start_date = st.date_input("From date", today - timedelta(days=1))
    end_date = st.date_input("To date", today)
    start_time = st.time_input("From time", dtime(0, 0, 0))
    end_time = st.time_input("To time", dtime(23, 59, 59))

    from_dt = datetime.combine(start_date, start_time).isoformat(timespec="seconds")
    to_dt = datetime.combine(end_date, end_time).isoformat(timespec="seconds")

    if st.button("Fetch time series", type="primary"):
        encoded_id = quote(entity_id, safe=":")
        ts_cfg = TIMESERIES_CONFIG[resource_key]
        path = ts_cfg["path_template"].format(entity_id=encoded_id)
        try:
            payload = api_get(base_url, path, {"fromDate": from_dt, "toDate": to_dt}, timeout_seconds)
            ts_df = timeseries_to_dataframe(payload, ts_cfg["value_name"])
            if ts_df.empty:
                st.warning("The endpoint returned no time-series rows for this interval.")
                st.json(payload)
                return
            st.dataframe(ts_df, use_container_width=True, hide_index=True)
            value_cols = [col for col in ts_df.columns if col != "timestamp"]
            if "timestamp" in ts_df.columns and value_cols:
                chart_df = ts_df.set_index("timestamp")[value_cols]
                st.line_chart(chart_df)
            make_csv_download(ts_df, "Download time series as CSV", f"montpellier_{resource_key}_timeseries.csv")
            with st.expander("Raw time-series JSON"):
                st.json(payload)
        except Exception as exc:
            st.error(str(exc))


def render_raw_json(base_url: str, timeout_seconds: int, limit: int) -> None:
    st.subheader("Raw JSON")
    resource_key = st.selectbox(
        "Resource for raw fetch",
        list(RESOURCE_CONFIG.keys()),
        format_func=lambda key: RESOURCE_CONFIG[key]["label"],
        key="raw_resource",
    )
    config = RESOURCE_CONFIG[resource_key]
    entity_id = st.text_input("Optional entity ID", "", placeholder=config["id_example"])
    params = dict(config.get("params", {}))
    if "limit" in params:
        params["limit"] = limit
    if entity_id.strip():
        params["id"] = entity_id.strip()

    st.code(f"GET {base_url.rstrip('/')}{config['path']}?" + "&".join(f"{k}={v}" for k, v in params.items()), language="text")
    if st.button("Fetch raw JSON"):
        try:
            payload = api_get(base_url, config["path"], params, timeout_seconds)
            st.json(payload)
        except Exception as exc:
            st.error(str(exc))


def render_docs() -> None:
    st.subheader("Documented API resources")
    docs_rows = []
    for key, cfg in RESOURCE_CONFIG.items():
        docs_rows.append(
            {
                "resource": key,
                "label": cfg["label"],
                "path": cfg["path"],
                "example_id": cfg["id_example"],
            }
        )
    st.dataframe(pd.DataFrame(docs_rows), use_container_width=True, hide_index=True)

    ts_rows = []
    for key, cfg in TIMESERIES_CONFIG.items():
        ts_rows.append(
            {
                "resource": key,
                "label": cfg["label"],
                "path_template": cfg["path_template"],
            }
        )
    st.dataframe(pd.DataFrame(ts_rows), use_container_width=True, hide_index=True)

    st.markdown(
        f"API portal: {API_PORTAL_URL}\n\n"
        f"OpenAPI YAML: {OPENAPI_URL}\n\n"
        "Note: the waste-container time-series endpoint is present only as a commented section in the YAML, so this app does not treat it as a stable endpoint."
    )


def main() -> None:
    st.set_page_config(page_title="Montpellier live open data", layout="wide")
    st.title("Montpellier live open-data dashboard")
    st.caption("Fiware / NGSI-style live data from Montpellier Méditerranée Métropole API portal.")

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("API base URL", DEFAULT_BASE_URL)
        timeout_seconds = st.slider("Request timeout, seconds", 3, 60, 15)
        limit = st.slider("Rows per resource", 10, 1000, 1000, step=10)
        st.divider()
        if st.button("Clear cache and refresh"):
            cached_get_json.clear()
            st.rerun()
        auto_refresh = st.checkbox("Auto-refresh page")
        refresh_seconds = st.slider("Refresh interval, seconds", 15, 600, 60, step=15, disabled=not auto_refresh)
        st.divider()
        st.markdown(f"[API portal]({API_PORTAL_URL})")
        st.markdown(f"[OpenAPI YAML]({OPENAPI_URL})")

    fetched: Dict[str, pd.DataFrame] = {}
    raw_payloads: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    with st.spinner("Fetching live resources..."):
        for key in RESOURCE_CONFIG:
            df, payload, error = fetch_resource(base_url, key, limit, timeout_seconds)
            if error:
                errors[key] = error
                fetched[key] = pd.DataFrame()
            else:
                fetched[key] = df if df is not None else pd.DataFrame()
                raw_payloads[key] = payload

    tab_dashboard, tab_explorer, tab_timeseries, tab_raw, tab_docs = st.tabs(
        ["Dashboard", "Data explorer", "Time series", "Raw JSON", "API docs"]
    )
    with tab_dashboard:
        render_dashboard(fetched, errors)
    with tab_explorer:
        render_data_explorer(fetched)
    with tab_timeseries:
        render_timeseries(base_url, timeout_seconds, limit)
    with tab_raw:
        render_raw_json(base_url, timeout_seconds, limit)
    with tab_docs:
        render_docs()

    st.caption(f"Last Streamlit page render: {datetime.now().isoformat(timespec='seconds')}")

    if auto_refresh:
        components.html(
            f"""
            <script>
            setTimeout(function() {{ window.parent.location.reload(); }}, {int(refresh_seconds) * 1000});
            </script>
            """,
            height=0,
        )


if __name__ == "__main__":
    main()
