"""Module 2 - Station finding using the India charging-station dataset."""
from __future__ import annotations
import pandas as pd
import config
from .geo import haversine_km

# Accepts many column spellings from real-world datasets (Kaggle / OCM / data.gov.in)
ALIASES = {
    "station_id": ["station_id", "id", "uid"],
    "name": ["name", "station_name", "title"],
    "latitude": ["latitude", "lat", "lattitude"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "city": ["city", "town"],
    "state": ["state"],
    "operator": ["operator", "network", "owner"],
    "power_kw": ["power_kw", "power", "max_power_kw", "kw"],
    "n_chargers": ["n_chargers", "chargers", "num_chargers", "ports", "no_of_ports"],
    "price_per_kwh": ["price_per_kwh", "tariff", "cost_per_kwh", "price"],
    "current_load_pct": ["current_load_pct", "load", "occupancy"],
    "avg_session_min": ["avg_session_min", "avg_session"],
    "connector_type": ["connector_type", "connector", "plug", "type"],
}
DEFAULTS = {"city": "", "state": "", "operator": "Unknown", "power_kw": 30.0, "n_chargers": 2,
            "price_per_kwh": 15.0, "current_load_pct": 40.0, "avg_session_min": 40.0,
            "connector_type": "CCS2"}


def load_stations(path=config.STATIONS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for std, names in ALIASES.items():
        for n in names:
            if n in low:
                out[std] = df[low[n]]
                break
    for must in ("latitude", "longitude"):
        if must not in out:
            raise ValueError(f"Station dataset needs a '{must}' column (found {list(df.columns)})")
    if "station_id" not in out:
        out["station_id"] = [f"ST{i:05d}" for i in range(len(out))]
    if "name" not in out:
        out["name"] = out["station_id"]
    for col, default in DEFAULTS.items():
        if col not in out:
            out[col] = default
    for col in ("latitude", "longitude", "power_kw", "n_chargers", "price_per_kwh",
                "current_load_pct", "avg_session_min"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"])
    out = out[out.latitude.between(6, 38) & out.longitude.between(68, 98)]   # India bounding box
    for col in ("power_kw", "n_chargers", "price_per_kwh", "current_load_pct", "avg_session_min"):
        out[col] = out[col].fillna(DEFAULTS[col])
    out["n_chargers"] = out["n_chargers"].clip(lower=1).astype(int)
    return out.reset_index(drop=True)


def find_nearby(stations: pd.DataFrame, lat: float, lon: float,
                radius_km: float = config.SEARCH_RADIUS_KM,
                road_factor: float = config.ROAD_FACTOR,
                connector: str | None = None) -> pd.DataFrame:
    """Add straight-line + estimated road distance, keep stations inside radius."""
    df = stations.copy()
    df["straight_km"] = haversine_km(lat, lon, df.latitude.values, df.longitude.values)
    df["distance_km"] = df["straight_km"] * road_factor
    df = df[df.straight_km <= radius_km]
    if connector:
        df = df[df.connector_type.astype(str).str.contains(connector, case=False, na=False)]
    return df.sort_values("distance_km").reset_index(drop=True)
