"""Module 3 - Reachability check + low-battery range constraint."""
from __future__ import annotations
import pandas as pd
import config
from .ev_data import EVState


def energy_required_kwh(distance_km, ev: EVState):
    """Battery energy needed to drive `distance_km`."""
    return distance_km * ev.consumption_kwh_per_km / ev.efficiency


def soc_at_arrival(distance_km, ev: EVState):
    return ev.soc - energy_required_kwh(distance_km, ev) / ev.battery_kwh * 100.0


def filter_reachable(df: pd.DataFrame, ev: EVState,
                     min_arrival_soc: float = config.MIN_ARRIVAL_SOC) -> pd.DataFrame:
    """Hard constraint: keep ONLY stations the car can reach with the battery it has now
    and still arrive with >= min_arrival_soc %. E.g. at 10% SOC only stations inside the
    remaining range survive; everything farther is removed."""
    df = df.copy()
    df["energy_to_station_kwh"] = energy_required_kwh(df["distance_km"], ev)
    df["soc_arrival"] = soc_at_arrival(df["distance_km"], ev)
    return df[df["soc_arrival"] >= min_arrival_soc].reset_index(drop=True)


def range_summary(ev: EVState, min_arrival_soc: float = config.MIN_ARRIVAL_SOC) -> dict:
    return {
        "current_soc_pct": ev.soc,
        "full_range_km": round(ev.max_range_km(0), 1),
        "safe_range_km": round(ev.max_range_km(min_arrival_soc), 1),
        "reserve_soc_pct": min_arrival_soc,
    }
