"""Full pipeline: EV input -> find stations -> reachability -> metrics -> rank -> recommend.

Charging time now uses CC-CV physics (src.charging_physics).
Dynamic pricing via exponential load formula (src.dynamic_pricing).
"""
from __future__ import annotations
import pandas as pd
import config
from .ev_data import EVState
from .stations import find_nearby
from .reachability import filter_reachable, range_summary
from .ranking import add_charging_metrics, rank_stations

SHOW_COLS = ["rank", "station_id", "name", "city", "operator", "connector_type", "power_kw",
             "distance_km", "soc_arrival", "travel_time_min", "wait_time_min", "charging_time_min",
             "total_time_min", "energy_to_charge_kwh", "price_per_kwh", "dynamic_price_per_kwh",
             "charging_cost", "station_load", "latitude", "longitude"]


def recommend(ev: EVState, stations: pd.DataFrame, objective: str = "balanced", forecaster=None,
              now: pd.Timestamp | None = None, top_n: int = 5,
              min_arrival_soc: float = config.MIN_ARRIVAL_SOC, weights: dict | None = None,
              match_connector: bool = True,
              use_dynamic_pricing: bool = True,
              lambda_load: float = 1.0) -> dict:
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    rng = range_summary(ev, min_arrival_soc)
    near = find_nearby(stations, ev.lat, ev.lon, radius_km=config.SEARCH_RADIUS_KM,
                       connector=ev.connector if match_connector else None)
    reach = filter_reachable(near, ev, min_arrival_soc)

    out = {"range": rng, "objective": objective, "n_candidates": len(near), "n_reachable": len(reach),
           "n_removed_unreachable": len(near) - len(reach), "ranked": pd.DataFrame(), "best": None,
           "load_factor": 1.0, "message": ""}
    if reach.empty:
        out["message"] = (f"No station reachable within safe range {rng['safe_range_km']} km at "
                          f"{ev.soc}% SOC. Stop safely, call roadside assistance / portable charger, "
                          f"or lower the arrival-SOC reserve if it is an emergency.")
        return out

    eta = now + pd.Timedelta(minutes=float(reach["distance_km"].min() / config.AVG_SPEED_KMPH * 60))
    lf = forecaster.load_factor(eta) if forecaster is not None else 1.0
    out["load_factor"] = lf
    metrics = add_charging_metrics(reach, ev, load_factor=lf,
                                   use_dynamic_pricing=use_dynamic_pricing,
                                   lambda_load=lambda_load)
    ranked = rank_stations(metrics, objective, weights)
    ranked = ranked[[c for c in SHOW_COLS if c in ranked.columns]].round(2)
    out["ranked"] = ranked.head(top_n)
    out["best"] = ranked.iloc[0].to_dict()
    b = out["best"]
    out["message"] = (f"Go to {b['name']} ({b['city']}): {b['distance_km']} km away, arrive with "
                      f"{b['soc_arrival']}% SOC, wait ~{b['wait_time_min']} min, charge "
                      f"{ev.soc}->{ev.target_soc}% in ~{b['charging_time_min']} min for Rs {b['charging_cost']}.")
    return out
