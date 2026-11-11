"""Module 4 - Best station selection (min distance / min cost / balanced).

Charging-time calculation now uses the realistic CC-CV physics model from
``src.charging_physics`` instead of the flat TAPER_DERATE coefficient.
Dynamic pricing from ``src.dynamic_pricing`` is applied when the caller
passes ``use_dynamic_pricing=True``.
"""
from __future__ import annotations
import math
import pandas as pd
import config
from .ev_data import EVState
from .charging_physics import cc_cv_charging_time_min, integrate_cc_cv_kwh
from .dynamic_pricing import ExponentialLoadPricer

MAX_WAIT_MIN = 60.0   # displayed wait is capped (60 = "60+ min")
OBJECTIVES = ("balanced", "min_distance", "min_cost", "min_time")


def erlang_c_wait_min(n_servers: int, rho: float, service_min: float) -> float:
    """Mean waiting time (min) in an M/M/c queue. rho = utilisation per server (0-1)."""
    rho = min(max(rho, 0.0), 0.90)                 # cap: drivers re-route before queues explode
    c = max(int(n_servers), 1)
    a = rho * c                                    # offered load
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / (math.factorial(c) * (1 - rho))
    p_wait = top / (s + top)
    return min(p_wait * service_min / (c * (1 - rho)), MAX_WAIT_MIN)


def add_charging_metrics(df: pd.DataFrame,
                          ev: EVState,
                          load_factor: float = 1.0,
                          use_dynamic_pricing: bool = True,
                          lambda_load: float = 1.0) -> pd.DataFrame:
    """
    Charging time/cost, expected wait and effective load at each station.

    Improvements over the original flat-derate model
    -------------------------------------------------
    * **CC-CV physics**: ``charging_time_min`` is computed by numerical
      integration of the exponential taper curve (via
      ``charging_physics.cc_cv_charging_time_min``) rather than a crude
      TAPER_DERATE multiplier.  This is accurate across the full 0–100 % SoC
      range, including sessions that span the CC→CV boundary.

    * **Dynamic pricing**: when ``use_dynamic_pricing=True`` the
      ``charging_cost`` uses ``ExponentialLoadPricer.effective_cost_per_kwh``
      which applies the demand-surge formula factored with the CV efficiency
      curve.  The ``dynamic_price_per_kwh`` column is also added.

    Parameters
    ----------
    load_factor : float
        From the arrival forecaster (>1 = busier than typical at ETA).
    use_dynamic_pricing : bool
        Apply exponential load-based price surge (default True).
    lambda_load : float
        Surge steepness λ in P = Base * exp(λ * Load/Cap) (default 1.0).
    """
    df = df.copy()
    df["station_load"] = (df["current_load_pct"] * load_factor).clip(0, 95)

    # ---- CC-CV charging time (per row, accounts for taper) -------------------
    arrive_soc = df["soc_arrival"].clip(lower=0)

    charging_times = []
    grid_kwhs = []
    for _, row in df.iterrows():
        soc_a = float(arrive_soc.loc[row.name])
        p_kw = min(float(row["power_kw"]), ev.max_dc_kw)
        t_min = cc_cv_charging_time_min(
            soc_start=soc_a,
            soc_end=ev.target_soc,
            battery_kwh=ev.battery_kwh,
            power_kw=p_kw,
            charger_eff=config.CHARGER_EFFICIENCY,
        )
        kwh = integrate_cc_cv_kwh(
            soc_start=soc_a,
            soc_end=ev.target_soc,
            battery_kwh=ev.battery_kwh,
            power_kw=p_kw,
            charger_eff=config.CHARGER_EFFICIENCY,
        )
        charging_times.append(t_min)
        grid_kwhs.append(kwh)

    df["charging_time_min"] = charging_times
    df["energy_to_charge_kwh"] = grid_kwhs

    # ---- Dynamic or static pricing -------------------------------------------
    if use_dynamic_pricing:
        pricer_rows = []
        eff_prices = []
        for _, row in df.iterrows():
            soc_a = float(arrive_soc.loc[row.name])
            pricer = ExponentialLoadPricer(
                base_price_per_kwh=float(row.get("price_per_kwh", 15.0)),
                lambda_load=lambda_load,
            )
            dyn = pricer.price_at_load(float(row["station_load"]))
            eff = pricer.effective_cost_per_kwh(
                current_load_pct=float(row["station_load"]),
                soc_start=soc_a,
                soc_end=ev.target_soc,
            )
            pricer_rows.append(dyn)
            eff_prices.append(eff)
        df["dynamic_price_per_kwh"] = pricer_rows
        df["charging_cost"] = [
            round(kwh * ep, 2) for kwh, ep in zip(df["energy_to_charge_kwh"], eff_prices)
        ]
    else:
        df["dynamic_price_per_kwh"] = df["price_per_kwh"]
        df["charging_cost"] = df["energy_to_charge_kwh"] * df["price_per_kwh"]

    # ---- Queue model (unchanged) ---------------------------------------------
    df["wait_time_min"] = [
        erlang_c_wait_min(n, l / 100.0, s)
        for n, l, s in zip(df["n_chargers"], df["station_load"], df["avg_session_min"])
    ]
    df["travel_time_min"] = df["distance_km"] / config.AVG_SPEED_KMPH * 60.0
    df["total_time_min"] = df["travel_time_min"] + df["wait_time_min"] + df["charging_time_min"]
    return df


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return pd.Series(0.0, index=s.index) if rng == 0 else (s - s.min()) / rng


def rank_stations(df: pd.DataFrame, objective: str = "balanced",
                  weights: dict | None = None) -> pd.DataFrame:
    """objective:
        min_distance -> nearest first (ties: cheaper)
        min_cost     -> cheapest total charging cost first (ties: nearer)
        min_time     -> lowest total time first
        balanced     -> weighted score of distance, travel time, cost, wait time, load"""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if df.empty:
        return df
    df = df.copy()
    w = weights or config.WEIGHTS
    df["score"] = sum(w[k] * _minmax(df[k]) for k in w)          # lower = better
    sort_map = {
        "balanced": ["score", "distance_km"],
        "min_distance": ["distance_km", "charging_cost"],
        "min_cost": ["charging_cost", "distance_km"],
        "min_time": ["total_time_min", "distance_km"],
    }
    df = df.sort_values(sort_map[objective]).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
