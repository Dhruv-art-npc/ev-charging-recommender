"""
charging_physics.py – Realistic CC-CV lithium-ion charging model
=================================================================
Implements a Constant-Current / Constant-Voltage (CC-CV) charging profile
where the charge rate (kW) is held constant below the taper threshold (CC
phase) and falls exponentially toward 100 % SoC (CV phase).

Physics reference
-----------------
In the CV phase the current (and therefore power) decays approximately as:
    P(t) = P_cc * exp( -k_taper * (SoC - SoC_taper) / (1 - SoC_taper) )

where
    SoC_taper  : transition point from CC → CV  (default 0.80, i.e. 80 %)
    k_taper    : decay steepness constant        (default ln(10) ≈ 2.3,
                 meaning power drops to ~10 % of P_cc at 100 % SoC)
    P_cc       : charging power during CC phase  (kW, limited by min of
                 station power and vehicle's max DC acceptance)

Usage
-----
>>> from src.charging_physics import cc_cv_charging_time_min, integrate_cc_cv_kwh
>>> t_min = cc_cv_charging_time_min(soc_start=20, soc_end=95, battery_kwh=40, power_kw=50)
>>> kwh   = integrate_cc_cv_kwh(soc_start=20, soc_end=95, battery_kwh=40, power_kw=50)

ESP32 / Hardware readiness
--------------------------
Both functions accept ``dt_s`` (time-step in seconds) and return stepwise
arrays via the ``_profile`` helpers – these can be published tick-by-tick
over MQTT to an ESP32 dashboard.  See main.py for the MQTT hook.
"""
from __future__ import annotations
import math
import numpy as np

# ---- Default physics constants ------------------------------------------------
SOC_TAPER_DEFAULT   = 0.80   # 80 % → start of CV phase
K_TAPER_DEFAULT     = 2.302  # ln(10): power decays to ~10 % at 100 % SoC
CHARGER_EFF_DEFAULT = 0.92   # grid-to-battery efficiency
MIN_POWER_FRAC      = 0.05   # stop integration when power < 5 % of rated (CV tail)


# ============================================================ helpers ===========

def _cv_power_fraction(soc_norm: float,
                        soc_taper: float = SOC_TAPER_DEFAULT,
                        k: float = K_TAPER_DEFAULT) -> float:
    """
    Power fraction in CV phase [0, 1] for a *normalised* SoC in [0, 1].

    CC phase (soc_norm <= soc_taper)  → fraction = 1.0
    CV phase (soc_norm >  soc_taper)  → fraction = exp(-k * x)
        where x = (soc_norm - soc_taper) / (1 - soc_taper)  ∈ [0, 1]
    """
    if soc_norm <= soc_taper:
        return 1.0
    x = (soc_norm - soc_taper) / max(1.0 - soc_taper, 1e-9)
    return math.exp(-k * x)


def cv_power_kw(soc_pct: float,
                p_cc_kw: float,
                soc_taper_pct: float = SOC_TAPER_DEFAULT * 100,
                k: float = K_TAPER_DEFAULT) -> float:
    """
    Instantaneous charging power (kW) at a given SoC percentage.

    Parameters
    ----------
    soc_pct       : current state of charge in % (0–100)
    p_cc_kw       : constant-current phase power (kW)
    soc_taper_pct : SoC % at which CC→CV transition begins (default 80 %)
    k             : exponential decay constant (default ln(10) ≈ 2.302)

    Returns
    -------
    Power in kW at this SoC.
    """
    return p_cc_kw * _cv_power_fraction(soc_pct / 100.0,
                                         soc_taper_pct / 100.0, k)


# =========================================================== integrators ========

def cc_cv_charging_time_min(soc_start: float,
                             soc_end: float,
                             battery_kwh: float,
                             power_kw: float,
                             soc_taper_pct: float = SOC_TAPER_DEFAULT * 100,
                             k: float = K_TAPER_DEFAULT,
                             charger_eff: float = CHARGER_EFF_DEFAULT,
                             dt_soc: float = 0.1) -> float:
    """
    Estimate total wall-clock charging time (minutes) for a CC-CV session.

    Uses a numerical SoC-domain integration with step ``dt_soc`` (%),
    which is equivalent to tracking energy in small increments.

    Parameters
    ----------
    soc_start, soc_end  : initial and target SoC in %
    battery_kwh         : usable battery capacity (kWh)
    power_kw            : rated / CC-phase power (kW), limited by car & station
    soc_taper_pct       : CC→CV transition SoC (default 80 %)
    k                   : CV exponential decay steepness
    charger_eff         : grid-to-battery efficiency
    dt_soc              : integration step size in % SoC (default 0.1 %)

    Returns
    -------
    Charging duration in minutes.
    """
    if soc_end <= soc_start:
        return 0.0
    total_min = 0.0
    soc = float(soc_start)
    while soc < soc_end:
        step = min(dt_soc, soc_end - soc)
        # Energy into battery for this SoC step (kWh)
        d_kwh_bat = step / 100.0 * battery_kwh
        # Grid draw (more kWh from grid because of charger inefficiency)
        d_kwh_grid = d_kwh_bat / charger_eff
        # Power available at this SoC (after CV taper)
        p_now = cv_power_kw(soc, power_kw, soc_taper_pct, k)
        p_now = max(p_now, power_kw * MIN_POWER_FRAC)   # avoid /0
        # Time for this slice (hours → minutes)
        total_min += (d_kwh_grid / p_now) * 60.0
        soc += step
    return total_min


def integrate_cc_cv_kwh(soc_start: float,
                         soc_end: float,
                         battery_kwh: float,
                         power_kw: float,
                         soc_taper_pct: float = SOC_TAPER_DEFAULT * 100,
                         k: float = K_TAPER_DEFAULT,
                         charger_eff: float = CHARGER_EFF_DEFAULT,
                         dt_soc: float = 0.1) -> float:
    """
    Total grid energy drawn (kWh) for a CC-CV session.

    Accounts for charger efficiency so this is what the station meters.
    """
    if soc_end <= soc_start:
        return 0.0
    kwh = (soc_end - soc_start) / 100.0 * battery_kwh / charger_eff
    return round(kwh, 4)


def cc_cv_power_profile(soc_start: float,
                         soc_end: float,
                         battery_kwh: float,
                         power_kw: float,
                         soc_taper_pct: float = SOC_TAPER_DEFAULT * 100,
                         k: float = K_TAPER_DEFAULT,
                         charger_eff: float = CHARGER_EFF_DEFAULT,
                         dt_s: float = 30.0) -> list[dict]:
    """
    Generate a time-series power profile suitable for MQTT telemetry to ESP32.

    Returns a list of dicts, one per ``dt_s``-second time step::

        [{"t_s": 0, "soc_pct": 20.0, "power_kw": 50.0, "phase": "CC"},
         {"t_s": 30, "soc_pct": 20.03, "power_kw": 50.0, "phase": "CC"},
         ...
         {"t_s": 1860, "soc_pct": 95.0, "power_kw": 8.3, "phase": "CV"}]

    This list can be iterated and each row published via paho-mqtt
    (see the MQTT hook in main.py).
    """
    profile = []
    soc = float(soc_start)
    t_s = 0.0
    while soc < soc_end:
        p_now = cv_power_kw(soc, power_kw, soc_taper_pct, k)
        p_now = max(p_now, power_kw * MIN_POWER_FRAC)
        phase = "CC" if soc <= soc_taper_pct else "CV"
        profile.append({
            "t_s": round(t_s, 1),
            "soc_pct": round(soc, 3),
            "power_kw": round(p_now, 3),
            "phase": phase,
        })
        # Energy delivered to battery in dt_s seconds
        d_kwh_bat = p_now * charger_eff * (dt_s / 3600.0)
        d_soc = d_kwh_bat / battery_kwh * 100.0
        soc = min(soc + d_soc, soc_end)
        t_s += dt_s
    # Final snapshot at target SoC
    profile.append({
        "t_s": round(t_s, 1),
        "soc_pct": round(soc_end, 3),
        "power_kw": round(cv_power_kw(soc_end, power_kw, soc_taper_pct, k), 3),
        "phase": "CV" if soc_end > soc_taper_pct else "CC",
    })
    return profile
