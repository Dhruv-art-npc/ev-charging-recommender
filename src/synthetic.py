"""Synthetic fallback data (used ONLY when you do not have the real files yet).
Real datasets: drop them in data/ using the same file names (see README)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import config

CITIES = {  # city: (state, lat, lon, n_stations)
    "Bhopal": ("Madhya Pradesh", 23.2599, 77.4126, 22), "Indore": ("Madhya Pradesh", 22.7196, 75.8577, 22),
    "Jabalpur": ("Madhya Pradesh", 23.1815, 79.9864, 10), "Gwalior": ("Madhya Pradesh", 26.2183, 78.1828, 8),
    "Ujjain": ("Madhya Pradesh", 23.1765, 75.7885, 6), "Sagar": ("Madhya Pradesh", 23.8388, 78.7378, 5),
    "Delhi": ("Delhi", 28.6139, 77.2090, 45), "Mumbai": ("Maharashtra", 19.0760, 72.8777, 45),
    "Pune": ("Maharashtra", 18.5204, 73.8567, 28), "Nagpur": ("Maharashtra", 21.1458, 79.0882, 14),
    "Bengaluru": ("Karnataka", 12.9716, 77.5946, 45), "Hyderabad": ("Telangana", 17.3850, 78.4866, 32),
    "Chennai": ("Tamil Nadu", 13.0827, 80.2707, 28), "Kolkata": ("West Bengal", 22.5726, 88.3639, 20),
    "Ahmedabad": ("Gujarat", 23.0225, 72.5714, 24), "Jaipur": ("Rajasthan", 26.9124, 75.7873, 18),
    "Lucknow": ("Uttar Pradesh", 26.8467, 80.9462, 14), "Kota": ("Rajasthan", 25.2138, 75.8648, 6),
    "Raipur": ("Chhattisgarh", 21.2514, 81.6296, 8), "Surat": ("Gujarat", 21.1702, 72.8311, 12),
}
OPERATORS = ["Tata Power", "Statiq", "ChargeZone", "Ather Grid", "EESL", "Jio-bp pulse", "Fortum"]
CONNECTORS = ["CCS2", "CCS2", "CHAdeMO", "Type2", "GB/T"]


def make_india_stations(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for city, (state, lat, lon, n) in CITIES.items():
        for i in range(n):
            power = float(rng.choice([7.4, 22, 30, 60, 120], p=[.15, .25, .3, .2, .1]))
            rows.append(dict(
                station_id=f"{city[:3].upper()}{i:03d}", name=f"{rng.choice(OPERATORS)} {city} #{i+1}",
                operator=rng.choice(OPERATORS), city=city, state=state,
                latitude=round(lat + rng.normal(0, 0.07), 5), longitude=round(lon + rng.normal(0, 0.07), 5),
                connector_type=rng.choice(CONNECTORS), power_kw=power, n_chargers=int(rng.integers(1, 7)),
                price_per_kwh=round(float(rng.uniform(10, 24) + (3 if power >= 60 else 0)), 2),
                current_load_pct=round(float(rng.uniform(10, 85)), 1),
                avg_session_min=round(float(rng.uniform(25, 60)), 0)))
    # highway stations between Bhopal-Indore-Ujjain-Sagar corridors
    for (a, b) in [("Bhopal", "Indore"), ("Bhopal", "Sagar"), ("Bhopal", "Jabalpur")]:
        la, oa = CITIES[a][1:3]; lb, ob = CITIES[b][1:3]
        for t in np.linspace(0.15, 0.85, 4):
            rows.append(dict(station_id=f"HWY-{a[:2]}{b[:2]}{int(t*100)}", name=f"Highway Hub {a}-{b} {int(t*100)}%",
                             operator=rng.choice(OPERATORS), city="Highway", state="Madhya Pradesh",
                             latitude=round(la + (lb - la) * t, 5), longitude=round(oa + (ob - oa) * t, 5),
                             connector_type="CCS2", power_kw=60.0, n_chargers=int(rng.integers(2, 5)),
                             price_per_kwh=round(float(rng.uniform(16, 24)), 2),
                             current_load_pct=round(float(rng.uniform(10, 60)), 1), avg_session_min=35.0))
    return pd.DataFrame(rows)


def make_epfl_like_sessions(seed: int = 7, start="2022-04-12", end="2023-07-04") -> pd.DataFrame:
    """Session-level table shaped like the EPFL Level-3 charging dataset (one row per session)."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h")
    hod, dow, month = hours.hour.values, hours.dayofweek.values, hours.month.values
    daily = np.array([.1, .05, .05, .05, .1, .3, .7, 1.2, 1.6, 1.5, 1.4, 1.6, 1.9, 1.7, 1.5, 1.5, 1.7, 2.0, 1.8, 1.2, .8, .5, .3, .2])
    lam = 0.9 * daily[hod] * np.where(dow >= 5, 1.25, 1.0) * (1 + 0.25 * np.sin((month - 4) / 12 * 2 * np.pi))
    lam = lam * (1 + 0.5 * (np.arange(len(hours)) / len(hours)))                # growing usage
    counts = rng.poisson(lam)
    models = [(30, 50), (40, 50), (52, 100), (64, 100), (77, 150), (82, 170), (100, 190)]
    rows = []
    for ts, k in zip(hours, counts):
        for _ in range(k):
            cap, pmax = models[rng.integers(len(models))]
            s0 = float(rng.uniform(8, 60)); s1 = float(min(100, s0 + rng.uniform(20, 55)))
            energy = (s1 - s0) / 100 * cap
            avg_p = pmax * (0.55 if s1 > 80 else 0.75) * rng.uniform(0.8, 1.0)
            dur = energy / avg_p * 60
            start_t = ts + pd.Timedelta(minutes=float(rng.uniform(0, 60)))
            rows.append(dict(start_time=start_t, end_time=start_t + pd.Timedelta(minutes=dur),
                             duration_min=round(dur, 2), energy_kwh=round(energy, 2),
                             max_power_kw=round(avg_p * 1.3, 1), soc_start=round(s0, 1), soc_end=round(s1, 1),
                             battery_kwh=cap))
    return pd.DataFrame(rows)


def write_all():
    config.DATA_DIR.mkdir(exist_ok=True)
    st = make_india_stations(); st.to_csv(config.STATIONS_CSV, index=False)
    ss = make_epfl_like_sessions(); ss.to_csv(config.EPFL_SESSIONS_CSV, index=False)
    return len(st), len(ss)
