"""Central configuration for the EV charging recommender."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

# Dataset paths (replace the files, keep the names -> everything works)
STATIONS_CSV = DATA_DIR / "india_charging_stations.csv"
EPFL_SESSIONS_CSV = DATA_DIR / "epfl_l3_sessions.csv"

# ---- Reachability / safety ----
ROAD_FACTOR = 1.30          # road distance = straight-line * 1.30
AVG_SPEED_KMPH = 45.0       # average speed used for travel time
MIN_ARRIVAL_SOC = 5.0       # % SOC that must remain on arrival (hard safety reserve)
SEARCH_RADIUS_KM = 150.0    # candidate pre-filter radius

# ---- Charging model ----
CHARGER_EFFICIENCY = 0.92   # grid -> battery efficiency
TAPER_SOC = 80.0            # above this SOC charging slows down
TAPER_DERATE = 0.80         # average power derate when charging goes beyond TAPER_SOC

# ---- Ranking weights (balanced mode) ----
WEIGHTS = {
    "distance_km": 0.25,
    "travel_time_min": 0.15,
    "charging_cost": 0.30,
    "wait_time_min": 0.15,
    "station_load": 0.15,
}
