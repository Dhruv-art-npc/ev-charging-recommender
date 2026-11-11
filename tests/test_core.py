import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np, pandas as pd, pytest
from src.ev_data import EVState
from src.geo import haversine_km
from src.stations import load_stations, find_nearby
from src.reachability import filter_reachable, soc_at_arrival
from src.ranking import add_charging_metrics, rank_stations, erlang_c_wait_min
from src.recommender import recommend
from src import synthetic


@pytest.fixture(scope="module")
def stations():
    p = pathlib.Path(__file__).parents[1] / "data" / "india_charging_stations.csv"
    if not p.exists(): synthetic.write_all()
    return load_stations(p)


def ev(**k):
    base = dict(lat=23.2599, lon=77.4126, battery_kwh=40, soc=10, target_soc=80, consumption_kwh_per_km=0.15, efficiency=0.9)
    base.update(k); return EVState(**base)


def test_haversine_bhopal_indore():
    assert 165 < haversine_km(23.2599, 77.4126, 22.7196, 75.8577) < 175


def test_range_10pct():
    # 4 kWh * 0.9 / 0.15 = 24 km full ; with 5% reserve: 2 kWh -> 12 km
    assert ev().max_range_km(0) == pytest.approx(24.0)
    assert ev().max_range_km(5) == pytest.approx(12.0)


def test_low_battery_removes_far_stations(stations):
    near = find_nearby(stations, 23.2599, 77.4126, radius_km=150)
    reach = filter_reachable(near, ev())
    assert len(reach) < len(near)
    assert (reach.soc_arrival >= 5).all()
    assert reach.distance_km.max() <= 12.0 + 1e-6


def test_all_objectives_respect_constraint(stations):
    for obj in ("balanced", "min_distance", "min_cost", "min_time"):
        r = recommend(ev(), stations, obj, top_n=50, match_connector=False)
        if not r["ranked"].empty:
            assert (r["ranked"].soc_arrival >= 5).all()


def test_objective_ordering(stations):
    e = ev(soc=60)
    d = recommend(e, stations, "min_distance", top_n=50, match_connector=False)["ranked"]
    c = recommend(e, stations, "min_cost", top_n=50, match_connector=False)["ranked"]
    assert d.distance_km.is_monotonic_increasing
    assert c.charging_cost.is_monotonic_increasing


def test_unreachable_message(stations):
    r = recommend(ev(lat=21.0, lon=70.0, soc=6), stations)   # remote + almost empty
    assert r["best"] is None and "No station reachable" in r["message"]


def test_validation():
    with pytest.raises(ValueError): ev(target_soc=5)
    with pytest.raises(ValueError): ev(soc=120)


def test_erlang_c_monotonic():
    assert erlang_c_wait_min(2, 0.3, 30) < erlang_c_wait_min(2, 0.9, 30)
