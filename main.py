"""
main.py – CLI entry-point for the EV Charging Recommender pipeline
==================================================================

Commands
--------
  python main.py generate-data           # write synthetic CSV files
  python main.py train                   # fit ML arrival forecaster + session models
  python main.py recommend               # score & rank real stations (uses CC-CV + dynamic pricing)
  python main.py run-pipeline            # ★ NEW: full end-to-end batch run, writes outputs/

MQTT / ESP32 Hardware Readiness
--------------------------------
This module contains a self-contained MQTT publisher hook (``MQTTPublisher``)
that wraps paho-mqtt.  When enabled (via --mqtt-host), every recommendation
result and the CC-CV charging profile is published as JSON to two topics:

  ev/recommendation   – top-ranked station + route summary
  ev/charging_profile – tick-by-tick SoC / power telemetry for the ESP32 HMI

The hook is deliberately modular: you can drop it into any microcontroller
project by importing ``MQTTPublisher`` and calling ``publish_recommendation()``.

Usage with ESP32
~~~~~~~~~~~~~~~~
  # On the Python side (gateway / Raspberry Pi):
  python main.py recommend --mqtt-host 192.168.1.42 --mqtt-port 1883

  # On the ESP32 (MicroPython / Arduino):
  client.subscribe("ev/recommendation")   # parse JSON for display
  client.subscribe("ev/charging_profile") # render SoC bar / power gauge
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import pandas as pd

import config
from src import synthetic, ev_data
from src.stations import load_stations
from src.forecasting import ArrivalForecaster, load_sessions, hourly_arrivals, train_session_models
from src.recommender import recommend
from src.charging_physics import cc_cv_power_profile
from src.dynamic_pricing import ExponentialLoadPricer, MathematicalArrivalForecaster


# ══════════════════════════════════════════════════════════════════════════════
#  MQTT HOOK  (paho-mqtt, optional – gracefully degrades if not installed)
# ══════════════════════════════════════════════════════════════════════════════

class MQTTPublisher:
    """
    Thin wrapper around paho-mqtt for publishing EV telemetry to an ESP32.

    Designed to be hardware-ready out of the box:
      • Connects once at construction; reconnects automatically on failure.
      • publish_recommendation() sends the ranked results as JSON.
      • publish_cc_cv_profile() streams the charging curve tick-by-tick.

    Topics (configurable)
    ---------------------
    ev/recommendation   – dict: best station, estimated cost, time, SoC.
    ev/charging_profile – dict per tick: {"t_s", "soc_pct", "power_kw", "phase"}

    Parameters
    ----------
    host : str   – MQTT broker IP or hostname (e.g. "192.168.1.42")
    port : int   – broker port (default 1883; TLS usually 8883)
    client_id : str – unique client name (shown in broker logs)
    topic_prefix : str – prefix for all topics (default "ev")
    """

    # ---- MQTT topic names ------------------------------------------------
    TOPIC_RECOMMENDATION = "ev/recommendation"
    TOPIC_CC_CV_PROFILE  = "ev/charging_profile"
    TOPIC_HEARTBEAT      = "ev/heartbeat"

    def __init__(self, host: str, port: int = 1883,
                 client_id: str = "ev_recommender_gw",
                 topic_prefix: str = "ev"):
        self._host = host
        self._port = port
        self._prefix = topic_prefix
        self._connected = False
        self._client = None
        self._init_client(client_id)

    def _init_client(self, client_id: str) -> None:
        """Initialise paho-mqtt client; silently skip if library is absent."""
        try:
            import paho.mqtt.client as mqtt  # noqa: PLC0415

            # ---- paho ≥2.0 requires CallbackAPIVersion ----
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self._client = mqtt.Client(
                    CallbackAPIVersion.VERSION2, client_id=client_id)
            except (ImportError, AttributeError):
                # paho <2.0 fallback
                self._client = mqtt.Client(client_id=client_id)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()          # background thread
        except ImportError:
            print("[MQTT] paho-mqtt not installed – pip install paho-mqtt to enable.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[MQTT] Connection failed ({exc}) – continuing without broker.",
                  file=sys.stderr)

    # ---- paho callbacks --------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        self._connected = (rc == 0)
        status = "connected" if self._connected else f"refused (rc={rc})"
        print(f"[MQTT] Broker {self._host}:{self._port} → {status}")

    def _on_disconnect(self, client, userdata, rc, *args) -> None:
        self._connected = False
        print(f"[MQTT] Disconnected (rc={rc})")

    def _publish(self, topic: str, payload: dict, qos: int = 1) -> None:
        """Internal helper: JSON-encode and publish; noop if not connected."""
        if self._client is None or not self._connected:
            return
        self._client.publish(topic, json.dumps(payload), qos=qos)

    # ---- public API ------------------------------------------------------
    def publish_recommendation(self, result: dict) -> None:
        """
        Publish the top recommendation to ``ev/recommendation``.

        Payload sent to ESP32
        ---------------------
        {
          "station_name": "...",
          "city": "...",
          "distance_km": 12.3,
          "charging_time_min": 34.5,
          "charging_cost_inr": 150.2,
          "dynamic_price_per_kwh": 18.4,
          "load_factor": 1.1,
          "soc_arrival": 15.0
        }
        """
        best = result.get("best") or {}
        payload = {
            "station_name": best.get("name", "N/A"),
            "city": best.get("city", ""),
            "distance_km": best.get("distance_km", 0),
            "charging_time_min": best.get("charging_time_min", 0),
            "charging_cost_inr": best.get("charging_cost", 0),
            "dynamic_price_per_kwh": best.get("dynamic_price_per_kwh",
                                               best.get("price_per_kwh", 0)),
            "load_factor": result.get("load_factor", 1.0),
            "soc_arrival": best.get("soc_arrival", 0),
        }
        self._publish(self.TOPIC_RECOMMENDATION, payload)
        print(f"[MQTT] → {self.TOPIC_RECOMMENDATION}: {payload}")

    def publish_cc_cv_profile(self, profile: list[dict]) -> None:
        """
        Stream the CC-CV power profile tick-by-tick to ``ev/charging_profile``.

        Each tick is published as a separate MQTT message so the ESP32 can
        update a live SoC bar and power gauge in real time.

        Payload per tick
        ----------------
        {"t_s": 60, "soc_pct": 22.1, "power_kw": 50.0, "phase": "CC"}
        """
        for tick in profile:
            self._publish(self.TOPIC_CC_CV_PROFILE, tick, qos=0)  # QoS-0 for speed

    def publish_heartbeat(self, ev_soc: float, ev_lat: float, ev_lon: float) -> None:
        """
        Periodic heartbeat so the ESP32 dashboard stays alive even if no
        recommendation has been generated yet.

        Payload: {"soc_pct": 45.0, "lat": 23.26, "lon": 77.41, "ts": "2026-..."}
        """
        self._publish(self.TOPIC_HEARTBEAT, {
            "soc_pct": ev_soc, "lat": ev_lat, "lon": ev_lon,
            "ts": pd.Timestamp.now().isoformat(),
        })

    def disconnect(self) -> None:
        """Clean shutdown."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(_):
    print("stations, sessions =", synthetic.write_all())


def cmd_train(a):
    sessions = load_sessions(a.sessions)
    series = hourly_arrivals(sessions)
    fc = ArrivalForecaster()
    m = fc.fit(series)
    fc.save()
    sm = train_session_models(sessions)
    config.OUT_DIR.mkdir(exist_ok=True)
    report = {"arrival_forecast": m, "session_models": sm, "n_sessions": len(sessions)}
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_recommend(a):
    ev = ev_data.from_json(a.json) if a.json else ev_data.from_manual(
        lat=a.lat, lon=a.lon, battery_kwh=a.battery, soc=a.soc, target_soc=a.target,
        consumption_kwh_per_km=a.consumption, efficiency=a.efficiency, max_dc_kw=a.max_dc,
        connector=a.connector)

    # ---- choose forecaster ---------------------------------------------------
    # Priority: 1) trained ML model  2) math forecaster (no training needed)
    fc = None
    mp = config.MODEL_DIR / "arrival_rf.joblib"
    if mp.exists() and not a.no_forecast:
        fc = ArrivalForecaster.load(mp)
        print("[Forecaster] Using trained Random-Forest arrival model.")
    elif not a.no_forecast:
        fc = MathematicalArrivalForecaster()
        print("[Forecaster] Using mathematical (Gaussian double-peak) arrival model.")

    res = recommend(ev, load_stations(a.stations), a.objective, fc, top_n=a.top,
                    min_arrival_soc=a.reserve, match_connector=not a.any_connector,
                    use_dynamic_pricing=not a.no_dynamic_pricing,
                    lambda_load=a.lambda_load)

    print("Range:", res["range"])
    print(f"Candidates {res['n_candidates']} | reachable {res['n_reachable']} | "
          f"removed (cannot reach) {res['n_removed_unreachable']} | "
          f"forecast load factor {res['load_factor']:.2f}")
    print(res["message"])
    if not res["ranked"].empty:
        pd.set_option("display.width", 300)
        pd.set_option("display.max_columns", 35)
        print(res["ranked"].drop(columns=["latitude", "longitude"]).to_string(index=False))

    # ---- MQTT hook -----------------------------------------------------------
    if a.mqtt_host:
        # ── Hardware readiness: publish result + CC-CV profile to ESP32 ────────
        publisher = MQTTPublisher(host=a.mqtt_host, port=a.mqtt_port)
        import time; time.sleep(0.5)          # wait for broker ACK
        publisher.publish_recommendation(res)
        if res["best"]:
            # Build CC-CV profile for the recommended session
            best = res["best"]
            profile = cc_cv_power_profile(
                soc_start=float(best.get("soc_arrival", ev.soc)),
                soc_end=ev.target_soc,
                battery_kwh=ev.battery_kwh,
                power_kw=float(best.get("power_kw", ev.max_dc_kw)),
            )
            publisher.publish_cc_cv_profile(profile)
            print(f"[MQTT] Published {len(profile)} CC-CV profile ticks.")
        publisher.disconnect()


# ══════════════════════════════════════════════════════════════════════════════
#  ★ NEW: run-pipeline – batch end-to-end, writes structured outputs/
# ══════════════════════════════════════════════════════════════════════════════

def cmd_run_pipeline(a):
    """
    Full batch pipeline:
      1. Load real EPFL sessions → compute mathematical forecasts
      2. Load India stations
      3. Run recommendations for a grid of representative EV scenarios
      4. Apply exponential dynamic pricing to all stations
      5. Write structured JSON + CSV outputs to outputs/

    Outputs produced
    ----------------
    outputs/pipeline_results.json   – per-scenario recommendation summaries
    outputs/pipeline_ranked.csv     – flattened ranked stations (all scenarios)
    outputs/dynamic_pricing.csv     – all stations with dynamic price columns
    outputs/cc_cv_profiles.json     – CC-CV power profiles for each scenario
    outputs/math_forecast_24h.csv   – 24-hour mathematical arrival forecast
    """
    config.OUT_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("EV CHARGING RECOMMENDER  –  Full Pipeline Run")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading sessions from {config.EPFL_SESSIONS_CSV} …")
    try:
        sessions = load_sessions(config.EPFL_SESSIONS_CSV)
        print(f"      {len(sessions):,} sessions loaded.")
    except Exception as exc:
        print(f"      WARNING: could not load sessions ({exc}). Skipping session stats.")
        sessions = pd.DataFrame()

    print(f"\n[2/5] Loading stations from {config.STATIONS_CSV} …")
    stations = load_stations(config.STATIONS_CSV)
    print(f"      {len(stations):,} valid stations in India bounding box.")

    # ── 2. Mathematical arrival forecast (24 h from now) ─────────────────────
    print("\n[3/5] Running mathematical arrival forecast (next 24 h) …")
    math_fc = MathematicalArrivalForecaster()
    now = pd.Timestamp.now()
    forecast_24h = math_fc.forecast(start=now, hours=24)
    fc_csv_path = config.OUT_DIR / "math_forecast_24h.csv"
    forecast_24h.to_csv(fc_csv_path, header=True)
    print(f"      Forecast saved → {fc_csv_path}")

    # ── 3. Dynamic pricing on all stations ───────────────────────────────────
    print("\n[4/5] Applying exponential dynamic pricing to all stations …")
    pricer = ExponentialLoadPricer(lambda_load=1.0)
    stations_priced = pricer.apply_to_stations(stations, soc_start=20.0, soc_end=80.0)
    dp_csv_path = config.OUT_DIR / "dynamic_pricing.csv"
    stations_priced.to_csv(dp_csv_path, index=False)
    print(f"      Dynamic pricing CSV saved → {dp_csv_path}")
    _print_pricing_summary(stations_priced)

    # ── 4. Recommendation scenarios ──────────────────────────────────────────
    print("\n[5/5] Running recommendation scenarios …")

    # Representative EV scenarios (city, battery, SoC, target, connector)
    scenarios = [
        dict(name="Bhopal_Low_SOC_CCS2",  lat=23.2599, lon=77.4126,
             battery_kwh=40.0, soc=10.0, target_soc=80.0,
             consumption_kwh_per_km=0.15, max_dc_kw=50.0, connector="CCS2"),
        dict(name="Delhi_Mid_SOC_Type2",   lat=28.6139, lon=77.2090,
             battery_kwh=30.2, soc=35.0, target_soc=90.0,
             consumption_kwh_per_km=0.13, max_dc_kw=22.0, connector="Type2"),
        dict(name="Mumbai_High_SOC_CCS2",  lat=19.0760, lon=72.8777,
             battery_kwh=64.0, soc=65.0, target_soc=95.0,
             consumption_kwh_per_km=0.17, max_dc_kw=120.0, connector="CCS2"),
        dict(name="Pune_Emergency_CHAdeMO", lat=18.5204, lon=73.8567,
             battery_kwh=50.3, soc=8.0, target_soc=70.0,
             consumption_kwh_per_km=0.16, max_dc_kw=50.0, connector="CHAdeMO"),
    ]

    all_results = []
    all_ranked_rows = []
    all_profiles = {}

    for sc in scenarios:
        sc_name = sc.pop("name")
        print(f"  → Scenario: {sc_name}")
        try:
            ev = ev_data.from_manual(**sc)
        except ValueError as exc:
            print(f"    SKIP ({exc})")
            continue

        try:
            res = recommend(ev, stations, objective="balanced",
                            forecaster=math_fc, top_n=5,
                            match_connector=True,
                            use_dynamic_pricing=True, lambda_load=1.0)
        except Exception as exc:
            print(f"    ERROR during recommendation: {exc}")
            traceback.print_exc()
            continue

        # Gather per-scenario summary
        summary = {
            "scenario": sc_name,
            "ev": ev.to_dict(),
            "range": res["range"],
            "load_factor": res["load_factor"],
            "n_candidates": res["n_candidates"],
            "n_reachable": res["n_reachable"],
            "message": res["message"],
            "best": res["best"],
        }
        all_results.append(summary)

        # Gather ranked rows for CSV
        if not res["ranked"].empty:
            ranked_copy = res["ranked"].copy()
            ranked_copy.insert(0, "scenario", sc_name)
            all_ranked_rows.append(ranked_copy)

        # CC-CV power profile for the best station
        if res["best"]:
            best = res["best"]
            profile = cc_cv_power_profile(
                soc_start=float(best.get("soc_arrival", ev.soc)),
                soc_end=ev.target_soc,
                battery_kwh=ev.battery_kwh,
                power_kw=min(float(best.get("power_kw", ev.max_dc_kw)), ev.max_dc_kw),
            )
            all_profiles[sc_name] = {
                "soc_start": float(best.get("soc_arrival", ev.soc)),
                "soc_end": ev.target_soc,
                "battery_kwh": ev.battery_kwh,
                "power_kw": min(float(best.get("power_kw", ev.max_dc_kw)), ev.max_dc_kw),
                "n_ticks": len(profile),
                "total_time_min": round(profile[-1]["t_s"] / 60, 2),
                "profile": profile,
            }
        print(f"    {res['message']}")

    # ── 5. Write outputs ──────────────────────────────────────────────────────
    results_path = config.OUT_DIR / "pipeline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✔  pipeline_results.json  → {results_path}")

    if all_ranked_rows:
        ranked_df = pd.concat(all_ranked_rows, ignore_index=True)
        ranked_path = config.OUT_DIR / "pipeline_ranked.csv"
        ranked_df.to_csv(ranked_path, index=False)
        print(f"✔  pipeline_ranked.csv    → {ranked_path}")

    profiles_path = config.OUT_DIR / "cc_cv_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, default=str)
    print(f"✔  cc_cv_profiles.json    → {profiles_path}")

    print(f"\n{'=' * 70}")
    print("Pipeline complete.  All outputs written to outputs/")
    print(f"{'=' * 70}")


def _print_pricing_summary(df: pd.DataFrame) -> None:
    """Pretty-print a quick pricing stats table."""
    cols = ["price_per_kwh", "dynamic_price_per_kwh", "effective_price_per_kwh"]
    present = [c for c in cols if c in df.columns]
    if present:
        print(f"\n      Price statistics across {len(df)} stations:")
        print(df[present].describe().round(2).to_string())
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _add_mqtt_args(parser: argparse.ArgumentParser) -> None:
    """
    Shared MQTT arguments for hardware-ready publishing.
    These flags wire the Python output directly to an ESP32 via a broker.
    """
    # ── MQTT broker connection ──────────────────────────────────────────────
    parser.add_argument("--mqtt-host", default=None,
                        help="MQTT broker IP/hostname (e.g. 192.168.1.42). "
                             "Omit to disable MQTT publishing.")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port (default 1883).")
    # ── Topics (optional overrides) ──────────────────────────────────────────
    parser.add_argument("--mqtt-topic-prefix", default="ev",
                        help="Topic prefix; messages go to <prefix>/recommendation, etc.")


def main():
    p = argparse.ArgumentParser(
        description="EV Charging Recommender – CC-CV physics + dynamic pricing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # generate-data
    sp.add_parser("generate-data",
                  help="Write synthetic station & session CSV files."
                  ).set_defaults(f=cmd_generate)

    # train
    t = sp.add_parser("train", help="Train ML arrival forecaster + session models.")
    t.add_argument("--sessions", default=config.EPFL_SESSIONS_CSV)
    t.set_defaults(f=cmd_train)

    # recommend
    r = sp.add_parser("recommend",
                      help="Score & rank real stations for one EV (CC-CV + dynamic pricing).")
    r.add_argument("--lat", type=float, default=23.2599)
    r.add_argument("--lon", type=float, default=77.4126)
    r.add_argument("--battery", type=float, default=40.0,
                   help="Usable battery capacity in kWh (default 40).")
    r.add_argument("--soc", type=float, default=10.0,
                   help="Current SoC in % (default 10).")
    r.add_argument("--target", type=float, default=80.0,
                   help="Target SoC in % (default 80).")
    r.add_argument("--consumption", type=float, default=0.15,
                   help="Energy consumption kWh/km (default 0.15).")
    r.add_argument("--efficiency", type=float, default=0.90,
                   help="Drivetrain efficiency 0-1 (default 0.90).")
    r.add_argument("--max-dc", type=float, default=50.0,
                   help="Max DC charging power the vehicle accepts (kW, default 50).")
    r.add_argument("--connector", default="CCS2")
    r.add_argument("--any-connector", action="store_true",
                   help="Ignore connector type filter.")
    r.add_argument("--objective", default="balanced",
                   choices=["balanced", "min_distance", "min_cost", "min_time"])
    r.add_argument("--reserve", type=float, default=config.MIN_ARRIVAL_SOC,
                   help="Minimum SoC %% to retain on arrival (safety reserve).")
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--json", help="IoT JSON string or file path.")
    r.add_argument("--stations", default=config.STATIONS_CSV)
    r.add_argument("--no-forecast", action="store_true",
                   help="Disable load forecasting (use load_factor=1).")
    # ── Dynamic pricing flags ───────────────────────────────────────────────
    r.add_argument("--no-dynamic-pricing", action="store_true",
                   help="Use static price_per_kwh instead of exponential surge pricing.")
    r.add_argument("--lambda-load", type=float, default=1.0,
                   help="Surge steepness λ in P=Base*exp(λ*Load/Cap). Default 1.0.")
    # ── MQTT / ESP32 flags ──────────────────────────────────────────────────
    _add_mqtt_args(r)
    r.set_defaults(f=cmd_recommend)

    # run-pipeline  ★ NEW
    pl = sp.add_parser("run-pipeline",
                       help="Full batch pipeline: load data, run all scenarios, write outputs/.")
    _add_mqtt_args(pl)
    pl.set_defaults(f=cmd_run_pipeline)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
