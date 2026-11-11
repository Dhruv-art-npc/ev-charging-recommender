"""
app.py – Interactive Streamlit dashboard for the EV Charging Recommender
========================================================================
Launch:
    streamlit run app.py

What this file does
-------------------
* Loads India station data and (optionally) the trained Random-Forest model.
* Provides a full sidebar form for EV parameters + pricing controls.
* Runs the CC-CV physics model and exponential dynamic pricing on every
  interaction.
* Renders:
    – Top metric cards (best station KPIs)
    – CC-CV power curve (Altair area chart split by phase)
    – CC vs CV time & energy breakdown (bar + donut)
    – Ranked stations dataframe with dynamic price column
    – Interactive map (EV location + all reachable stations)
    – Forecaster comparison tab (Math Gaussian vs RF baseline)
    – Station pricing heatmap across all 424 India stations
* MQTT publisher hook is kept intact and can be enabled via sidebar toggle
  without affecting the UI when disabled.
"""
from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import math
import sys
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st

# ── make sure project root is on sys.path (works when cwd ≠ project root) ──
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── project imports ──────────────────────────────────────────────────────────
import config
from src import ev_data
from src.stations import load_stations
from src.reachability import filter_reachable, range_summary
from src.ranking import add_charging_metrics, rank_stations
from src.recommender import recommend, SHOW_COLS
from src.charging_physics import (
    cc_cv_power_profile,
    cc_cv_charging_time_min,
    integrate_cc_cv_kwh,
    cv_power_kw,
    SOC_TAPER_DEFAULT,
)
from src.dynamic_pricing import ExponentialLoadPricer, MathematicalArrivalForecaster
from src.forecasting import ArrivalForecaster

# ── optional MQTT (graceful degradation) ─────────────────────────────────────
try:
    from main import MQTTPublisher          # noqa: E402  (defined in main.py)
    _MQTT_AVAILABLE = True
except Exception:
    _MQTT_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ EV Charging Recommender",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS TWEAKS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Metric card value size */
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
/* Soft card background for metric containers */
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 12px;
    padding: 14px 18px 10px 18px;
    border: 1px solid #2d3348;
}
/* Tab font */
[data-baseweb="tab"] button { font-size: 0.9rem !important; }
/* Slim horizontal rules */
hr { border-color: #2d3348 !important; margin: 0.6rem 0 !important; }
/* Sidebar section headers */
.sidebar-section { font-size: 0.75rem; font-weight: 700;
    color: #7ec8e3; text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 0.9rem; margin-bottom: 0.1rem; }
/* Phase pill badges */
.pill-cc  { background:#1a7f3c; color:#d4f5e2; padding:2px 10px;
            border-radius:12px; font-size:0.78rem; font-weight:600; }
.pill-cv  { background:#7f4f1a; color:#f5e3d4; padding:2px 10px;
            border-radius:12px; font-size:0.78rem; font-weight:600; }
/* Recommendation banner */
.rec-banner {
    background: linear-gradient(90deg, #0d3b1e 0%, #0d2a3b 100%);
    border-left: 4px solid #2ecc71;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
.rec-banner-warn {
    background: linear-gradient(90deg, #3b1a0d 0%, #3b2a0d 100%);
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading India charging station data…")
def _load_stations() -> pd.DataFrame:
    return load_stations(config.STATIONS_CSV)


@st.cache_resource(show_spinner="Loading trained RF forecaster…")
def _load_rf_forecaster():
    p = config.MODEL_DIR / "arrival_rf.joblib"
    if p.exists():
        try:
            return ArrivalForecaster.load(p)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def _math_forecaster() -> MathematicalArrivalForecaster:
    return MathematicalArrivalForecaster()


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _phase_breakdown(soc_start: float, soc_end: float,
                      battery_kwh: float, power_kw: float,
                      taper_pct: float = SOC_TAPER_DEFAULT * 100) -> dict:
    """
    Compute CC vs CV phase time/energy breakdown for a single session.
    Returns a dict with keys: cc_min, cv_min, cc_kwh, cv_kwh,
    total_min, total_kwh, cc_pct_time, cv_pct_time.
    """
    soc_start = float(soc_start)
    soc_end   = float(soc_end)
    taper     = float(taper_pct)

    cc_end = min(soc_end, taper)   # CC phase goes up to taper or soc_end
    cv_start = max(soc_start, taper)

    cc_min = cc_kwh = cv_min = cv_kwh = 0.0

    if soc_start < cc_end:
        cc_min = cc_cv_charging_time_min(soc_start, cc_end, battery_kwh, power_kw)
        cc_kwh = integrate_cc_cv_kwh(soc_start, cc_end, battery_kwh, power_kw)

    if cv_start < soc_end:
        cv_min = cc_cv_charging_time_min(cv_start, soc_end, battery_kwh, power_kw)
        cv_kwh = integrate_cc_cv_kwh(cv_start, soc_end, battery_kwh, power_kw)

    total_min = cc_min + cv_min
    total_kwh = cc_kwh + cv_kwh
    return {
        "cc_min": round(cc_min, 2),  "cv_min": round(cv_min, 2),
        "cc_kwh": round(cc_kwh, 3),  "cv_kwh": round(cv_kwh, 3),
        "total_min": round(total_min, 2),
        "total_kwh": round(total_kwh, 3),
        "cc_pct_time": round(cc_min / total_min * 100, 1) if total_min else 0,
        "cv_pct_time": round(cv_min / total_min * 100, 1) if total_min else 0,
    }


def _build_power_curve_df(profile: list[dict]) -> pd.DataFrame:
    """Convert raw profile list to a DataFrame ready for Altair / st.area_chart."""
    df = pd.DataFrame(profile)
    df["time_min"] = df["t_s"] / 60.0
    return df


def _soc_curve_df(soc_start: float, soc_end: float,
                   battery_kwh: float, power_kw: float) -> pd.DataFrame:
    """
    Build a dense SoC-vs-time curve by integrating the CC-CV profile.
    Returns columns: time_min, soc_pct, power_kw, phase.
    """
    # Use fine 10-second ticks for a smooth curve
    profile = cc_cv_power_profile(soc_start, soc_end, battery_kwh, power_kw, dt_s=10.0)
    df = pd.DataFrame(profile)
    df["time_min"] = (df["t_s"] / 60.0).round(3)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  –  EV Parameters & Controls
# ════════════════════════════════════════════════════════════════════════════

sb = st.sidebar

sb.image("https://em-content.zobj.net/source/twitter/376/electric-plug_1f50c.png", width=48)
sb.title("EV Charging Recommender")
sb.caption("CC-CV Physics · Exponential Dynamic Pricing · Math Forecasting")
sb.divider()

# ── EV data source ───────────────────────────────────────────────────────────
sb.markdown('<p class="sidebar-section">📡 Input Source</p>', unsafe_allow_html=True)
src_mode = sb.radio("Input Source", ["Manual", "IoT JSON", "Simulated IoT"], label_visibility="collapsed")

if src_mode == "Manual":
    sb.markdown('<p class="sidebar-section">📍 Location</p>', unsafe_allow_html=True)
    lat = sb.number_input("Latitude",  value=23.2599, format="%.5f", step=0.001)
    lon = sb.number_input("Longitude", value=77.4126, format="%.5f", step=0.001)

    sb.markdown('<p class="sidebar-section">🔋 Battery</p>', unsafe_allow_html=True)
    cap  = sb.number_input("Capacity (kWh)", 10.0, 150.0, 40.0, step=0.5)
    soc  = sb.slider("Current SoC %",  1,  99, 15)
    tgt  = sb.slider("Target SoC %",   2, 100, 80)
    if tgt <= soc:
        sb.warning("Target SoC must be greater than current SoC.")
        tgt = min(soc + 1, 100)

    sb.markdown('<p class="sidebar-section">🚗 Vehicle</p>', unsafe_allow_html=True)
    cons = sb.number_input("Consumption (kWh/km)", 0.05, 0.40, 0.15, 0.01)
    eff  = sb.slider("Drivetrain efficiency", 0.50, 1.0,  0.90, 0.01)
    mdc  = sb.number_input("Max DC power (kW)", 3.0, 350.0, 50.0, 1.0)
    conn = sb.selectbox("Connector", ["CCS2", "CHAdeMO", "Type2", "GB/T", "Any"])

    try:
        ev = ev_data.from_manual(
            lat=lat, lon=lon, battery_kwh=cap, soc=soc, target_soc=tgt,
            consumption_kwh_per_km=cons, efficiency=eff, max_dc_kw=mdc,
            connector="" if conn == "Any" else conn,
        )
    except ValueError as exc:
        st.error(f"EV input error: {exc}")
        st.stop()

elif src_mode == "IoT JSON":
    default_json = json.dumps({
        "lat": 23.2599, "lon": 77.4126, "battery_kwh": 40,
        "soc": 15, "target_soc": 80, "consumption_kwh_per_km": 0.15,
    }, indent=2)
    txt = sb.text_area("Paste JSON payload", default_json, height=180)
    try:
        ev = ev_data.from_json(txt)
    except Exception as exc:
        st.error(f"JSON parse error: {exc}")
        st.stop()
    sb.success("✓ Parsed IoT JSON")

else:  # Simulated IoT
    seed = sb.number_input("Random seed", 0, 999, 42)
    ev   = ev_data.simulate_iot_reading(seed=int(seed))
    sb.json(ev.to_dict())

# ── Routing & search ─────────────────────────────────────────────────────────
sb.divider()
sb.markdown('<p class="sidebar-section">🗺️ Routing</p>', unsafe_allow_html=True)
obj     = sb.selectbox("Objective", ["balanced", "min_cost", "min_distance", "min_time"])
reserve = sb.slider("Min SoC on arrival %", 0, 20, int(config.MIN_ARRIVAL_SOC))
top_n   = sb.slider("Top N stations", 1, 10, 5)
match_conn = sb.checkbox("Match connector type", value=True)

# ── Pricing ──────────────────────────────────────────────────────────────────
sb.divider()
sb.markdown('<p class="sidebar-section">💰 Dynamic Pricing</p>', unsafe_allow_html=True)
use_dyn_pricing = sb.toggle("Exponential load pricing", value=True)
lambda_load     = sb.slider("Surge steepness λ", 0.1, 3.0, 1.0, 0.1,
                             help="P_dyn = Base × exp(λ × Load/Cap).  Higher λ = steeper surges.")
price_cap_on    = sb.checkbox("Enable price cap")
price_cap       = sb.number_input("Price cap (₹/kWh)", 10.0, 200.0, 50.0, 5.0,
                                   disabled=not price_cap_on)

# ── Forecaster ───────────────────────────────────────────────────────────────
sb.divider()
sb.markdown('<p class="sidebar-section">📈 Arrival Forecaster</p>', unsafe_allow_html=True)
rf_fc    = _load_rf_forecaster()
fc_mode  = sb.radio(
    "Forecaster",
    ["Math (Gaussian peaks)", "Random Forest (ML)", "None (load factor = 1)"],
    index=0 if rf_fc is None else 1,
)

# ── MQTT / ESP32 ─────────────────────────────────────────────────────────────
sb.divider()
sb.markdown('<p class="sidebar-section">📶 MQTT / ESP32</p>', unsafe_allow_html=True)
mqtt_on   = sb.toggle("Enable MQTT publish", value=False, disabled=not _MQTT_AVAILABLE,
                       help="Requires paho-mqtt and a running broker.")
mqtt_host = sb.text_input("Broker host", "192.168.1.42", disabled=not mqtt_on)
mqtt_port = sb.number_input("Port", 1, 65535, 1883, disabled=not mqtt_on)
if not _MQTT_AVAILABLE:
    sb.caption("⚠ paho-mqtt not installed (`pip install paho-mqtt`).")

# ── Run button ────────────────────────────────────────────────────────────────
sb.divider()
run_btn = sb.button("⚡ Find Optimal Station", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  RESOLVE FORECASTER
# ════════════════════════════════════════════════════════════════════════════

math_fc = _math_forecaster()

if fc_mode == "Math (Gaussian peaks)":
    active_fc = math_fc
elif fc_mode == "Random Forest (ML)" and rf_fc is not None:
    active_fc = rf_fc
else:
    active_fc = None


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("## ⚡ EV Charging Station Recommender")
st.caption(
    "CC-CV lithium-ion physics · Exponential load-based dynamic pricing · "
    "Mathematical & ML arrival forecasting · India station dataset"
)
st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  COMPUTE ON BUTTON PRESS (or on first load)
# ════════════════════════════════════════════════════════════════════════════

if "result" not in st.session_state or run_btn:
    with st.spinner("Running recommendation pipeline…"):
        stations_df = _load_stations()
        res = recommend(
            ev=ev,
            stations=stations_df,
            objective=obj,
            forecaster=active_fc,
            top_n=top_n,
            min_arrival_soc=reserve,
            match_connector=match_conn,
            use_dynamic_pricing=use_dyn_pricing,
            lambda_load=lambda_load,
        )
        st.session_state["result"]    = res
        st.session_state["ev"]        = ev
        st.session_state["stations"]  = stations_df
        st.session_state["fc_mode"]   = fc_mode

        # ── MQTT publish if enabled ───────────────────────────────────────
        if mqtt_on and _MQTT_AVAILABLE and res.get("best"):
            try:
                pub = MQTTPublisher(host=mqtt_host, port=int(mqtt_port))
                import time; time.sleep(0.4)
                pub.publish_recommendation(res)
                best = res["best"]
                profile_ticks = cc_cv_power_profile(
                    soc_start=float(best.get("soc_arrival", ev.soc)),
                    soc_end=ev.target_soc,
                    battery_kwh=ev.battery_kwh,
                    power_kw=min(float(best.get("power_kw", ev.max_dc_kw)), ev.max_dc_kw),
                )
                pub.publish_cc_cv_profile(profile_ticks)
                pub.disconnect()
                st.session_state["mqtt_status"] = f"✅ Published to {mqtt_host}:{mqtt_port}"
            except Exception as exc:
                st.session_state["mqtt_status"] = f"⚠ MQTT error: {exc}"
        elif mqtt_on:
            st.session_state["mqtt_status"] = "⚠ MQTT not available."

res = st.session_state["result"]
ev  = st.session_state["ev"]


# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════

(
    tab_rec,      # Recommendation
    tab_physics,  # CC-CV Breakdown
    tab_forecast, # Forecaster comparison
    tab_pricing,  # Station pricing heatmap
    tab_map,      # Map
) = st.tabs([
    "🏆 Recommendation",
    "⚡ CC-CV Physics",
    "📈 Forecaster Comparison",
    "💰 Pricing Heatmap",
    "🗺️ Map",
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 – RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════

with tab_rec:

    # ── Banner ────────────────────────────────────────────────────────────────
    if res["best"]:
        st.markdown(
            f'<div class="rec-banner">🟢 &nbsp;<b>{res["message"]}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="rec-banner-warn">🔴 &nbsp;{res["message"]}</div>',
            unsafe_allow_html=True,
        )

    # ── MQTT status ───────────────────────────────────────────────────────────
    if "mqtt_status" in st.session_state:
        st.info(st.session_state["mqtt_status"])

    # ── Top metric cards ──────────────────────────────────────────────────────
    best = res["best"] or {}
    rng  = res["range"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "🏁 Best Station",
        best.get("name", "—"),
        delta=best.get("city", ""),
        delta_color="off",
    )
    c2.metric(
        "📏 Distance",
        f"{best.get('distance_km', 0):.1f} km" if best else "—",
        delta=f"arrive {best.get('soc_arrival', 0):.1f}% SoC" if best else "",
        delta_color="normal",
    )
    c3.metric(
        "⏱ Charging Time",
        f"{best.get('charging_time_min', 0):.0f} min" if best else "—",
        delta=f"wait ~{best.get('wait_time_min', 0):.0f} min queue" if best else "",
        delta_color="inverse",
    )
    c4.metric(
        "💰 Dynamic Price",
        f"₹ {best.get('dynamic_price_per_kwh', best.get('price_per_kwh', 0)):.2f}/kWh" if best else "—",
        delta=f"base ₹{best.get('price_per_kwh', 0):.2f}" if best else "",
        delta_color="inverse",
    )
    c5.metric(
        "🔌 Station Load",
        f"{best.get('station_load', 0):.0f}%" if best else "—",
        delta=f"λ-factor {res['load_factor']:.2f}",
        delta_color="inverse",
    )
    c6.metric(
        "⚡ Total Cost",
        f"₹ {best.get('charging_cost', 0):.2f}" if best else "—",
        delta=f"{best.get('energy_to_charge_kwh', 0):.2f} kWh grid",
        delta_color="off",
    )

    # ── Range summary ─────────────────────────────────────────────────────────
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Safe range", f"{rng['safe_range_km']} km",   delta="with reserve SoC")
    r2.metric("Full range",  f"{rng['full_range_km']} km",  delta="at 0% reserve")
    r3.metric("Candidates nearby", res["n_candidates"])
    r4.metric("Reachable stations", res["n_reachable"],
              delta=f"{res['n_removed_unreachable']} removed (range)",
              delta_color="inverse")

    # ── Ranked stations table ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Ranked Stations")

    if res["ranked"].empty:
        st.warning("No reachable stations found — try relaxing the connector filter or lowering the reserve SoC.")
    else:
        display_df = res["ranked"].drop(columns=["latitude", "longitude"], errors="ignore").copy()

        # Colour-code dynamic price column
        def _highlight_load(val):
            try:
                v = float(val)
                if v > 70:  return "background-color:#4a1a1a; color:#ff9999"
                if v > 40:  return "background-color:#4a3a1a; color:#ffcc99"
                return "background-color:#1a4a2a; color:#99ffbb"
            except Exception:
                return ""

        styler = display_df.style.format(precision=2, na_rep="—")
        if hasattr(styler, "map"):
            styler = styler.map(_highlight_load, subset=["station_load"] if "station_load" in display_df.columns else [])
        else:
            styler = styler.applymap(_highlight_load, subset=["station_load"] if "station_load" in display_df.columns else [])
        styled = (
            styler
            .bar(subset=["charging_time_min"] if "charging_time_min" in display_df.columns else [],
                 color=["#1a3d6e", "#2a5d9e"])
            .bar(subset=["charging_cost"] if "charging_cost" in display_df.columns else [],
                 color=["#4a1a1a", "#8e3a3a"])
        )
        st.dataframe(styled, use_container_width=True, height=min(40 + 36 * len(display_df), 380))

        # ── Comparison bar chart: total_time vs components ────────────────────
        st.subheader("⏰ Time Breakdown per Station")
        ranked = res["ranked"].copy()
        time_cols = ["travel_time_min", "wait_time_min", "charging_time_min"]
        present   = [c for c in time_cols if c in ranked.columns]
        if present:
            chart_df = ranked[["name"] + present].set_index("name")
            chart_df.columns = ["Travel", "Queue wait", "Charging"]
            st.bar_chart(chart_df, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 – CC-CV PHYSICS BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

with tab_physics:

    best = res["best"] or {}
    if not best:
        st.info("Run a recommendation first (click **Find Optimal Station** in the sidebar).")
        st.stop()

    # Determine the actual charging window for the best station
    soc_arr   = float(best.get("soc_arrival", ev.soc))
    soc_tgt   = ev.target_soc
    bat_kwh   = ev.battery_kwh
    station_kw = float(best.get("power_kw", ev.max_dc_kw))
    eff_p_kw  = min(station_kw, ev.max_dc_kw)

    taper_pct = SOC_TAPER_DEFAULT * 100   # 80 %

    bd = _phase_breakdown(soc_arr, soc_tgt, bat_kwh, eff_p_kw, taper_pct)

    st.subheader(f"⚡ CC-CV Analysis — {best.get('name', 'Best Station')}")
    st.caption(
        f"Session: **{soc_arr:.1f}% → {soc_tgt:.1f}% SoC** · "
        f"Battery: **{bat_kwh} kWh** · "
        f"Station power: **{station_kw} kW** · "
        f"Vehicle max DC: **{ev.max_dc_kw} kW** · "
        f"Effective: **{eff_p_kw} kW**"
    )
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("🟢 CC time",      f"{bd['cc_min']:.1f} min",
              delta=f"{bd['cc_pct_time']:.0f}% of session")
    p2.metric("🟠 CV time",      f"{bd['cv_min']:.1f} min",
              delta=f"{bd['cv_pct_time']:.0f}% of session", delta_color="inverse")
    p3.metric("⚡ CC energy",    f"{bd['cc_kwh']:.2f} kWh")
    p4.metric("⚡ CV energy",    f"{bd['cv_kwh']:.2f} kWh")
    p5.metric("🕐 Total time",   f"{bd['total_min']:.1f} min")
    p6.metric("🔌 Total grid kWh", f"{bd['total_kwh']:.2f} kWh")

    # ── Phase badges ──────────────────────────────────────────────────────────
    st.markdown(
        f'CC phase → SoC {soc_arr:.1f}% → {min(soc_tgt, taper_pct):.0f}% &nbsp;'
        f'<span class="pill-cc">CONSTANT CURRENT&nbsp;{eff_p_kw:.0f} kW</span>'
        f'&nbsp;&nbsp;&nbsp;'
        f'CV phase → SoC {taper_pct:.0f}% → {soc_tgt:.1f}% &nbsp;'
        f'<span class="pill-cv">EXPONENTIAL TAPER</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Power curve chart ─────────────────────────────────────────────────────
    st.subheader("📉 Instantaneous Charging Power vs Time")

    curve_df = _soc_curve_df(soc_arr, soc_tgt, bat_kwh, eff_p_kw)

    # Build separate series for CC and CV so they render as different colours
    cc_df = curve_df[curve_df["phase"] == "CC"][["time_min", "power_kw"]].rename(
        columns={"power_kw": "CC power (kW)"})
    cv_df = curve_df[curve_df["phase"] == "CV"][["time_min", "power_kw"]].rename(
        columns={"power_kw": "CV power (kW)"})

    # Merge on time_min for the chart
    merged = pd.merge(cc_df, cv_df, on="time_min", how="outer").sort_values("time_min")

    # Add a vertical marker at the taper transition time
    taper_time = curve_df[curve_df["phase"] == "CV"]["time_min"].min() \
        if (curve_df["phase"] == "CV").any() else None

    st.line_chart(
        merged.set_index("time_min"),
        height=320,
        use_container_width=True,
        color=["#2ecc71", "#e67e22"],
    )

    if taper_time is not None:
        st.caption(
            f"🔵 CC → CV transition at **t = {taper_time:.1f} min** "
            f"(SoC = {taper_pct:.0f}%). "
            f"Power drops exponentially from {eff_p_kw:.0f} kW to "
            f"~{cv_power_kw(soc_tgt, eff_p_kw):.1f} kW at target SoC."
        )

    st.divider()

    # ── SoC vs Time curve ─────────────────────────────────────────────────────
    st.subheader("🔋 State-of-Charge vs Time")
    soc_time_df = curve_df[["time_min", "soc_pct"]].rename(
        columns={"soc_pct": "SoC (%)"})
    st.line_chart(soc_time_df.set_index("time_min"), height=280,
                  use_container_width=True, color=["#3498db"])

    st.divider()

    # ── CC vs CV breakdown bars ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("⏱ Time Split")
        time_split = pd.DataFrame({
            "Phase": ["CC (Constant Current)", "CV (Taper)"],
            "Minutes": [bd["cc_min"], bd["cv_min"]],
        })
        st.bar_chart(time_split.set_index("Phase"), height=240,
                     use_container_width=True, color=["#2ecc71"])

    with col_b:
        st.subheader("⚡ Energy Split (grid kWh)")
        energy_split = pd.DataFrame({
            "Phase": ["CC energy", "CV energy"],
            "kWh": [bd["cc_kwh"], bd["cv_kwh"]],
        })
        st.bar_chart(energy_split.set_index("Phase"), height=240,
                     use_container_width=True, color=["#e67e22"])

    # ── Physics explainer ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 CC-CV Physics — How it works", expanded=False):
        st.markdown(f"""
**Constant Current (CC) Phase** — SoC 0 % → {taper_pct:.0f} %
> The charger delivers full rated power **P_cc = {eff_p_kw:.0f} kW** at a
> constant current.  Charging is fastest and most grid-efficient here.

**Constant Voltage (CV) Phase** — SoC {taper_pct:.0f} % → 100 %
> The charger holds voltage constant; current (and power) decays exponentially
> to prevent lithium plating:

$$P(\\text{{SoC}}) = P_{{cc}} \\cdot \\exp\\!\\left(-k \\cdot
    \\frac{{\\text{{SoC}} - {taper_pct:.0f}\\%}}{{100\\% - {taper_pct:.0f}\\%}}\\right)$$

where **k = ln(10) ≈ 2.302**, so power reaches ~10 % of P_cc at 100 % SoC.

**Practical implication:** charging from {taper_pct:.0f} % to {soc_tgt:.0f} %
takes {bd['cv_min']:.1f} min ({bd['cv_pct_time']:.0f}% of the session)
for only {bd['cv_kwh']:.2f} kWh — a significant time penalty if your target
SoC is above {taper_pct:.0f} %.  Consider stopping at {taper_pct:.0f} % for
long road trips to minimise total charge-stop duration.
""")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 – FORECASTER COMPARISON
# ════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.subheader("📈 Arrival Forecaster Comparison")
    st.caption(
        "Compare the Mathematical (Gaussian double-peak) forecaster against "
        "the trained Random-Forest baseline on a 24-hour window."
    )

    hrs = st.slider("Forecast horizon (hours)", 6, 72, 24, key="fc_hrs")
    now_h = pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1)

    # ── Mathematical forecaster ───────────────────────────────────────────────
    math_series = math_fc.forecast(start=now_h, hours=hrs)
    math_lf     = math_fc.load_factor(now_h)

    # ── RF forecaster (if available) ─────────────────────────────────────────
    rf_series = None
    rf_lf     = None
    if rf_fc is not None:
        try:
            rf_series = rf_fc.forecast(start=now_h, hours=hrs)
            rf_lf     = rf_fc.load_factor(now_h)
        except Exception as exc:
            st.warning(f"RF forecast failed: {exc}")

    # ── Build comparison DataFrame ────────────────────────────────────────────
    fc_df = pd.DataFrame({"Math (Gaussian)": math_series})
    if rf_series is not None:
        fc_df["Random Forest (ML)"] = rf_series
    fc_df.index = fc_df.index.strftime("%H:%M %a")

    # ── Side-by-side metric summary ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Math — load factor now", f"{math_lf:.3f}",
              delta="peak" if math_lf > 1 else "off-peak",
              delta_color="inverse" if math_lf > 1 else "normal")
    m2.metric("Math — peak arrivals/h",
              f"{math_series.max():.2f}", delta=f"at {math_series.idxmax()}")

    if rf_lf is not None:
        m3.metric("RF — load factor now", f"{rf_lf:.3f}",
                  delta="peak" if rf_lf > 1 else "off-peak",
                  delta_color="inverse" if rf_lf > 1 else "normal")
        m4.metric("RF — peak arrivals/h",
                  f"{rf_series.max():.2f}", delta=f"at {rf_series.idxmax()}")
    else:
        m3.metric("RF — load factor", "N/A", delta="run `python main.py train` first")
        m4.metric("RF — peak arrivals/h", "N/A")

    st.divider()

    # ── Combined line chart ───────────────────────────────────────────────────
    colors = ["#3498db", "#e74c3c"] if rf_series is not None else ["#3498db"]
    st.line_chart(fc_df, height=340, use_container_width=True, color=colors)

    # ── Side-by-side toggle: same data as a table ────────────────────────────
    with st.expander("📋 Raw forecast values (table)", expanded=False):
        st.dataframe(fc_df.style.format("{:.3f}"), use_container_width=True)

    # ── RF model metrics ──────────────────────────────────────────────────────
    if rf_fc is not None and rf_fc.metrics:
        st.divider()
        st.subheader("🧪 RF Model Quality Metrics (from training)")
        m = rf_fc.metrics
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            st.markdown("**Random Forest vs baselines**")
            metrics_rows = []
            for key in ("random_forest", "baseline_same_hour_yesterday",
                        "baseline_same_hour_last_week", "baseline_train_mean"):
                if key in m:
                    metrics_rows.append({"Model": key.replace("_", " ").title(),
                                         **m[key]})
            if metrics_rows:
                st.dataframe(pd.DataFrame(metrics_rows).set_index("Model")
                             .style.format(precision=4), use_container_width=True)
        with col_rf2:
            st.markdown("**Top feature importances**")
            if "top_features" in m:
                fi = pd.Series(m["top_features"], name="Importance").sort_values(ascending=True)
                st.bar_chart(fi, height=240, use_container_width=True)

    # ── Mathematical forecaster explainer ─────────────────────────────────────
    st.divider()
    with st.expander("📖 Mathematical Forecaster — How it works", expanded=False):
        st.markdown(r"""
The **Mathematical Arrival Forecaster** needs **zero training data**.
It models the hourly EV arrival rate as a sum of two Gaussian peaks:

$$\lambda(h) = A_{AM} \cdot e^{-\frac{(h - \mu_{AM})^2}{2\sigma_{AM}^2}}
             + A_{PM} \cdot e^{-\frac{(h - \mu_{PM})^2}{2\sigma_{PM}^2}}$$

Scaled by:
- **Weekend factor** (0.65×, Sat/Sun are quieter)
- **Seasonal cosine envelope** (±15 %, peaking in Indian EV season Dec–Jan)

The resulting **load factor** = predicted arrivals / long-run hourly mean,
clipped to [0.5, 2.0], and used to adjust station queue time.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| μ_AM      | 08:00   | Morning commute peak |
| μ_PM      | 18:00   | Evening commute peak |
| σ_AM      | 1.5 h   | AM spread            |
| σ_PM      | 2.0 h   | PM spread            |
| A_AM      | 4.5     | AM amplitude         |
| A_PM      | 6.0     | PM amplitude         |
""")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 – PRICING HEATMAP
# ════════════════════════════════════════════════════════════════════════════

with tab_pricing:
    st.subheader("💰 Station Pricing Heatmap — All India Stations")
    st.caption(
        f"Exponential dynamic pricing applied across all 424 stations.  "
        f"λ = {lambda_load:.1f} · "
        f"Session window: SoC {ev.soc:.0f}% → {ev.target_soc:.0f}%"
    )

    stations_all = st.session_state.get("stations", _load_stations())

    # Apply dynamic pricing across entire dataset
    cap_val = price_cap if price_cap_on else None
    pricer  = ExponentialLoadPricer(lambda_load=lambda_load, max_price_cap=cap_val)
    priced  = pricer.apply_to_stations(stations_all,
                                        soc_start=ev.soc,
                                        soc_end=ev.target_soc)

    # ── Summary metrics ────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Avg base ₹/kWh",    f"₹ {priced['price_per_kwh'].mean():.2f}")
    h2.metric("Avg dynamic ₹/kWh", f"₹ {priced['dynamic_price_per_kwh'].mean():.2f}",
              delta=f"+{(priced['dynamic_price_per_kwh'] - priced['price_per_kwh']).mean():.2f} surge",
              delta_color="inverse")
    h3.metric("Max dynamic ₹/kWh", f"₹ {priced['dynamic_price_per_kwh'].max():.2f}")
    h4.metric("Min dynamic ₹/kWh", f"₹ {priced['dynamic_price_per_kwh'].min():.2f}",
              delta_color="normal")

    st.divider()

    # ── Scatter: base price vs dynamic price, coloured by load ───────────────
    col_sc, col_hi = st.columns([3, 2])
    with col_sc:
        st.markdown("**Base → Dynamic price by station load (%)**")
        scatter_df = priced[["name", "city", "price_per_kwh",
                               "dynamic_price_per_kwh", "current_load_pct"]].copy()
        scatter_df = scatter_df.rename(columns={
            "price_per_kwh": "Base ₹/kWh",
            "dynamic_price_per_kwh": "Dynamic ₹/kWh",
            "current_load_pct": "Load %",
        })
        st.scatter_chart(
            scatter_df,
            x="Base ₹/kWh",
            y="Dynamic ₹/kWh",
            color="Load %",
            size="Load %",
            height=360,
            use_container_width=True,
        )

    with col_hi:
        st.markdown("**Top 15 most expensive stations (dynamic price)**")
        top15 = (priced[["name", "city", "current_load_pct",
                           "price_per_kwh", "dynamic_price_per_kwh"]]
                 .sort_values("dynamic_price_per_kwh", ascending=False)
                 .head(15)
                 .rename(columns={
                     "current_load_pct": "Load %",
                     "price_per_kwh": "Base",
                     "dynamic_price_per_kwh": "Dynamic",
                 }))
        st.dataframe(top15.reset_index(drop=True).style.format(
            {"Load %": "{:.1f}", "Base": "₹{:.2f}", "Dynamic": "₹{:.2f}"}),
            use_container_width=True, height=360)

    st.divider()

    # ── Distribution of surge multipliers ─────────────────────────────────────
    st.markdown("**Price surge distribution (Dynamic / Base ratio)**")
    priced["Surge multiplier"] = priced["dynamic_price_per_kwh"] / priced["price_per_kwh"]
    hist_df = priced["Surge multiplier"].value_counts(bins=20, sort=False).reset_index()
    hist_df.columns = ["Surge multiplier range", "Count"]
    hist_df["Surge multiplier range"] = hist_df["Surge multiplier range"].astype(str)
    st.bar_chart(hist_df.set_index("Surge multiplier range"), height=240,
                 use_container_width=True)

    st.divider()
    with st.expander("📋 Full pricing table (all stations)", expanded=False):
        display_cols = ["name", "city", "operator", "power_kw",
                        "current_load_pct", "price_per_kwh",
                        "dynamic_price_per_kwh", "effective_price_per_kwh"]
        present = [c for c in display_cols if c in priced.columns]
        st.dataframe(
            priced[present].sort_values("dynamic_price_per_kwh", ascending=False)
            .reset_index(drop=True)
            .style.format(precision=2),
            use_container_width=True,
            height=400,
        )

    # ── Pricing formula explainer ─────────────────────────────────────────────
    with st.expander("📖 Exponential pricing formula", expanded=False):
        st.markdown(rf"""
**Dynamic price formula:**

$$P_{{\text{{dynamic}}}} = P_{{\text{{base}}}} \times e^{{\lambda \cdot \frac{{\text{{Load}}}}{{\text{{Capacity}}}}}}$$

**Effective cost per kWh** additionally weights the price by the average
inverse power-fraction over the charging window (accounting for the CV-phase
efficiency penalty):

$$P_{{\text{{eff}}}} = P_{{\text{{dynamic}}}} \times \overline{{
    \frac{{1}}{{P_{{cc}}(\text{{SoC}}) / P_{{cc}}}}
}}_{{\text{{SoC}} \in [{ev.soc:.0f}\%, {ev.target_soc:.0f}\%]}}$$

where $P_{{cc}}(\text{{SoC}})$ follows the CC-CV exponential taper.

| λ | Load 50% multiplier | Load 95% multiplier |
|---|---------------------|---------------------|
| 0.5 | ×1.28 | ×1.61 |
| 1.0 | ×1.65 | ×2.59 |
| 1.5 | ×2.12 | ×4.16 |
| 2.0 | ×2.72 | ×6.69 |

Current λ setting: **{lambda_load:.1f}**
""")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 – MAP
# ════════════════════════════════════════════════════════════════════════════

with tab_map:
    st.subheader("🗺️ Station Map")

    ranked = res.get("ranked", pd.DataFrame())
    all_st = st.session_state.get("stations", _load_stations())

    map_layer = st.radio(
        "Show stations",
        ["Reachable (ranked)", "All India stations"],
        horizontal=True,
    )

    if map_layer == "Reachable (ranked)" and not ranked.empty:
        map_df = ranked[["latitude", "longitude"]].copy()
    else:
        map_df = all_st[["latitude", "longitude"]].copy()

    # Add EV location as a row
    ev_point = pd.DataFrame([{"latitude": ev.lat, "longitude": ev.lon}])
    combined = pd.concat([ev_point, map_df], ignore_index=True)

    st.map(combined, zoom=10)

    if not ranked.empty:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("EV location", f"{ev.lat:.4f}°N, {ev.lon:.4f}°E")
        col_m2.metric("Closest reachable station",
                      f"{ranked.iloc[0]['name'] if 'name' in ranked.columns else '—'}  "
                      f"({ranked.iloc[0]['distance_km']:.1f} km)")

    st.caption(
        "🔵 Blue dot = your EV.  All other dots = stations.  "
        "Zoom in for individual station positions."
    )


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "EV Charging Recommender · CC-CV physics: `src/charging_physics.py` · "
    "Dynamic pricing: `src/dynamic_pricing.py` · "
    "Stations: `data/india_charging_stations.csv` · "
    "MQTT hook: `main.MQTTPublisher` · "
    "Built with Streamlit"
)
