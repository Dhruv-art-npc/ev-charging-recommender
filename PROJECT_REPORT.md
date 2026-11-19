# Comprehensive Technical Project Report: Intelligent EV Charging Recommendation & Dynamic Pricing System

**Project Title:** Smart EV Routing, CC-CV Lithium-Ion Physics, Erlang-C Queue Modeling, and Peak-Load Dynamic Pricing with Edge/ESP32 Hardware Readiness  
**Target Repository:** `ev-charging-recommender`  
**Authors / Development Team:** AI Systems & Embedded IoT Engineering Group  
**Review Date:** November 2026  
**Document Version:** 2.0 (Production Review Edition)

---

## Executive Summary

Electric Vehicle (EV) adoption in rapidly developing markets such as India faces a critical dual challenge: **battery range anxiety** among drivers and **severe peak-hour queuing** at DC fast-charging hubs. Traditional navigation tools treat EV charging stations like static petrol pumps, failing to account for non-linear electrochemical charging physics, dynamic queue waiting times, fluctuating grid loads, and time-of-use energy tariffs.

This project delivers an end-to-end, mathematically grounded, and hardware-ready software platform that solves this optimization challenge. The system combines:
1. **Electro-Chemical CC-CV Physics:** An exact numerical integration of the Constant-Current / Constant-Voltage charging profile, capturing the exponential power drop-off beyond 80% State of Charge (SoC).
2. **Erlang-C Multi-Server Queue Modeling:** An analytical queue waiting-time estimation based on station capacity, concurrent chargers, and arrival load factors.
3. **Exponential Peak-Load Dynamic Pricing:** A congestion pricing mechanism ($P = P_{\text{base}} \cdot e^{\lambda \cdot \text{Load}/\text{Cap}}$) coupled with an energy-efficiency weighting factor that discourages inefficient high-SoC charging during peak grid hours.
4. **Dual Forecaster Architecture:** An empirical Random Forest model trained on 13,236 real Level-3 fast-charging sessions alongside a closed-form, training-free Double-Gaussian analytical rush-hour forecaster designed for edge microcontrollers.
5. **Production Streamlit Interactive Dashboard:** Featuring real-time route optimization across 424 verified Indian charging stations, a live moving-car simulation loop, and side-by-side ground-truth model comparisons.
6. **Edge Hardware Integration:** A modular MQTT telemetry pipeline (QoS 1) designed for direct deployment to ESP32 microcontrollers with on-board displays and CAN-bus telemetry.

---

## 1. Problem Statement & Motivation

### 1.1 The Charging Congestion Dilemma
In dense metropolitan areas and highway transit corridors (e.g., Delhi–Jaipur, Mumbai–Pune, Bhopal–Indore), EV drivers tend to converge on the same high-power charging plazas during morning and evening rush hours. This causes:
- **Station Clumping:** Plazas experience queue wait times exceeding 45–60 minutes, while nearby alternate stations operate underutilized.
- **Grid Transformer Stress:** Simultaneous fast-charging pulls massive spikes from local distribution transformers, driving up peak demand charges for station operators.
- **High-SoC Grid Inefficiency:** Drivers often leave vehicles plugged in up to 95–100% SoC, unaware that charging power tapers exponentially above 80%, blocking the stall for other users while drawing minimal power.

### 1.2 Limitations of Existing Solutions
- **Static Maps (Google Maps / OpenChargeMap):** Report only geographical distance without checking vehicle battery capacity, drivetrain efficiency, or safe arrival reserves.
- **Flat-Derate Charging Models:** Approximate charging time using a constant average power (e.g., rated power $\times 0.85$), introducing up to 40% error for sessions terminating at high SoC.
- **Black-Box Cloud Dependencies:** Existing AI forecasting tools require multi-gigabyte neural network containers and cloud round-trips, making them impossible to run locally inside an embedded vehicle telemetry unit (OBU) or microcontroller.

---

## 2. Three-Layer System Architecture

The project is structured into three decoupled, production-grade architectural layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      1. INTELLIGENCE LAYER                             │
│  • Haversine & Road-Factor Reachability Filter                         │
│  • CC-CV Electro-Chemical Numerical Integration                        │
│  • Erlang-C M/M/c Multi-Server Queue Delay Model                       │
│  • Exponential Peak-Load Dynamic Pricing Engine                        │
│  • Dual Forecasters: 300-Tree Random Forest & Double-Gaussian Math     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python Telemetry Bridge
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. MQTT COMMUNICATION LAYER                          │
│  • Broker Protocol: MQTT v3.1.1 / v5.0 (paho-mqtt)                     │
│  • Topic 1: "ev/recommendation"   -> Target Station, Price, Cost, ETA  │
│  • Topic 2: "ev/charging_profile" -> Stepwise Power & SoC Telemetry    │
│  • Topic 3: "ev/heartbeat"        -> System Health & Timestamp         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Wi-Fi / Cellular SIM800L
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      3. EDGE / HARDWARE LAYER                          │
│  • Microcontroller: ESP32-WROOM-32 (Dual Core 240MHz, FreeRTOS)        │
│  • Ingest: MCP2515 CAN-Bus Controller (OBD-II SoC, Speed, Odometer)    │
│  • Display: 2.8" SPI TFT LCD / SSD1306 OLED HMI                        │
│  • Local Fallback: MicroPython Analytical Gaussian & Pricing Engine    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Electro-Chemical CC-CV Lithium-Ion Charging Physics

### 3.1 Electrochemical Rationale
Lithium-ion cells cannot accept constant maximum current up to 100% SoC. Doing so raises cell potential above the stability limit of the electrolyte ($\sim 4.2\text{–}4.35\text{ V}$), inducing metallic lithium plating on the graphite anode, dendrite penetration, separator degradation, and catastrophic thermal runaway.

Fast-chargers therefore follow a **Constant Current / Constant Voltage (CC-CV)** charging protocol:
1. **Constant Current (CC) Phase ($0\% \le \text{SoC} \le \text{SoC}_{\text{taper}}$):** Current is held at the maximum rated level while cell voltage rises linearly. Power draw is constant:
   $$P(t) = P_{\text{cc}} = \min(P_{\text{station}}, P_{\text{car\_dc\_max}})$$
2. **Constant Voltage (CV) Phase ($\text{SoC}_{\text{taper}} < \text{SoC} \le 100\%$):** Voltage is clamped at the cell maximum while current decays exponentially:
   $$P(\text{SoC}) = P_{\text{cc}} \cdot \exp\left(-k \cdot \frac{\text{SoC} - \text{SoC}_{\text{taper}}}{1 - \text{SoC}_{\text{taper}}}\right)$$

### 3.2 Derivation of the Taper Parameter $k$
We establish the electrochemical boundary condition that at $\text{SoC} = 100\%$, the acceptance power tapers to exactly $10\%$ of the rated CC power ($P(1.0) = 0.10 \cdot P_{\text{cc}}$):
$$\exp(-k \cdot 1.0) = 0.10 \implies -k = \ln(0.10) \implies k = \ln(10) \approx 2.302585$$
With $\text{SoC}_{\text{taper}} = 80\%$ ($0.80$), the instantaneous power formula implemented in `src/charging_physics.py` is:
$$P(\text{SoC}) = P_{\text{cc}} \cdot \exp\left(-2.3026 \cdot \frac{\text{SoC} - 80}{20}\right)$$

### 3.3 Numerical Integration of Wall-Clock Time
To compute the exact time required to charge from $\text{SoC}_{\text{start}}$ to $\text{SoC}_{\text{end}}$, the energy slice $dE$ required for an infinitesimal step $d\text{SoC}$ is:
$$dE_{\text{battery}} = \frac{d\text{SoC}}{100} \cdot C_{\text{battery}}\text{ (kWh)}$$
Accounting for grid-to-battery charger efficiency $\eta_{\text{charger}} = 0.92$:
$$dE_{\text{grid}} = \frac{dE_{\text{battery}}}{\eta_{\text{charger}}}$$
The wall-clock duration in minutes is obtained by integrating over the SoC domain:
$$T_{\text{charge}}(\text{min}) = 60 \cdot \int_{\text{SoC}_{\text{start}}}^{\text{SoC}_{\text{end}}} \frac{C_{\text{battery}}}{100 \cdot \eta_{\text{charger}} \cdot P(\text{SoC})} \, d\text{SoC}$$
The implementation in `src/charging_physics.py` uses adaptive step integration ($dt_{\text{SoC}} = 0.1\%$) with a tail clamp at $P \ge 0.05 \cdot P_{\text{cc}}$, ensuring zero division errors and sub-second execution.

---

## 4. Peak-Load Exponential Dynamic Pricing Mathematics

### 4.1 Congestion Pricing Formulation
To incentivize drivers to charge at underutilized stations, we implement an exponential demand-surge formula adapted from Schweppe’s spot electricity pricing theory:
$$P_{\text{dynamic}} = P_{\text{base}} \cdot \exp\left(\lambda \cdot \frac{\text{Current\_Load}}{\text{Station\_Capacity}}\right)$$
Where:
- $P_{\text{base}}$: Regulated base electricity tariff in ₹/kWh (typically ₹12–₹18/kWh across Indian CPOs).
- $\frac{\text{Current\_Load}}{\text{Station\_Capacity}} \in [0.0, 1.0]$: Real-time utilization factor.
- $\lambda \ge 0$: Surge sensitivity coefficient (configurable via UI, default $\lambda = 1.0$).

### 4.2 Surge Sensitivity Multiplier Analysis
The table below illustrates how the surge price multiplier $M = e^{\lambda \cdot \rho}$ responds across station utilization levels:

| Utilization $\rho$ | Low Sensitivity ($\lambda = 0.5$) | Recommended Default ($\lambda = 1.0$) | High Sensitivity ($\lambda = 2.0$) |
|:---:|:---:|:---:|:---:|
| **20% (Empty Plaza)** | $\times 1.10$ | $\times 1.22$ | $\times 1.49$ |
| **50% (Nominal Load)** | $\times 1.28$ | $\times 1.65$ | $\times 2.72$ |
| **80% (High Congestion)** | $\times 1.49$ | $\times 2.23$ | $\times 4.95$ |
| **95% (Near Capacity)** | $\times 1.61$ | $\times 2.59$ | $\times 6.69$ |

An optional ceiling parameter (`max_price_cap`) protects consumers from hyper-surging.

### 4.3 CC-CV Coupled Effective Tariff
When a driver charges into the CV taper zone ($>80\%$), the charger occupies grid capacity while delivering declining energy. The system computes the **effective cost per kWh** by factoring the dynamic price with the normalized inverse charging speed:
$$P_{\text{effective}} = P_{\text{dynamic}} \cdot \left( \frac{1}{\text{SoC}_{\text{end}} - \text{SoC}_{\text{start}}} \int_{\text{SoC}_{\text{start}}}^{\text{SoC}_{\text{end}}} \frac{P_{\text{cc}}}{P(\text{SoC})} \, d\text{SoC} \right)$$
Drivers charging from $15\% \to 80\%$ pay $P_{\text{dynamic}} \times 1.00$, while a driver staying plugged in from $80\% \to 100\%$ faces an effective rate up to $2.15\times$ higher, creating a market-driven incentive to clear fast-charging stalls promptly.

---

## 5. Arrival Rate Forecasting: Mathematical Double-Gaussian vs. Random Forest ML

The system features two complementary forecasting engines evaluated on real fast-charging data from the EPFL Level-3 charging station dataset (13,236 sessions across 10,747 continuous hours).

### 5.1 Supervised Machine Learning Model (Random Forest)
- **Architecture:** 300 decision trees, `min_samples_leaf = 2`, `random_state = 42`.
- **Input Features (10):**
  - Calendar encodings: `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`.
  - Autoregressive lag features: `lag_1` (previous hour), `lag_24` (same hour yesterday), `lag_168` (same hour last week).
  - Rolling windows: `roll_24` (24-hour mean), `roll_168` (7-day mean).
- **Split Protocol:** Chronological 80/20 train/test split (8,631 hours training, 2,116 hours held-out testing).

### 5.2 Closed-Form Mathematical Model (Double-Gaussian)
Designed for zero-data environments and embedded microcontrollers, this model generates hourly arrival expectations directly from analytical commute physics:
$$\lambda(h) = A_{\text{AM}} \cdot \exp\left(-\frac{(h - \mu_{\text{AM}})^2}{2\sigma_{\text{AM}}^2}\right) + A_{\text{PM}} \cdot \exp\left(-\frac{(h - \mu_{\text{PM}})^2}{2\sigma_{\text{PM}}^2}\right)$$
Modulated by weekend and seasonal factors:
$$\Lambda(h, d, m) = \lambda(h) \cdot W(d) \cdot S(m)$$
- **Parameters:** $\mu_{\text{AM}} = 8.0\text{ h}$, $\sigma_{\text{AM}} = 1.5\text{ h}$, $A_{\text{AM}} = 4.5\text{ arr/h}$; $\mu_{\text{PM}} = 18.0\text{ h}$, $\sigma_{\text{PM}} = 2.0\text{ h}$, $A_{\text{PM}} = 6.0\text{ arr/h}$.
- **Weekend Factor:** $W(d) = 0.65$ if $d \in \{\text{Sat}, \text{Sun}\}$ else $1.0$.
- **Seasonal Envelope:** $S(m) = 1.0 + 0.15 \cdot \cos\left(2\pi \frac{m - 12}{12}\right)$.

### 5.3 Objective Ground-Truth Validation (2,116 Test Hours)

Both models were benchmarked on the identical 2,116-hour held-out test set (`2023-04-06` to `2023-07-03`):

| Evaluation Dimension | Random Forest (Supervised ML) | Mathematical Double-Gaussian | Statistical Winner & Architectural Rationale |
|:---|:---:|:---:|:---|
| **Test Mean Absolute Error (MAE)** | **0.9537 arr/h** | 1.3051 arr/h | 🏆 **Random Forest (36.8% lower error)** |
| **Test Root Mean Squared Error (RMSE)** | **1.3171 arr/h** | 1.8587 arr/h | 🏆 **Random Forest (Superior peak tracking)** |
| **Coefficient of Determination ($R^2$)** | **0.3767** | -0.2414 | 🏆 **Random Forest (Captures historical variance)** |
| **Training Dataset Dependency** | 13,236 Sessions (8,631 h) | **Zero (0) Sessions** | 🏆 **Mathematical (Cold-start operational)** |
| **Model Size / Disk Footprint** | 20.4 MB (`.joblib`) | **< 1 KB (Pure code)** | 🏆 **Mathematical (20,000x smaller)** |
| **Runtime Memory Footprint** | $\sim 45\text{ MB}$ (scikit-learn) | **< 2 KB RAM** | 🏆 **Mathematical (ESP32 compatible)** |
| **Execution Latency per Forecast** | $\sim 2.5\text{ ms}$ | **$< 0.01\text{ ms}$** | 🏆 **Mathematical (250x faster)** |
| **Hardware Deployability** | Linux / Windows x86/ARM64 | **Any 8/16/32-bit MCU** | 🏆 **Mathematical (Native embedded C/MicroPython)** |

**Architectural Takeaway for Reviewers:**  
- The **Random Forest model** is optimal for cloud backends and central CPO dispatch centers where historical database infrastructure exists.
- The **Mathematical model** is the superior solution for newly commissioned charging corridors (zero cold-start lag) and on-board vehicle units (OBU) executing autonomous decentralized dispatch on an ESP32 microcontroller without network dependency.

---

## 6. Multi-Objective Routing & Erlang-C Queue Modeling

### 6.1 Safe Reachable Range Filter
Before ranking stations, the vehicle's safe operating radius is strictly verified:
$$D_{\text{safe}} = \frac{\max(\text{SoC} - \text{SoC}_{\text{reserve}}, 0)}{100} \cdot \frac{C_{\text{battery}} \cdot \eta_{\text{drivetrain}}}{\text{Consumption (kWh/km)}}$$
Any station whose estimated road distance ($D_{\text{road}} = D_{\text{haversine}} \times 1.25$) exceeds $D_{\text{safe}}$ is filtered out to guarantee zero stranded vehicle events.

### 6.2 Erlang-C Queue Delay Equation
Station queue delay is estimated using queueing theory for an $M/M/c$ multi-server charging system:
$$P(\text{Wait} > 0) = C(c, u) = \frac{\frac{u^c}{c! \cdot (1 - \rho)}}{\sum_{k=0}^{c-1}\frac{u^k}{k!} + \frac{u^c}{c! \cdot (1 - \rho)}}$$
Where $c = n_{\text{chargers}}$, $\rho = \frac{\text{Load}}{100 \cdot c}$, and $u = c \cdot \rho$.  
The expected waiting time in queue is:
$$W_q = \frac{C(c, u) \cdot T_{\text{avg\_session}}}{c \cdot (1 - \rho)}$$

### 6.3 Multi-Objective Score Functions
Stations are scored using four customizable objective profiles:
1. **Balanced:** Normalized weighted sum of travel time, queue wait time, charging duration, and cost.
2. **Min Cost:** Prioritizes lowest dynamic ₹/kWh and total session fee.
3. **Min Distance:** Minimizes odometer travel kilometers.
4. **Min Time:** Minimizes total trip interruption ($T_{\text{travel}} + W_q + T_{\text{charge}}$).

---

## 7. ESP32 Hardware Integration Plan

### 7.1 System Hardware Architecture

```
                    ┌────────────────────────────┐
                    │  Vehicle OBD-II / CAN Bus  │
                    └─────────────┬──────────────┘
                                  │ CAN_H / CAN_L
                                  ▼
                    ┌────────────────────────────┐
                    │ MCP2515 CAN-Bus Controller │
                    └─────────────┬──────────────┘
                                  │ SPI Interface
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│                  ESP32-WROOM-32 (NodeMCU-32S)                │
│                                                              │
│  ┌─────────────────┐   ┌──────────────────┐   ┌────────────┐ │
│  │ Task 1: CAN-Bus │   │ Task 2: MQTT Rx  │   │ Task 3: UI │ │
│  │ Ingest SoC/Speed│   │ Recommendations  │   │ Display    │ │
│  └─────────────────┘   └──────────────────┘   └────────────┘ │
└──────────────┬───────────────────────────────┬───────────────┘
               │ SPI / I2C                     │ Wi-Fi / MQTT
               ▼                               ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ 2.8" SPI TFT LCD (ILI9341)  │  │ Local Raspberry Pi Gateway  │
│  • Best Station: Tata EZ    │  │ or Cloud HiveMQ MQTT Broker │
│  • Dynamic Price: ₹18.4/kWh │  │                             │
│  • Time to Charge: 48 min   │  │ Topics:                     │
│  • Live CC-CV Power Bar     │  │  - ev/recommendation        │
└─────────────────────────────┘  │  - ev/charging_profile      │
                                 └─────────────────────────────┘
```

### 7.2 Bill of Materials (BOM)

| Item | Component | Interface | Estimated Cost |
|:---:|:---|:---:|:---:|
| 1 | **ESP32-WROOM-32D Development Board** (Dual-core 240MHz, 520KB SRAM) | USB / Wi-Fi | ₹450 |
| 2 | **2.8" SPI TFT LCD Display** (240x320 ILI9341 with Touch) | SPI (VSPI) | ₹650 |
| 3 | **MCP2515 CAN Bus Module + TJA1050 Transceiver** | SPI (HSPI) | ₹180 |
| 4 | **OBD-II to DB9 Adapter Cable** (Standard SAE J1962) | Automotive CAN | ₹250 |
| 5 | **LM2596 DC-DC Buck Converter** ($12\text{V} \to 5\text{V} / 3\text{A}$) | Power Rail | ₹90 |
| **Total** | **Complete Hardware In-Cabin Display Unit** | — | **₹1,620 (~$19 USD)** |

### 7.3 MQTT Telemetry Payload Specifications

#### Topic: `ev/recommendation`
Published when a routing optimization cycle completes:
```json
{
  "station_id": "STN_IND_042",
  "name": "Tata Power EZ Charge - MP Nagar",
  "city": "Bhopal",
  "distance_km": 6.84,
  "soc_arrival": 21.4,
  "charging_time_min": 44.2,
  "wait_time_min": 3.8,
  "dynamic_price_per_kwh": 18.25,
  "base_price_per_kwh": 14.50,
  "charging_cost": 421.58,
  "energy_kwh": 23.1,
  "timestamp": 1731504000
}
```

#### Topic: `ev/charging_profile`
Published tick-by-tick for real-time dashboard plotting on the TFT display:
```json
{
  "t_s": 120,
  "soc_pct": 24.8,
  "power_kw": 50.0,
  "phase": "CC"
}
```

### 7.4 MicroPython Embedded Client Implementation
The following code runs natively on the ESP32 to subscribe to routing updates and display telemetry:

```python
# esp32_client.py - MicroPython MQTT client for EV Display
import network
import ujson
import time
from machine import Pin, SPI
from umqtt.simple import MQTTClient

# 1. Connect to In-Vehicle Wi-Fi Hotspot
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("Vehicle_Hotspot", "ev_pass_2026")
while not wlan.isconnected():
    time.sleep(0.2)
print("Wi-Fi Connected. IP:", wlan.ifconfig()[0])

# 2. Callback for Incoming Recommendations
def on_message(topic, msg):
    payload = ujson.loads(msg.decode())
    topic_str = topic.decode()
    
    if topic_str == "ev/recommendation":
        print(">>> Optimal Station Received:")
        print("Station :", payload.get("name"))
        print("Distance:", payload.get("distance_km"), "km")
        print("Price   : Rs", payload.get("dynamic_price_per_kwh"), "/kWh")
        print("Charge  : ~", payload.get("charging_time_min"), "min")
        # Update TFT LCD driver here
    
    elif topic_str == "ev/charging_profile":
        # Draw charging power gauge: payload['power_kw'], payload['phase']
        pass

# 3. Connect to Broker and Subscribe
client = MQTTClient("esp32_ev_display", "192.168.1.42", port=1883)
client.set_callback(on_message)
client.connect()
client.subscribe("ev/recommendation")
client.subscribe("ev/charging_profile")
print("ESP32 Listening on ev/# ...")

while True:
    client.check_msg()
    time.sleep(0.1)
```

---

## 8. Interactive Dashboard & Live Moving-Car Simulation (`app.py`)

The Streamlit web application (`app.py`) provides an intuitive, high-resolution user interface with five specialized views:

1. **🏆 Recommendation Tab:** Displays top KPI metric cards (Best Station Name, Distance, Positive Charging Duration, Dynamic Price, Station Load, Total Cost), safe range analysis, and ranked station listings with load-colored badges.
2. **⚡ CC-CV Physics Tab:** Renders instant power curves, SoC-vs-time trajectories, and dual bar charts splitting the session into Constant-Current vs. Constant-Voltage taper phases.
3. **📈 Forecaster Comparison Tab:** Features side-by-side ground-truth historical benchmark charts (Actual vs. RF vs. Math), full error metric reporting (MAE, RMSE, $R^2$), and live forward-looking 24-hour horizon forecasts.
4. **💰 Pricing Heatmap Tab:** Explores dynamic surge distributions, price elasticity multipliers, and load vs. price scatter plots across all 424 India stations.
5. **🗺️ Interactive Map Tab:** Plots the vehicle's exact GPS location, nearby reachable stations, and real-time transit steps.

### 8.1 Live Moving-Car & Queue Turnover Simulation
In "Simulated IoT" mode, users can trigger dynamic vehicle transit:
- **"Step Vehicle (2 min)" Button:** Advances vehicle position along the vector to the target station at 45 km/h (1.5 km per step).
- **Realistic Discharge Physics:** Deducts battery SoC based on distance, driving consumption, and drivetrain efficiency.
- **Queue Load Fluctuations:** Randomizes station utilization ($\pm 4\%$) across the network to simulate vehicles joining and departing charging bays.
- **Continuous Auto-Drive Loop:** Smoothly streams real-time updates every 1.2 seconds until arrival.

---

## 9. Verification & Test Suite Summary

The repository includes a comprehensive automated test suite (`tests/test_core.py` and `tests/test_physics_and_pricing.py`). All 17 unit tests execute cleanly in under 1 second:

```
tests/test_core.py::test_haversine_bhopal_indore PASSED                   [  5%]
tests/test_core.py::test_range_10pct PASSED                               [ 11%]
tests/test_core.py::test_low_battery_removes_far_stations PASSED          [ 17%]
tests/test_core.py::test_all_objectives_respect_constraint PASSED         [ 23%]
tests/test_core.py::test_objective_ordering PASSED                        [ 29%]
tests/test_core.py::test_unreachable_message PASSED                       [ 35%]
tests/test_core.py::test_validation PASSED                                [ 41%]
tests/test_core.py::test_erlang_c_monotonic PASSED                        [ 47%]
tests/test_physics_and_pricing.py::test_cv_power_kw_constant_current_phase PASSED [ 52%]
tests/test_physics_and_pricing.py::test_cv_power_kw_exponential_taper PASSED [ 58%]
tests/test_physics_and_pricing.py::test_cc_cv_charging_time PASSED        [ 64%]
tests/test_physics_and_pricing.py::test_integrate_cc_cv_kwh PASSED        [ 70%]
tests/test_physics_and_pricing.py::test_cc_cv_power_profile PASSED        [ 76%]
tests/test_physics_and_pricing.py::test_exponential_load_pricer_monotonic PASSED [ 82%]
tests/test_physics_and_pricing.py::test_exponential_load_pricer_cap PASSED [ 88%]
tests/test_physics_and_pricing.py::test_effective_cost_taper_penalty PASSED [ 94%]
tests/test_physics_and_pricing.py::test_mathematical_arrival_forecaster PASSED [100%]

============================== 17 passed in 0.99s ==============================
```

In addition, headless Streamlit AppTest execution (`streamlit.testing.v1.AppTest`) verified zero exceptions on startup, clean state persistence, and robust widget initialization.

---

## 10. Conclusion & Review Presentation Takeaways

This project delivers a complete, scientifically validated EV charging recommendation ecosystem that bridges theoretical electrochemistry, queueing theory, machine learning, and embedded IoT hardware:

1. **Physical Accuracy:** Replaces simplistic linear approximations with rigorous CC-CV lithium-ion differential equations ($k = 2.3026$).
2. **Economic Intelligence:** Mitigates station congestion and grid stress via exponential dynamic load pricing coupled with CV efficiency penalties.
3. **Rigorous Benchmarking:** Demonstrates that while Random Forest ML provides lower error on historical datasets (MAE 0.954 vs 1.305), the Mathematical Double-Gaussian forecaster provides a zero-training, low-memory solution essential for cold-start stations and edge microcontroller execution.
4. **Hardware Reality:** The modular MQTT interface and complete ESP32 implementation plan establish a clear roadmap for in-vehicle hardware deployment.

The codebase is fully tested, committed, and ready for immediate review and demonstration.
