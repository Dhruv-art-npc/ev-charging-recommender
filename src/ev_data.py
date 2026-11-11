"""Module 1 - EV data input (manual / JSON file / IoT-MQTT / simulated IoT)."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, asdict


@dataclass
class EVState:
    lat: float
    lon: float
    battery_kwh: float = 40.0          # usable battery capacity
    soc: float = 50.0                  # current state of charge (%)
    target_soc: float = 80.0           # desired SOC after charging (%)
    consumption_kwh_per_km: float = 0.15
    efficiency: float = 0.90           # drivetrain efficiency (0-1)
    max_dc_kw: float = 50.0            # max DC charging power the car accepts
    connector: str = "CCS2"

    def __post_init__(self):
        if not (0 <= self.soc <= 100):
            raise ValueError("soc must be between 0 and 100")
        if not (0 < self.target_soc <= 100):
            raise ValueError("target_soc must be in (0, 100]")
        if self.target_soc <= self.soc:
            raise ValueError("target_soc must be greater than current soc")
        if not (0 < self.efficiency <= 1):
            raise ValueError("efficiency must be in (0, 1]")
        if self.battery_kwh <= 0 or self.consumption_kwh_per_km <= 0:
            raise ValueError("battery_kwh and consumption must be positive")

    # ---- derived ----
    @property
    def available_kwh(self) -> float:
        return self.battery_kwh * self.soc / 100.0

    def max_range_km(self, reserve_soc: float = 0.0) -> float:
        """Distance the car can drive while keeping `reserve_soc` % in the battery."""
        usable = max(self.soc - reserve_soc, 0.0) / 100.0 * self.battery_kwh
        return usable * self.efficiency / self.consumption_kwh_per_km

    def to_dict(self) -> dict:
        return asdict(self)


def from_manual(**kwargs) -> EVState:
    """Manual input (CLI / web form)."""
    return EVState(**kwargs)


def from_json(path_or_str: str) -> EVState:
    """IoT gateway dumps a JSON message/file, e.g. {"lat":..,"lon":..,"soc":..}."""
    try:
        payload = json.loads(path_or_str)
    except (json.JSONDecodeError, TypeError):
        with open(path_or_str, "r", encoding="utf-8") as f:
            payload = json.load(f)
    alias = {"latitude": "lat", "longitude": "lon", "battery_capacity_kwh": "battery_kwh",
             "current_soc": "soc", "consumption": "consumption_kwh_per_km"}
    payload = {alias.get(k, k): v for k, v in payload.items()}
    allowed = EVState.__dataclass_fields__.keys()
    return EVState(**{k: v for k, v in payload.items() if k in allowed})


def simulate_iot_reading(lat=23.2599, lon=77.4126, seed=None) -> EVState:
    """Fake IoT telemetry for demos/tests."""
    rng = random.Random(seed)
    soc = rng.uniform(8, 60)
    return EVState(lat=lat + rng.uniform(-0.05, 0.05), lon=lon + rng.uniform(-0.05, 0.05),
                   battery_kwh=rng.choice([30.2, 40.0, 50.3, 64.0]), soc=round(soc, 1),
                   target_soc=80.0, consumption_kwh_per_km=round(rng.uniform(0.12, 0.18), 3))


def listen_mqtt(host: str, topic: str, on_state, port: int = 1883):  # pragma: no cover
    """Real IoT: subscribe to an MQTT topic whose payload is the JSON above.
    Needs `pip install paho-mqtt`. `on_state(EVState)` is called for every message."""
    import paho.mqtt.client as mqtt
    client = mqtt.Client()
    client.on_message = lambda c, u, m: on_state(from_json(m.payload.decode()))
    client.connect(host, port)
    client.subscribe(topic)
    client.loop_forever()
