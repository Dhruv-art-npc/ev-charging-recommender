"""
dynamic_pricing.py – Raw mathematical forecasting & dynamic pricing
====================================================================
Provides a **non-ML, formula-driven** alternative to the Random-Forest
arrival forecaster.  Two components live here:

1. ExponentialLoadPricer
   Applies the formula:
       P_dynamic = Base_Price * exp( Lambda * (Load / Capacity) )
   to compute a demand-adjusted price per kWh for each station.
   The term is additionally scaled by the battery's CC-CV charging
   efficiency curve so that the *effective* cost per kWh reflects the
   slower (less efficient at high SoC) part of the charge curve.

2. MathematicalArrivalForecaster
   A lightweight, interpretable forecaster that needs no training data:
   it reconstructs the arrival rate from two Gaussian peaks (morning
   commute + evening commute) modulated by a weekday/weekend coefficient
   and a seasonal amplitude – all from first-principles parameters.

Both classes expose a ``load_factor()`` method that is drop-in compatible
with the ML ``ArrivalForecaster.load_factor()`` already used by
``src/recommender.py``.

References
----------
* Price elasticity exponential model: Schweppe et al., "Spot Pricing of
  Electricity", Kluwer 1988.
* Double-Gaussian load profile: NREL EV Infrastructure Working Group, 2021.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ============================================================ constants =========
_LN2 = math.log(2)


# ============================================================ pricing ============

class ExponentialLoadPricer:
    """
    Dynamic electricity pricing via the exponential peak-load formula.

        P_dynamic(t) = Base_Price
                       * exp( lambda_load * (Load(t) / Capacity) )
                       * CV_efficiency_weight(SoC)

    Parameters
    ----------
    base_price_per_kwh : float
        Off-peak / base tariff in ₹/kWh (or any currency unit).
    lambda_load : float
        Steepness of load-based price surge.  Typical range 0.5–2.0.
        At lambda=1 and 100 % load, price ≈ 2.72 × base.
        At lambda=2 and 100 % load, price ≈ 7.39 × base.
    max_price_cap : float | None
        Hard ceiling on the computed dynamic price (same units as base).
        ``None`` → no cap.
    """

    def __init__(self,
                 base_price_per_kwh: float = 15.0,
                 lambda_load: float = 1.0,
                 max_price_cap: float | None = None):
        self.base_price_per_kwh = base_price_per_kwh
        self.lambda_load = lambda_load
        self.max_price_cap = max_price_cap

    # ---- core formula ----------------------------------------------------------
    def price_at_load(self,
                      current_load_pct: float,
                      station_capacity_pct: float = 100.0) -> float:
        """
        Compute dynamic price (₹/kWh) based on station utilisation.

        Parameters
        ----------
        current_load_pct    : observed load in % (0–100).
        station_capacity_pct: total capacity normalisation (default 100).

        Returns
        -------
        Dynamic price per kWh.
        """
        load_ratio = max(current_load_pct, 0.0) / max(station_capacity_pct, 1.0)
        p = self.base_price_per_kwh * math.exp(self.lambda_load * load_ratio)
        if self.max_price_cap is not None:
            p = min(p, self.max_price_cap)
        return round(p, 4)

    # ---- CC-CV aware effective cost -------------------------------------------
    def effective_cost_per_kwh(self,
                                current_load_pct: float,
                                soc_start: float,
                                soc_end: float,
                                soc_taper_pct: float = 80.0,
                                k_taper: float = 2.302) -> float:
        """
        Effective cost per kWh accounting for the CC-CV efficiency curve.

        When charging above the taper SoC the charger draws *more* grid
        energy per kWh delivered to the battery (CV phase is less wall-plug
        efficient per unit time).  We weight the base price by the average
        inverse power-fraction over the [soc_start, soc_end] window.

        Parameters
        ----------
        current_load_pct : station utilisation % for the demand charge.
        soc_start, soc_end : charging window in % SoC.
        soc_taper_pct    : CC→CV transition point (default 80 %).
        k_taper          : CV exponential decay constant (default ln10).

        Returns
        -------
        Effective ₹/kWh (always ≥ dynamic price at given load).
        """
        base_dynamic = self.price_at_load(current_load_pct)
        if soc_end <= soc_taper_pct:
            # Entirely in CC phase – no extra cost penalty
            return base_dynamic

        # Numerical average of (1 / power_fraction) over soc_start..soc_end
        # = weighted extra grid energy factor
        pts = np.linspace(soc_start, soc_end, 200)
        fracs = np.array([
            self._power_frac(s, soc_taper_pct, k_taper) for s in pts
        ])
        avg_inv_frac = float(np.mean(1.0 / np.maximum(fracs, 0.01)))
        return round(base_dynamic * avg_inv_frac, 4)

    @staticmethod
    def _power_frac(soc: float, taper: float, k: float) -> float:
        if soc <= taper:
            return 1.0
        x = (soc - taper) / max(100.0 - taper, 1e-9)
        return math.exp(-k * x)

    # ---- DataFrame vectorised helper ------------------------------------------
    def apply_to_stations(self,
                           df: pd.DataFrame,
                           soc_start: float = 20.0,
                           soc_end: float = 80.0) -> pd.DataFrame:
        """
        Add ``dynamic_price_per_kwh`` and ``effective_price_per_kwh`` columns
        to a stations DataFrame that has ``current_load_pct`` and
        ``price_per_kwh``.

        Uses each station's own ``price_per_kwh`` as the base tariff.
        """
        df = df.copy()
        df["dynamic_price_per_kwh"] = df.apply(
            lambda r: ExponentialLoadPricer(
                base_price_per_kwh=float(r.get("price_per_kwh", self.base_price_per_kwh)),
                lambda_load=self.lambda_load,
                max_price_cap=self.max_price_cap,
            ).price_at_load(float(r.get("current_load_pct", 50.0))),
            axis=1,
        )
        df["effective_price_per_kwh"] = df.apply(
            lambda r: ExponentialLoadPricer(
                base_price_per_kwh=float(r.get("price_per_kwh", self.base_price_per_kwh)),
                lambda_load=self.lambda_load,
                max_price_cap=self.max_price_cap,
            ).effective_cost_per_kwh(
                current_load_pct=float(r.get("current_load_pct", 50.0)),
                soc_start=soc_start,
                soc_end=soc_end,
            ),
            axis=1,
        )
        return df


# ============================================================ math forecaster ===

class MathematicalArrivalForecaster:
    """
    Closed-form, training-free EV arrival forecaster.

    Models arrivals as the sum of two Gaussian peaks (morning and evening
    commutes) on a 24-hour clock, scaled by a weekday/weekend factor and
    a month-of-year seasonal envelope.

    Load factor
    -----------
    ``load_factor(eta)`` returns the ratio of predicted arrivals at ``eta``
    to the long-run hourly mean, clipped to [0.5, 2.0] – the same interface
    as ``ArrivalForecaster.load_factor()``.

    Parameters
    ----------
    peak_am_h    : hour of morning arrival peak   (default 8.0)
    peak_pm_h    : hour of evening arrival peak   (default 18.0)
    sigma_am_h   : standard deviation of AM Gaussian (hours, default 1.5)
    sigma_pm_h   : standard deviation of PM Gaussian (hours, default 2.0)
    amp_am       : amplitude of AM peak (arrivals/hour at peak)
    amp_pm       : amplitude of PM peak
    weekend_scale: multiplicative factor for Sat/Sun (< 1 → quieter)
    seasonal_amp : fractional seasonal variation (0 = none, 0.2 = ±20 %)
    """

    def __init__(self,
                 peak_am_h: float = 8.0,
                 peak_pm_h: float = 18.0,
                 sigma_am_h: float = 1.5,
                 sigma_pm_h: float = 2.0,
                 amp_am: float = 4.5,
                 amp_pm: float = 6.0,
                 weekend_scale: float = 0.65,
                 seasonal_amp: float = 0.15):
        self.peak_am_h = peak_am_h
        self.peak_pm_h = peak_pm_h
        self.sigma_am_h = sigma_am_h
        self.sigma_pm_h = sigma_pm_h
        self.amp_am = amp_am
        self.amp_pm = amp_pm
        self.weekend_scale = weekend_scale
        self.seasonal_amp = seasonal_amp
        # pre-compute long-run mean for normalisation
        self._hourly_mean = self._compute_weekday_mean()

    # ---- core formula ----------------------------------------------------------
    def _gaussian(self, h: float, mu: float, sigma: float) -> float:
        """Standard unnormalised Gaussian on a periodic 24-h clock."""
        # wrap distance to handle 23 → 0 wrap
        d = ((h - mu + 12) % 24) - 12
        return math.exp(-0.5 * (d / sigma) ** 2)

    def _seasonal_factor(self, month: int) -> float:
        """
        Cosine seasonal modulation.
        Peak in December (month 12) in India's EV season; troughs in June.
        """
        # Map month 1–12 to angle; peak at month 12 (winter)
        angle = 2 * math.pi * (month - 12) / 12.0
        return 1.0 + self.seasonal_amp * math.cos(angle)

    def arrivals(self, ts: pd.Timestamp) -> float:
        """Predicted arrivals/hour at a given timestamp."""
        h = ts.hour + ts.minute / 60.0
        is_weekend = ts.dayofweek >= 5
        w = self.weekend_scale if is_weekend else 1.0
        s = self._seasonal_factor(ts.month)
        base = (self.amp_am * self._gaussian(h, self.peak_am_h, self.sigma_am_h)
                + self.amp_pm * self._gaussian(h, self.peak_pm_h, self.sigma_pm_h))
        return max(base * w * s, 0.0)

    def _compute_weekday_mean(self) -> float:
        """Average arrivals/hour over a representative 24-h weekday."""
        ref = pd.Timestamp("2024-01-15 00:00")  # Monday mid-January
        return float(np.mean([self.arrivals(ref + pd.Timedelta(hours=h)) for h in range(24)]))

    def forecast(self, start: pd.Timestamp, hours: int = 24) -> pd.Series:
        """Return a Series of predicted arrivals for ``hours`` hours from ``start``."""
        idx = pd.date_range(start, periods=hours, freq="h")
        return pd.Series([self.arrivals(t) for t in idx], index=idx, name="math_arrivals")

    def load_factor(self, eta: pd.Timestamp) -> float:
        """
        Drop-in replacement for ``ArrivalForecaster.load_factor()``.
        Returns ratio of predicted arrivals at ``eta`` to the long-run mean,
        clipped to [0.5, 2.0].
        """
        p = self.arrivals(pd.Timestamp(eta))
        base = self._hourly_mean or 1.0
        return float(np.clip(p / base, 0.5, 2.0))
