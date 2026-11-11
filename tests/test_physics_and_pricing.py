"""Tests for CC-CV charging physics, exponential dynamic pricing, and mathematical forecasting."""
import pytest
import pandas as pd
from src.charging_physics import (
    cc_cv_charging_time_min,
    integrate_cc_cv_kwh,
    cv_power_kw,
    cc_cv_power_profile,
    SOC_TAPER_DEFAULT,
)
from src.dynamic_pricing import (
    ExponentialLoadPricer,
    MathematicalArrivalForecaster,
)


def test_cv_power_kw_constant_current_phase():
    """Below taper SoC (default 80%), power should remain at rated P_cc."""
    assert cv_power_kw(soc_pct=20.0, p_cc_kw=50.0) == 50.0
    assert cv_power_kw(soc_pct=50.0, p_cc_kw=50.0) == 50.0
    assert cv_power_kw(soc_pct=80.0, p_cc_kw=50.0) == 50.0


def test_cv_power_kw_exponential_taper():
    """Above 80% SoC, power should drop exponentially."""
    p_80 = cv_power_kw(soc_pct=80.0, p_cc_kw=50.0)
    p_90 = cv_power_kw(soc_pct=90.0, p_cc_kw=50.0)
    p_95 = cv_power_kw(soc_pct=95.0, p_cc_kw=50.0)
    p_100 = cv_power_kw(soc_pct=100.0, p_cc_kw=50.0)

    assert p_80 > p_90 > p_95 > p_100
    # At 100% SoC with k=ln(10) ≈ 2.302, power decays to ~10% of rated
    assert p_100 == pytest.approx(50.0 * 0.10, rel=1e-2)


def test_cc_cv_charging_time():
    """Charging duration should be positive and increase monotonically with target SoC."""
    t_cc = cc_cv_charging_time_min(soc_start=20, soc_end=80, battery_kwh=40, power_kw=50)
    t_full = cc_cv_charging_time_min(soc_start=20, soc_end=95, battery_kwh=40, power_kw=50)

    assert t_cc > 0
    assert t_full > t_cc
    # Zero duration if start >= end
    assert cc_cv_charging_time_min(soc_start=80, soc_end=50, battery_kwh=40, power_kw=50) == 0.0


def test_integrate_cc_cv_kwh():
    """Grid energy accounts for charger efficiency (default 0.92)."""
    # 20% to 80% of 40 kWh = 24 kWh delivered to battery
    # Grid draw = 24 / 0.92 ≈ 26.087 kWh
    kwh = integrate_cc_cv_kwh(soc_start=20, soc_end=80, battery_kwh=40, power_kw=50, charger_eff=0.92)
    assert kwh == pytest.approx(24.0 / 0.92, rel=1e-3)


def test_cc_cv_power_profile():
    """Power profile generates sequential time steps with CC and CV phase labels."""
    profile = cc_cv_power_profile(soc_start=20, soc_end=90, battery_kwh=40, power_kw=50, dt_s=60.0)
    assert len(profile) > 1
    assert profile[0]["soc_pct"] == 20.0
    assert profile[0]["phase"] == "CC"
    assert profile[-1]["soc_pct"] == pytest.approx(90.0, abs=0.1)
    assert profile[-1]["phase"] == "CV"


def test_exponential_load_pricer_monotonic():
    """Dynamic price must increase exponentially with station load."""
    pricer = ExponentialLoadPricer(base_price_per_kwh=15.0, lambda_load=1.0)
    p_0 = pricer.price_at_load(0.0)
    p_50 = pricer.price_at_load(50.0)
    p_90 = pricer.price_at_load(90.0)

    assert p_0 == pytest.approx(15.0)
    assert p_0 < p_50 < p_90


def test_exponential_load_pricer_cap():
    """Dynamic price must respect maximum price ceiling."""
    pricer = ExponentialLoadPricer(base_price_per_kwh=15.0, lambda_load=3.0, max_price_cap=40.0)
    p_high = pricer.price_at_load(95.0)
    assert p_high <= 40.0


def test_effective_cost_taper_penalty():
    """Sessions entering the CV phase should carry higher effective cost per kWh."""
    pricer = ExponentialLoadPricer(base_price_per_kwh=15.0, lambda_load=1.0)
    cost_cc = pricer.effective_cost_per_kwh(current_load_pct=50, soc_start=20, soc_end=80)
    cost_cv = pricer.effective_cost_per_kwh(current_load_pct=50, soc_start=20, soc_end=95)
    assert cost_cv > cost_cc


def test_mathematical_arrival_forecaster():
    """Mathematical forecaster generates 24h series and bounded load factors."""
    forecaster = MathematicalArrivalForecaster()
    now = pd.Timestamp("2026-11-12 12:00:00")
    fc = forecaster.forecast(start=now, hours=24)
    assert len(fc) == 24
    assert (fc >= 0).all()

    # Load factor should be clipped to [0.5, 2.0]
    lf_peak = forecaster.load_factor(pd.Timestamp("2026-11-12 18:00:00"))
    lf_night = forecaster.load_factor(pd.Timestamp("2026-11-12 03:00:00"))
    assert 0.5 <= lf_night <= 2.0
    assert 0.5 <= lf_peak <= 2.0
    assert lf_peak > lf_night
