"""Module 5 - AI forecasting with Random Forest on the EPFL Level-3 charging data.

What we predict from the EPFL dataset
  A) arrivals_per_hour  (MAIN)   -> how many EVs will arrive in each future hour
                                    -> converted to a load factor used for waiting-time estimate
  B) session_duration_min        -> how long a car will occupy a charger
  C) session_energy_kwh          -> energy it will draw (revenue / grid load)
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

ALIASES = {
    "start_time": ["start_time", "start", "session_start", "starttime", "start_datetime", "timestamp"],
    "end_time": ["end_time", "end", "session_end", "endtime", "end_datetime"],
    "duration_min": ["duration_min", "duration", "charging_time_min"],
    "energy_kwh": ["energy_kwh", "energy", "delivered_energy_kwh", "kwh"],
    "max_power_kw": ["max_power_kw", "peak_power_kw", "max_power", "peak_power"],
    "soc_start": ["soc_start", "initial_soc", "start_soc"],
    "soc_end": ["soc_end", "final_soc", "end_soc"],
    "battery_kwh": ["battery_kwh", "battery_capacity_kwh", "capacity_kwh"],
}


def load_sessions(path=config.EPFL_SESSIONS_CSV) -> pd.DataFrame:
    """Load EPFL L3 sessions (one row per charging session; raw time-series should be
    aggregated to sessions first). Column names are matched via ALIASES."""
    raw = pd.read_csv(path)
    low = {c.lower().strip(): c for c in raw.columns}
    df = pd.DataFrame()
    for std, names in ALIASES.items():
        for n in names:
            if n in low:
                df[std] = raw[low[n]]
                break
    if "start_time" not in df:
        raise ValueError(f"No start-time column found. Columns: {list(raw.columns)}")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
        if "duration_min" not in df:
            df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    df = df.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    return df


def hourly_arrivals(sessions: pd.DataFrame) -> pd.Series:
    s = sessions.set_index("start_time").resample("h").size().astype(float)
    s.name = "arrivals"
    return s


# ---------------------------------------------------------------- features
def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    return f


def _lag_row(history: list[float]) -> dict:
    h = np.asarray(history)
    return {"lag_1": h[-1], "lag_24": h[-24], "lag_168": h[-168],
            "roll_24": h[-24:].mean(), "roll_168": h[-168:].mean()}


def make_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    f = _calendar(series.index)
    f["lag_1"] = series.shift(1)
    f["lag_24"] = series.shift(24)
    f["lag_168"] = series.shift(168)
    f["roll_24"] = series.shift(1).rolling(24).mean()
    f["roll_168"] = series.shift(1).rolling(168).mean()
    f = f.dropna()
    return f, series.loc[f.index]


def _metrics(y, p) -> dict:
    return {"MAE": round(float(mean_absolute_error(y, p)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "R2": round(float(r2_score(y, p)), 4)}


class ArrivalForecaster:
    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_leaf=2, n_jobs=-1, random_state=random_state)
        self.history: pd.Series | None = None
        self.hour_mean: pd.Series | None = None
        self.metrics: dict = {}

    # ---------------- training + honest evaluation
    def fit(self, series: pd.Series, test_frac: float = 0.2) -> dict:
        X, y = make_supervised(series)
        cut = int(len(X) * (1 - test_frac))                    # chronological split, no shuffling
        Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
        self.model.fit(Xtr, ytr)
        pred = self.model.predict(Xte)
        self.metrics = {
            "test_rows": int(len(yte)),
            "random_forest": _metrics(yte, pred),
            "baseline_same_hour_yesterday": _metrics(yte, Xte["lag_24"]),
            "baseline_same_hour_last_week": _metrics(yte, Xte["lag_168"]),
            "baseline_train_mean": _metrics(yte, np.full(len(yte), ytr.mean())),
            "top_features": dict(sorted(zip(X.columns, map(lambda v: round(float(v), 4),
                                                            self.model.feature_importances_)),
                                        key=lambda kv: -kv[1])[:5]),
        }
        self.model.fit(X, y)                                   # refit on everything for deployment
        self.history = series.copy()
        self.hour_mean = series.groupby(series.index.hour).mean()
        return self.metrics

    # ---------------- recursive multi-step forecast
    def forecast(self, start: pd.Timestamp | None = None, hours: int = 24) -> pd.Series:
        """Recursive multi-step forecast. If `start` is right after the training data we
        continue from the real history; if it is later (e.g. today's ETA) we seed the lag
        window with the typical hour-of-week profile, so it stays fast and sensible."""
        if self.history is None:
            raise RuntimeError("fit() or load() first")
        last = self.history.index[-1]
        idx0 = last + pd.Timedelta(hours=1) if start is None else pd.Timestamp(start).floor("h")
        if idx0 <= last:
            raise ValueError("start must be after the last training hour")
        if idx0 == last + pd.Timedelta(hours=1):
            hist = list(self.history.values[-168:])
        else:
            recent = self.history.iloc[-24 * 7 * 8:]
            prof = recent.groupby([recent.index.dayofweek, recent.index.hour]).mean()
            seed_idx = pd.date_range(idx0 - pd.Timedelta(hours=168), periods=168, freq="h")
            hist = [float(prof.get((t.dayofweek, t.hour), recent.mean())) for t in seed_idx]
        idx = pd.date_range(idx0, periods=hours, freq="h")
        preds = []
        for ts in idx:
            row = {**_calendar(pd.DatetimeIndex([ts])).iloc[0].to_dict(), **_lag_row(hist)}
            X = pd.DataFrame([row])[list(self.model.feature_names_in_)]
            p = float(max(self.model.predict(X)[0], 0.0))
            hist.append(p); preds.append(p)
        return pd.Series(preds, index=idx, name="predicted_arrivals")

    def load_factor(self, eta: pd.Timestamp) -> float:
        """>1 means busier than the typical same-hour demand. Used to scale station load."""
        if self.history is None:
            return 1.0
        eta = pd.Timestamp(eta).floor("h")
        p = float(self.forecast(start=eta, hours=1).iloc[0])
        base = float(self.hour_mean.get(eta.hour, self.history.mean())) or 1.0
        return float(np.clip(p / base, 0.5, 2.0))

    # ---------------- persistence
    def save(self, path=config.MODEL_DIR / "arrival_rf.joblib"):
        config.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": self.model, "history": self.history, "hour_mean": self.hour_mean,
                     "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path=config.MODEL_DIR / "arrival_rf.joblib") -> "ArrivalForecaster":
        d = joblib.load(path)
        f = cls(); f.model, f.history, f.hour_mean, f.metrics = d["model"], d["history"], d["hour_mean"], d["metrics"]
        return f


# ---------------------------------------------------------------- session models (B, C)
def train_session_models(sessions: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """Predict session duration and energy from when/who arrives."""
    df = sessions.copy()
    cal = _calendar(pd.DatetimeIndex(df["start_time"]))
    X = cal.reset_index(drop=True)
    for c in ("soc_start", "battery_kwh", "max_power_kw"):
        if c in df:
            X[c] = df[c].values
    X = X.fillna(X.median(numeric_only=True))
    cut = int(len(X) * (1 - test_frac))
    res = {}
    for target in ("duration_min", "energy_kwh"):
        if target not in df:
            continue
        y = df[target].values
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42)
        m.fit(X.iloc[:cut], y[:cut])
        res[target] = {**_metrics(y[cut:], m.predict(X.iloc[cut:])),
                       "baseline_mean_MAE": round(float(mean_absolute_error(y[cut:], np.full(len(y) - cut, y[:cut].mean()))), 4)}
        m.fit(X, y)
        joblib.dump({"model": m, "columns": list(X.columns)}, config.MODEL_DIR / f"session_{target}.joblib")
    return res
