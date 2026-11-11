"""
Reference forecasting models for the notebook's model comparison (offline; never deployed).

They use exactly the same inputs as the browser models — the unencrypted data pack
(data/processed/favorita-pack.json), the same features (app/src/data/features.ts), the same
closed-day imputation (app/src/models/history.ts), the same 6 × 28-day rolling origins and the
same metric definitions (app/src/models/metrics.ts) — so accuracy is directly comparable.

Models
  * LightGBM      gradient-boosted trees on the Ridge feature set (exogenous + lag 7/14, 7/28-day means), recursive
  * SARIMAX       (1,1,1)×(1,0,1)₇ on log sales with promotion/holiday/payday/Christmas/yearly regressors
  * Keras LSTM    the same architecture and training scheme as the TF.js LSTM (retrained per fold)
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAIN_PACK = ROOT / "data" / "processed" / "favorita-pack.json"
HORIZON, FOLDS, MIN_TRAIN_DAYS = 28, 6, 400
YEAR = 365.25
EXOG = ["trend", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
        "sin1", "cos1", "sin2", "cos2", "sin3", "cos3",
        "log_promo", "holiday", "event", "quake", "work_day", "payday", "xmas", "oil_lag7"]


@dataclass
class Series:
    id: str
    dates: pd.DatetimeIndex
    sales: np.ndarray          # NaN = unobserved
    promo: np.ndarray
    closed: np.ndarray
    flags: pd.DataFrame        # holiday, event, quake, workDay
    oil: np.ndarray
    model_start: int
    replay_start: int
    fractional: bool


def load_pack(path: Path = PLAIN_PACK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series_from_pack(pack: dict, sid: str) -> Series:
    ps = next(s for s in pack["series"] if s["id"] == sid)
    st = next(s for s in pack["stores"] if s["store"] == ps["store"])
    cal = pack["calendar"]
    dates = pd.date_range(cal["start"], periods=cal["days"], freq="D")
    at = lambda iso: int((pd.Timestamp(iso) - dates[0]).days)
    return Series(
        id=sid, dates=dates,
        sales=np.array([np.nan if v is None else v for v in ps["sales"]], float),
        promo=np.array(ps["promo"], float), closed=np.array(st["closed"], int),
        flags=pd.DataFrame({k: st[k] for k in ("holiday", "event", "quake", "workDay")}, index=dates),
        oil=np.array(cal["oil"], float), model_start=at(cal["modelStart"]), replay_start=at(cal["replayStart"]),
        fractional=ps["fractional"])


def exog_matrix(s: Series) -> np.ndarray:
    """Same definitions as app/src/data/features.ts (DEFAULT_FEATURES)."""
    d = s.dates
    doy = d.dayofyear.to_numpy()
    cols = {"trend": (np.arange(len(d)) - s.model_start) / YEAR}
    for k, name in enumerate(["tue", "wed", "thu", "fri", "sat", "sun"], start=1):
        cols[f"dow_{name}"] = (d.dayofweek == k).astype(float)
    for k in (1, 2, 3):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * doy / YEAR)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * doy / YEAR)
    cols["log_promo"] = np.log1p(s.promo)
    cols["holiday"] = s.flags["holiday"].to_numpy(float)
    cols["event"] = s.flags["event"].to_numpy(float)
    cols["quake"] = s.flags["quake"].to_numpy(float)
    cols["work_day"] = s.flags["workDay"].to_numpy(float)
    cols["payday"] = ((d.day == 15) | d.is_month_end).astype(float)
    cols["xmas"] = np.where((d.month == 12) & (d.day <= 24), d.day / 24.0, 0.0)
    oil_lag = np.empty(len(d))
    for i in range(len(d)):
        w = s.oil[max(0, i - 7):i]
        oil_lag[i] = w.mean() if len(w) else s.oil[i]
    cols["oil_lag7"] = oil_lag
    return np.column_stack([cols[c] for c in EXOG])


def history(s: Series, end: int) -> tuple[np.ndarray, np.ndarray]:
    """log1p sales with closed/unobserved days imputed by y[i-7]; open = usable target (history.ts)."""
    y = np.zeros(end + 1)
    op = np.zeros(end + 1, bool)
    last = 0.0
    for i in range(end + 1):
        v = s.sales[i]
        if s.closed[i] == 0 and np.isfinite(v):
            y[i] = np.log1p(v); op[i] = True; last = y[i]
        else:
            y[i] = y[i - 7] if i >= 7 else last
    return y, op


def ar_row(y: np.ndarray, t: int) -> list[float]:
    return [y[t - 7], y[t - 14], y[t - 7:t].mean(), y[t - 28:t].mean()]


def fold_origins(s: Series, as_of: int) -> list[int]:
    return [as_of - k * HORIZON for k in range(FOLDS, 0, -1) if as_of - k * HORIZON - s.model_start >= MIN_TRAIN_DAYS]


def actuals(s: Series, origin: int) -> np.ndarray:
    idx = np.arange(origin + 1, origin + 1 + HORIZON)
    a = s.sales[idx].copy()
    a[s.closed[idx] == 1] = np.nan
    return a


def naive_scale(s: Series, to: int) -> float:
    ok = lambda i: s.closed[i] == 0 and np.isfinite(s.sales[i])
    diffs = [abs(s.sales[t] - s.sales[t - 7]) for t in range(max(s.model_start, 7), to + 1) if ok(t) and ok(t - 7)]
    return float(np.mean(diffs))


def metrics(a: np.ndarray, f: np.ndarray, scale: float) -> dict:
    m = np.isfinite(a) & np.isfinite(f)
    a, f = a[m], f[m]
    e = f - a
    return {"n": int(m.sum()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "wape": float(np.abs(e).sum() / a.sum()), "bias": float(e.sum() / a.sum()),
            "mase": float(np.abs(e).mean() / scale)}


def _to_units(log_pred: float, smear: float) -> float:
    return max(0.0, np.exp(log_pred) * smear - 1)


# ---------------------------------------------------------------------------------------------
# LightGBM (recursive, same features as Ridge)
# ---------------------------------------------------------------------------------------------
def lgbm_fold(s: Series, X_exog: np.ndarray, origin: int, seed: int = 7) -> np.ndarray:
    import lightgbm as lgb
    y, op = history(s, origin)
    targets = [t for t in range(max(s.model_start, 28), origin + 1) if op[t]]
    X = np.array([np.r_[X_exog[t], ar_row(y, t)] for t in targets])
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(X, y[targets])
    smear = float(np.mean(np.exp(y[targets] - model.predict(X))))
    y_ext = np.r_[y, np.zeros(HORIZON)]
    out = np.zeros(HORIZON)
    for k in range(1, HORIZON + 1):
        t = origin + k
        pred = float(model.predict(np.r_[X_exog[t], ar_row(y_ext, t)][None, :])[0])
        if s.closed[t]:
            out[k - 1], y_ext[t] = 0.0, y_ext[t - 7]
        else:
            out[k - 1], y_ext[t] = _to_units(pred, smear), pred
    return out


# ---------------------------------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------------------------------
SARIMAX_EXOG = ["log_promo", "holiday", "event", "payday", "xmas", "sin1", "cos1"]


def sarimax_fold(s: Series, X_exog: np.ndarray, origin: int, window: int = 730) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    cols = [EXOG.index(c) for c in SARIMAX_EXOG]
    lo = max(s.model_start, origin - window + 1)
    y = np.log1p(s.sales[lo:origin + 1].copy())
    y[s.closed[lo:origin + 1] == 1] = np.nan                    # Kalman filter skips missing days
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, exog=X_exog[lo:origin + 1, cols], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    smear = float(np.nanmean(np.exp(res.resid[30:])))
    f = res.forecast(HORIZON, exog=X_exog[origin + 1:origin + 1 + HORIZON, cols])
    idx = np.arange(origin + 1, origin + 1 + HORIZON)
    return np.where(s.closed[idx] == 1, 0.0, np.maximum(0, np.exp(f) * smear - 1))


# ---------------------------------------------------------------------------------------------
# Keras LSTM — mirrors app/src/models/lstm.ts
# ---------------------------------------------------------------------------------------------
W = 28


def _level28(y, t):
    return y[t - 28:t].mean()


def _seq(s: Series, y, open_flag, t, scale):
    lvl = _level28(y, t)
    rows = []
    for j in range(t - W, t):
        dow = s.dates[j].dayofweek
        rows.append([(y[j] - lvl) / scale, np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), open_flag(j)])
    return rows


def _exo(s: Series, t, promo_scale):
    d = s.dates[t]
    return [np.sin(2 * np.pi * d.dayofweek / 7), np.cos(2 * np.pi * d.dayofweek / 7),
            np.sin(2 * np.pi * d.dayofyear / YEAR), np.cos(2 * np.pi * d.dayofyear / YEAR),
            np.log1p(s.promo[t]) / promo_scale, s.flags["holiday"].iat[t], s.flags["event"].iat[t],
            s.flags["quake"].iat[t], float(d.day == 15 or d.is_month_end),
            d.day / 24.0 if (d.month == 12 and d.day <= 24) else 0.0]


def keras_lstm_fold(s: Series, origin: int, seed: int = 7) -> np.ndarray:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    import tensorflow as tf
    from keras import layers
    tf.get_logger().setLevel("ERROR")   # silence TF1-compat deprecation notices from clear_session()
    keras.utils.set_random_seed(seed)
    y, op = history(s, origin)
    targets = [t for t in range(max(s.model_start, W + 28), origin + 1) if op[t]]
    dev = np.array([y[t] - _level28(y, t) for t in targets])
    scale = max(1e-3, dev.std())
    promo_scale = max(1.0, max(np.log1p(s.promo[t]) for t in targets))
    seq = np.array([_seq(s, y, lambda j: float(op[j]), t, scale) for t in targets], np.float32)
    exo = np.array([_exo(s, t, promo_scale) for t in targets], np.float32)
    tgt = (dev / scale).astype(np.float32)
    n_val = max(1, int(len(targets) * 0.15))
    perm = np.random.default_rng(seed).permutation(len(targets) - n_val)

    seq_in, exo_in = keras.Input((W, 4)), keras.Input((10,))
    h = layers.LSTM(32, kernel_initializer=keras.initializers.GlorotUniform(seed),
                    recurrent_initializer=keras.initializers.GlorotUniform(seed + 1))(seq_in)
    h = layers.Dense(16, activation="relu")(layers.Concatenate()([h, exo_in]))
    net = keras.Model([seq_in, exo_in], layers.Dense(1)(h))
    net.compile(optimizer=keras.optimizers.Adam(0.005), loss="mse")
    net.fit([seq[perm], exo[perm]], tgt[perm], validation_data=([seq[-n_val:], exo[-n_val:]], tgt[-n_val:]),
            epochs=40, batch_size=32, shuffle=False, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)])
    resid = (tgt - net.predict([seq, exo], verbose=0)[:, 0]) * scale
    smear = float(np.mean(np.exp(resid)))

    y_ext = np.r_[y, np.zeros(HORIZON)]
    open_flag = lambda j: float(op[j]) if j <= origin else float(s.closed[j] == 0)
    out = np.zeros(HORIZON)
    for k in range(1, HORIZON + 1):
        t = origin + k
        if s.closed[t]:
            out[k - 1], y_ext[t] = 0.0, y_ext[t - 7]
            continue
        x_seq = np.array([_seq(s, y_ext, open_flag, t, scale)], np.float32)
        x_exo = np.array([_exo(s, t, promo_scale)], np.float32)
        log_pred = _level28(y_ext, t) + float(net([x_seq, x_exo], training=False).numpy()[0, 0]) * scale
        y_ext[t] = log_pred
        out[k - 1] = _to_units(log_pred, smear)
    keras.backend.clear_session()
    return out


MODELS = {"lightgbm": "LightGBM (trees)", "sarimax": "SARIMAX", "keras_lstm": "Keras LSTM"}


def evaluate(pack: dict, sid: str, models: list[str]) -> dict:
    """Rolling-origin evaluation identical to the browser engine (as of the replay start)."""
    s = series_from_pack(pack, sid)
    as_of = s.replay_start - 1
    origins = fold_origins(s, as_of)
    X = exog_matrix(s)
    scale = naive_scale(s, origins[0])
    a = np.concatenate([actuals(s, o) for o in origins])
    out = {}
    for m in models:
        if m == "lightgbm":
            f = np.concatenate([lgbm_fold(s, X, o) for o in origins])
        elif m == "sarimax":
            f = np.concatenate([sarimax_fold(s, X, o) for o in origins])
        elif m == "keras_lstm":
            f = np.concatenate([keras_lstm_fold(s, o) for o in origins])
        else:
            raise ValueError(m)
        out[m] = metrics(a, f, scale)
    return out
