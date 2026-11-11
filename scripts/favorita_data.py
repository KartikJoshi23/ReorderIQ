"""
Shared data-preparation logic for the AI reorder-point project (Kaggle "Store Sales" / Favorita).

Every cleaning rule from docs/Phase2_Implementation_Plan.md §1.2 is implemented here exactly once,
so the EDA notebook, the encrypted data-pack builder and the golden test values all agree.

Units: sales are units per day (BEVERAGES, CLEANING, PERSONAL CARE are whole units; GROCERY I
contains weighed goods and can be fractional). Dates are calendar days.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "store-sales-time-series-forecasting"

PACK_START = pd.Timestamp("2013-01-01")    # first date in train.csv
TRAIN_END = pd.Timestamp("2017-08-15")     # last date with observed sales
PACK_END = pd.Timestamp("2017-08-31")      # last date in test.csv (promotions planned in advance)
MODEL_START = pd.Timestamp("2014-04-01")   # promotions are only recorded from this date (zeros before = missing)
REPLAY_START = pd.Timestamp("2017-06-21")  # final 8 weeks are held back and "arrive" in the dashboard

# Curated non-perishable series (screened in Phase 2: on-time store openings, no long closures, no zeros).
SHORTLIST: list[tuple[int, str]] = [
    (44, "BEVERAGES"),
    (44, "CLEANING"),
    (44, "PERSONAL CARE"),
    (44, "GROCERY I"),
    (45, "BEVERAGES"),
    (47, "BEVERAGES"),
    (3, "BEVERAGES"),
    (45, "PERSONAL CARE"),
    (1, "CLEANING"),
]
DEFAULT_SERIES = (44, "BEVERAGES")

# holidays_events.csv types that mean "a non-working celebration happens on this date".
HOLIDAY_TYPES = {"Holiday", "Additional", "Bridge", "Transfer"}


@dataclass(frozen=True)
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    stores: pd.DataFrame


def series_id(store: int, family: str) -> str:
    return f"{store}-{family}"


def daily_index(start: pd.Timestamp = PACK_START, end: pd.Timestamp = PACK_END) -> pd.DatetimeIndex:
    """Continuous daily calendar (the raw data skips 25 December every year)."""
    return pd.date_range(start, end, freq="D")


def load_raw(raw_dir: Path = RAW_DIR, stores_subset: set[int] | None = None) -> RawData:
    """Load the Kaggle CSVs. `stores_subset` limits train/test rows to save memory."""
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"],
                        dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"})
    test = pd.read_csv(raw_dir / "test.csv", parse_dates=["date"],
                       dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"})
    if stores_subset is not None:
        train = train[train.store_nbr.isin(stores_subset)]
        test = test[test.store_nbr.isin(stores_subset)]
    oil = pd.read_csv(raw_dir / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(raw_dir / "holidays_events.csv", parse_dates=["date"])
    holidays["transferred"] = holidays["transferred"].astype(str).str.lower().eq("true")
    stores = pd.read_csv(raw_dir / "stores.csv")
    return RawData(train=train.drop(columns="id"), test=test.drop(columns="id"),
                   oil=oil, holidays=holidays, stores=stores)


def oil_daily(oil: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """WTI price on every calendar day: non-trading days and the 43 null prices are filled by
    linear interpolation; the leading null (2013-01-01) takes the next available price."""
    s = oil.set_index("date")["dcoilwtico"].astype(float)
    full = pd.date_range(min(s.index.min(), index[0]), max(s.index.max(), index[-1]), freq="D")
    s = s.reindex(full).interpolate(method="linear", limit_direction="both")
    return s.reindex(index)


def store_calendar(holidays: pd.DataFrame, store_row: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-store holiday/event flags on the daily index.

    Rules (Kaggle data notes):
      * rows with transferred=True are dropped: the holiday was officially moved and the original
        date behaves like a normal day; the date it moved to appears as type "Transfer";
      * Holiday, Additional, Bridge, Transfer -> holiday = 1;
      * Work Day (a normally non-working day made working) -> work_day = 1;
      * Event -> quake = 1 for the 2016 earthquake ("Terremoto Manabi"), otherwise event = 1;
      * National rows apply to every store, Regional rows match the store's state,
        Local rows match the store's city.
    """
    h = holidays[~holidays["transferred"]]
    applies = ((h.locale == "National")
               | ((h.locale == "Regional") & (h.locale_name == store_row["state"]))
               | ((h.locale == "Local") & (h.locale_name == store_row["city"])))
    h = h[applies & h.date.between(index[0], index[-1])]

    cal = pd.DataFrame(0, index=index, columns=["holiday", "event", "quake", "work_day"], dtype="int8")
    names: dict[pd.Timestamp, list[str]] = {}
    for row in h.itertuples(index=False):
        if row.type in HOLIDAY_TYPES:
            cal.at[row.date, "holiday"] = 1
        elif row.type == "Work Day":
            cal.at[row.date, "work_day"] = 1
        elif row.type == "Event":
            col = "quake" if "terremoto" in row.description.lower() else "event"
            cal.at[row.date, col] = 1
        names.setdefault(row.date, []).append(row.description)
    cal["name"] = pd.Series({d: "; ".join(dict.fromkeys(v)) for d, v in names.items()}, dtype="object")
    return cal


def closed_days(train: pd.DataFrame, store: int, index: pd.DatetimeIndex) -> pd.Series:
    """1 when the store sold nothing at all that day (25 Dec is absent from the data; 1 Jan and
    temporary closures show as zero total sales). Dates after TRAIN_END are assumed open except
    25 Dec / 1 Jan."""
    total = train[train.store_nbr == store].groupby("date", observed=True)["sales"].sum().reindex(index)
    observed = index <= TRAIN_END
    closed = observed & (total.fillna(0).to_numpy() == 0)
    md = index.strftime("%m-%d")
    closed |= (~observed) & np.isin(md, ["12-25", "01-01"])
    return pd.Series(closed.astype("int8"), index=index)


def extract_series(raw: RawData, store: int, family: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily sales (NaN after TRAIN_END = future) and onpromotion (train + planned test promotions)."""
    tr = raw.train[(raw.train.store_nbr == store) & (raw.train.family == family)].set_index("date")
    te = raw.test[(raw.test.store_nbr == store) & (raw.test.family == family)].set_index("date")
    sales = tr["sales"].astype(float).reindex(index)
    sales[(index <= TRAIN_END) & sales.isna().to_numpy()] = 0.0   # 25 Dec rows: closed, nothing sold
    promo = pd.concat([tr["onpromotion"], te["onpromotion"]]).reindex(index).fillna(0).astype(int)
    return pd.DataFrame({"sales": sales, "onpromotion": promo})


def payday(index: pd.DatetimeIndex) -> np.ndarray:
    """Public-sector wages are paid on the 15th and on the last day of each month."""
    return ((index.day == 15) | index.is_month_end).astype(int)


def xmas_ramp(index: pd.DatetimeIndex) -> np.ndarray:
    """Pre-Christmas build-up: day/24 for 1-24 December, else 0."""
    return np.where((index.month == 12) & (index.day <= 24), index.day / 24.0, 0.0)


def open_window_stats(sales: pd.Series, closed: pd.Series, end: pd.Timestamp = TRAIN_END,
                      days: int = 365) -> dict[str, float]:
    """Baseline D-bar and sigma_D: mean and sample SD (ddof=1) of daily sales over the trailing
    `days` calendar days ending `end`, store-open days only."""
    w = (sales.index > end - pd.Timedelta(days=days)) & (sales.index <= end) & (closed.to_numpy() == 0)
    x = sales[w].to_numpy()
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1))}


def demo_design(index: pd.DatetimeIndex, promo: np.ndarray, holiday: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Design matrix used to estimate the synthetic-demo generator (log1p scale)."""
    t = ((index - MODEL_START).days / 365.25).to_numpy()
    doy = index.dayofyear.to_numpy()
    cols = {"const": np.ones(len(index)), "trend": t}
    for d, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
        if d > 0:   # Monday is the reference level
            cols[f"dow_{name}"] = (index.dayofweek == d).astype(float)
    for k in (1, 2, 3):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    cols["log_promo"] = np.log1p(promo)
    cols["holiday"] = holiday.astype(float)
    cols["payday"] = payday(index).astype(float)
    cols["xmas"] = xmas_ramp(index)
    return np.column_stack(list(cols.values())), list(cols.keys())


def estimate_demo_params(raw: RawData, store: int = DEFAULT_SERIES[0], family: str = DEFAULT_SERIES[1]) -> dict:
    """Aggregate statistics for the clearly-labelled synthetic demo mode (no raw data is exported):
    log-linear demand coefficients, AR(1) noise, and a two-state Markov promotion process."""
    idx = daily_index(MODEL_START, TRAIN_END)
    s = extract_series(raw, store, family, idx)
    store_row = raw.stores.set_index("store_nbr").loc[store]
    cal = store_calendar(raw.holidays, store_row, idx)
    closed = closed_days(raw.train, store, idx).to_numpy() == 0
    X, names = demo_design(idx, s.onpromotion.to_numpy(), cal.holiday.to_numpy())
    y = np.log1p(s.sales.to_numpy())
    beta, *_ = np.linalg.lstsq(X[closed], y[closed], rcond=None)
    resid = y[closed] - X[closed] @ beta
    phi = float(np.corrcoef(resid[1:], resid[:-1])[0, 1])
    innov_sd = float(np.std(resid, ddof=len(beta)) * np.sqrt(1 - phi ** 2))

    on = s.onpromotion.to_numpy() > 0
    p_on_on = float(np.mean(on[1:][on[:-1]]))
    p_off_on = float(np.mean(on[1:][~on[:-1]]))
    logc = np.log(s.onpromotion.to_numpy()[on])
    return {
        "source": f"Estimated from Favorita store {store} {family}, {MODEL_START.date()}..{TRAIN_END.date()}",
        "coefficients": dict(zip(names, map(float, beta))),
        "noise": {"ar1": phi, "innovationSd": innov_sd},
        "promo": {"pOnGivenOn": p_on_on, "pOnGivenOff": p_off_on,
                  "logCountMean": float(logc.mean()), "logCountSd": float(logc.std(ddof=1))},
        "fit": {"r2": float(1 - resid.var() / y[closed].var()), "nDays": int(closed.sum())},
    }


def build_pack(raw: RawData, shortlist: list[tuple[int, str]] = SHORTLIST) -> dict:
    """Assemble the compact JSON structure that is gzipped + encrypted for the dashboard."""
    index = daily_index()
    oil = oil_daily(raw.oil, index)
    stores_meta = raw.stores.set_index("store_nbr")
    stores_out, series_out = [], []
    for store in sorted({s for s, _ in shortlist}):
        row = stores_meta.loc[store]
        cal = store_calendar(raw.holidays, row, index)
        closed = closed_days(raw.train, store, index)
        names = {int(i): n for i, n in enumerate(cal["name"]) if isinstance(n, str)}
        stores_out.append({
            "store": int(store), "city": row["city"], "state": row["state"],
            "type": row["type"], "cluster": int(row["cluster"]),
            "holiday": cal["holiday"].tolist(), "event": cal["event"].tolist(),
            "quake": cal["quake"].tolist(), "workDay": cal["work_day"].tolist(),
            "closed": closed.tolist(), "names": names,
        })
    for store, family in shortlist:
        s = extract_series(raw, store, family, index)
        fractional = bool(((s.sales.dropna() % 1) != 0).any())
        sales = [None if np.isnan(v) else (round(float(v), 3) if fractional else int(round(v)))
                 for v in s.sales.to_numpy()]
        series_out.append({
            "id": series_id(store, family), "store": int(store), "family": family,
            "unit": "units", "fractional": fractional,
            "sales": sales, "promo": s.onpromotion.astype(int).tolist(),
        })
    return {
        "schema": "favorita-pack/1",
        "meta": {
            "source": "Kaggle 'Store Sales - Time Series Forecasting' (Corporación Favorita, Ecuador)",
            "url": "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",
            "terms": "Used under the competition rules for non-commercial academic education; not for redistribution.",
        },
        "calendar": {
            "start": str(PACK_START.date()), "days": len(index), "trainEnd": str(TRAIN_END.date()),
            "modelStart": str(MODEL_START.date()), "replayStart": str(REPLAY_START.date()),
            "promoKnownFrom": str(MODEL_START.date()),
            "oil": [round(float(v), 2) for v in oil.to_numpy()],
        },
        "stores": stores_out,
        "series": series_out,
        "defaultSeries": series_id(*DEFAULT_SERIES),
    }
