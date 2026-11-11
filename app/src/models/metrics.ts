/**
 * Forecast accuracy metrics (original units, store-open days only).
 *   MAE  = mean |a − f|
 *   RMSE = √ mean (a − f)²
 *   WAPE = Σ|a − f| / Σ a              (robust to zero-sales days, unlike MAPE)
 *   Bias = Σ(f − a) / Σ a              (+ = over-forecasting)
 *   MASE = MAE / in-sample MAE of the seasonal-naive (lag-7) forecast  (< 1 beats seasonal naive)
 */
export interface ForecastMetrics {
  n: number;
  mae: number;
  rmse: number;
  wape: number;
  bias: number;
  mase: number;
}

export function computeMetrics(actual: ArrayLike<number>, forecast: ArrayLike<number>, naiveScale: number): ForecastMetrics {
  let n = 0;
  let abs = 0;
  let sq = 0;
  let sumA = 0;
  let sumE = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] as number;
    const f = forecast[i] as number;
    if (!Number.isFinite(a) || !Number.isFinite(f)) continue;
    const e = f - a;
    n++;
    abs += Math.abs(e);
    sq += e * e;
    sumA += a;
    sumE += e;
  }
  const mae = n ? abs / n : Number.NaN;
  return {
    n,
    mae,
    rmse: n ? Math.sqrt(sq / n) : Number.NaN,
    wape: sumA > 0 ? abs / sumA : Number.NaN,
    bias: sumA > 0 ? sumE / sumA : Number.NaN,
    mase: naiveScale > 0 ? mae / naiveScale : Number.NaN,
  };
}

/** In-sample mean absolute error of the seasonal-naive forecast y_t ≈ y_{t−m}, over pairs where both days are observed. */
export function seasonalNaiveScale(sales: ArrayLike<number>, usable: (i: number) => boolean, from: number, to: number, m = 7): number {
  let sum = 0;
  let n = 0;
  for (let t = Math.max(from, m); t <= to; t++) {
    if (!usable(t) || !usable(t - m)) continue;
    sum += Math.abs((sales[t] as number) - (sales[t - m] as number));
    n++;
  }
  return n ? sum / n : Number.NaN;
}

/** Empirical quantile with linear interpolation (numpy default), ignoring non-finite values. */
export function quantile(values: ArrayLike<number>, q: number): number {
  const v = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return Number.NaN;
  const pos = (v.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return (v[lo] as number) + ((v[hi] as number) - (v[lo] as number)) * (pos - lo);
}
