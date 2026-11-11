/**
 * Prediction intervals by split-conformal calibration on out-of-sample backtest errors.
 * Demand errors scale with the level (weekends sell more), so relative errors r = a / f − 1 are
 * pooled by horizon bucket (days 1–7, 8–14, 15–21, 22–28) and their empirical quantiles define
 * the bands: [f·(1 + q_low), f·(1 + q_high)]. Coverage is then checked on new data as it arrives.
 */
import type { FoldForecast } from "./backtest";
import { quantile } from "./metrics";

export const BUCKET_DAYS = 7;

export interface IntervalModel {
  /** One entry per horizon bucket. */
  buckets: { lo95: number; lo80: number; hi80: number; hi95: number; n: number }[];
}

export interface Bands {
  lo80: Float64Array;
  hi80: Float64Array;
  lo95: Float64Array;
  hi95: Float64Array;
}

export function fitIntervals(folds: FoldForecast[], horizon: number): IntervalModel {
  const nBuckets = Math.ceil(horizon / BUCKET_DAYS);
  const rel: number[][] = Array.from({ length: nBuckets }, () => []);
  for (const fold of folds) {
    for (let k = 0; k < Math.min(horizon, fold.actual.length); k++) {
      const a = fold.actual[k] as number;
      const f = fold.forecast[k] as number;
      if (Number.isFinite(a) && f > 0) (rel[Math.floor(k / BUCKET_DAYS)] as number[]).push(a / f - 1);
    }
  }
  // Buckets without data borrow the nearest earlier bucket (never narrower than it).
  let prev = { lo95: -0.3, lo80: -0.15, hi80: 0.15, hi95: 0.3, n: 0 };
  const buckets = rel.map((r) => {
    if (r.length < 5) return { ...prev, n: r.length };
    prev = { lo95: quantile(r, 0.025), lo80: quantile(r, 0.1), hi80: quantile(r, 0.9), hi95: quantile(r, 0.975), n: r.length };
    return prev;
  });
  return { buckets };
}

export function applyIntervals(forecast: Float64Array, model: IntervalModel): Bands {
  const n = forecast.length;
  const out: Bands = { lo80: new Float64Array(n), hi80: new Float64Array(n), lo95: new Float64Array(n), hi95: new Float64Array(n) };
  const last = model.buckets[model.buckets.length - 1];
  for (let k = 0; k < n; k++) {
    const b = model.buckets[Math.floor(k / BUCKET_DAYS)] ?? last;
    const f = forecast[k] as number;
    if (!b) continue;
    out.lo95[k] = Math.max(0, f * (1 + b.lo95));
    out.lo80[k] = Math.max(0, f * (1 + b.lo80));
    out.hi80[k] = f * (1 + b.hi80);
    out.hi95[k] = f * (1 + b.hi95);
  }
  return out;
}

/** Share of observed actuals inside [lo, hi]. */
export function coverage(actual: ArrayLike<number>, lo: ArrayLike<number>, hi: ArrayLike<number>): { inside: number; n: number; rate: number } {
  let inside = 0;
  let n = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] as number;
    if (!Number.isFinite(a)) continue;
    n++;
    if (a >= (lo[i] as number) && a <= (hi[i] as number)) inside++;
  }
  return { inside, n, rate: n ? inside / n : Number.NaN };
}
