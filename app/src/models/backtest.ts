/**
 * Rolling-origin backtest (no random splits): for each origin, models are fitted on data up to the
 * origin only, forecast the next `horizon` days, and are scored against what actually happened.
 * Default: 6 folds × 28 days = the last 24 weeks before the current "as of" date.
 */
import type { SeriesData } from "../data/series";
import { computeMetrics, type ForecastMetrics } from "./metrics";

export const MODEL_IDS = ["ets", "ridge", "lstm"] as const;
export type ModelId = (typeof MODEL_IDS)[number];
export type BaselineId = "ma28" | "snaive";
export type AnyModelId = ModelId | BaselineId | "ensemble";

export const MODEL_INFO: Record<AnyModelId, { label: string; kind: string }> = {
  ets: { label: "Holt-Winters (ETS)", kind: "Time series" },
  ridge: { label: "Ridge regression", kind: "Regression" },
  lstm: { label: "LSTM network", kind: "Deep learning" },
  ensemble: { label: "Adaptive ensemble", kind: "AI combination" },
  ma28: { label: "28-day moving average", kind: "Benchmark" },
  snaive: { label: "Seasonal naive (last week)", kind: "Benchmark" },
};

export interface BacktestSettings {
  folds: number;
  horizon: number;
}

export const DEFAULT_BACKTEST: BacktestSettings = { folds: 6, horizon: 28 };

/** Minimum training span (days after the modelling start) before the first fold origin. */
export const MIN_TRAIN_DAYS = 400;

export interface FoldForecast {
  origin: number;
  forecast: Float64Array;
  /** Actual sales for origin+1 … origin+horizon; NaN on closed or unobserved days. */
  actual: Float64Array;
}

/** Fold origins in ascending order: asOf − folds·h, …, asOf − h (dropping any with too little history). */
export function foldOrigins(s: SeriesData, asOf: number, b: BacktestSettings = DEFAULT_BACKTEST): number[] {
  const out: number[] = [];
  for (let k = b.folds; k >= 1; k--) {
    const o = asOf - k * b.horizon;
    if (o - s.modelStart >= MIN_TRAIN_DAYS) out.push(o);
  }
  return out;
}

export function actualsAfter(s: SeriesData, origin: number, horizon: number): Float64Array {
  const a = new Float64Array(horizon);
  for (let k = 1; k <= horizon; k++) {
    const i = origin + k;
    const v = i < s.length ? s.sales[i] : Number.NaN;
    a[k - 1] = i < s.length && s.closed[i] === 0 && v !== undefined && Number.isFinite(v) ? v : Number.NaN;
  }
  return a;
}

/** Benchmarks: 28-day moving average of open days, and seasonal naive (same weekday last week). */
export function forecastBaseline(kind: BaselineId, s: SeriesData, origin: number, horizon: number): Float64Array {
  const out = new Float64Array(horizon);
  const isOpen = (i: number) => i < s.length && s.closed[i] === 0;
  if (kind === "ma28") {
    let sum = 0;
    let n = 0;
    for (let i = origin - 27; i <= origin; i++) {
      const v = s.sales[i];
      if (isOpen(i) && v !== undefined && Number.isFinite(v)) {
        sum += v;
        n++;
      }
    }
    out.fill(n ? sum / n : 0);
  } else {
    for (let k = 1; k <= horizon; k++) {
      let j = origin + k - 7 * Math.ceil(k / 7);
      while (j > 0 && !isOpen(j)) j -= 7; // skip closures to the previous same weekday
      out[k - 1] = s.sales[j] ?? 0;
    }
  }
  for (let k = 1; k <= horizon; k++) if (!isOpen(origin + k) && origin + k < s.length) out[k - 1] = 0;
  return out;
}

/** Pooled metrics over all folds. */
export function foldMetrics(folds: FoldForecast[], naiveScale: number): ForecastMetrics {
  const a: number[] = [];
  const f: number[] = [];
  for (const fold of folds) {
    for (let k = 0; k < fold.actual.length; k++) {
      a.push(fold.actual[k] as number);
      f.push(fold.forecast[k] as number);
    }
  }
  return computeMetrics(a, f, naiveScale);
}
