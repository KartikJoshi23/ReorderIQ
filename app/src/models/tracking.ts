/**
 * Live accuracy tracking: every forecast the system issues is kept, and as new actuals arrive it is
 * scored on data it never saw. This is the evidence that continuous updating improves (or not)
 * accuracy over time.
 */
import type { SeriesData } from "../data/series";
import { dateAt } from "../data/series";
import { coverage } from "./intervals";
import { computeMetrics } from "./metrics";

export interface IssuedForecast {
  version: number;
  asOf: number;
  /** Ensemble forecast for asOf+1 … asOf+n. */
  forecast: Float64Array;
  lo80: Float64Array;
  hi80: Float64Array;
}

export interface LiveScore {
  version: number;
  asOfDate: string;
  /** Number of scored days (observed, open) within the first `days` of the forecast. */
  n: number;
  wape: number;
  bias: number;
  coverage80: number;
}

export function scoreIssued(issued: readonly IssuedForecast[], s: SeriesData, knownThrough: number, days = 7): LiveScore[] {
  const out: LiveScore[] = [];
  for (const f of issued) {
    const a: number[] = [];
    const p: number[] = [];
    const lo: number[] = [];
    const hi: number[] = [];
    for (let k = 0; k < Math.min(days, f.forecast.length); k++) {
      const i = f.asOf + 1 + k;
      if (i > knownThrough || i >= s.length) break;
      const v = s.sales[i] as number;
      if (s.closed[i] === 1 || !Number.isFinite(v)) continue;
      a.push(v);
      p.push(f.forecast[k] as number);
      lo.push(f.lo80[k] as number);
      hi.push(f.hi80[k] as number);
    }
    if (!a.length) continue;
    const m = computeMetrics(a, p, Number.NaN);
    out.push({ version: f.version, asOfDate: dateAt(s, f.asOf), n: a.length, wape: m.wape, bias: m.bias, coverage80: coverage(a, lo, hi).rate });
  }
  return out;
}
