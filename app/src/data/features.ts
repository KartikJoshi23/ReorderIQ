/**
 * Exogenous features — every input is known in advance for the forecast horizon (no leakage).
 * Names and definitions match scripts/favorita_data.py (demo_design), so Python and TypeScript agree.
 *
 *   trend        years since the modelling start
 *   dow_*        weekday dummies (Monday = reference)
 *   sin{k},cos{k} yearly Fourier terms, k = 1..3, using day-of-year / 365.25
 *   log_promo    log(1 + items on promotion)            (promotions — planned in advance)
 *   holiday, event, quake, work_day                     (external: national/regional/local calendar)
 *   payday       15th and last day of the month         (external: public-sector wages)
 *   xmas         day/24 for 1–24 December               (pre-Christmas build-up)
 *   oil_lag7     mean WTI price over the previous 7 days (external: oil-dependent economy)
 */
import { dayParts } from "./dates";
import type { SeriesData } from "./series";

export type FeatureFn = (s: SeriesData, i: number) => number;

const DOW_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"] as const;
const YEAR = 365.25;

const fourier = (k: number, fn: (x: number) => number): FeatureFn => (s, i) =>
  fn((2 * Math.PI * k * dayParts(s.epoch0 + i).dayOfYear) / YEAR);

const registry: Record<string, FeatureFn> = {
  const: () => 1,
  trend: (s, i) => (i - s.modelStart) / YEAR,
  log_promo: (s, i) => Math.log1p(s.promo[i] ?? 0),
  holiday: (s, i) => s.holiday[i] ?? 0,
  event: (s, i) => s.event[i] ?? 0,
  quake: (s, i) => s.quake[i] ?? 0,
  work_day: (s, i) => s.workDay[i] ?? 0,
  payday: (s, i) => {
    const p = dayParts(s.epoch0 + i);
    return p.day === 15 || p.isMonthEnd ? 1 : 0;
  },
  xmas: (s, i) => {
    const p = dayParts(s.epoch0 + i);
    return p.month === 12 && p.day <= 24 ? p.day / 24 : 0;
  },
  oil_lag7: (s, i) => {
    let sum = 0;
    let n = 0;
    for (let j = Math.max(0, i - 7); j < i; j++) {
      const v = s.oil[j];
      if (v !== undefined && Number.isFinite(v)) {
        sum += v;
        n++;
      }
    }
    return n ? sum / n : (s.oil[i] ?? 0);
  },
};
DOW_NAMES.forEach((name, d) => {
  if (d > 0) registry[`dow_${name}`] = (s, i) => (dayParts(s.epoch0 + i).dow === d ? 1 : 0);
});
for (const k of [1, 2, 3]) {
  registry[`sin${k}`] = fourier(k, Math.sin);
  registry[`cos${k}`] = fourier(k, Math.cos);
}

export const FEATURES: Readonly<Record<string, FeatureFn>> = registry;

/** Feature set used by the regression and LSTM models (Part B may prune it via model selection). */
export const DEFAULT_FEATURES: readonly string[] = [
  "trend",
  "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
  "sin1", "cos1", "sin2", "cos2", "sin3", "cos3",
  "log_promo", "holiday", "event", "quake", "work_day", "payday", "xmas", "oil_lag7",
];

export interface Design {
  names: readonly string[];
  rows: number;
  cols: number;
  /** Row-major values: data[r * cols + c]. */
  data: Float64Array;
}

export function featureValue(name: string, s: SeriesData, i: number): number {
  const fn = FEATURES[name];
  if (!fn) throw new Error(`Unknown feature "${name}"`);
  return fn(s, i);
}

/** Build a row-major design matrix for the given day indices. */
export function designMatrix(s: SeriesData, indices: readonly number[], names: readonly string[] = DEFAULT_FEATURES): Design {
  const fns = names.map((n) => {
    const fn = FEATURES[n];
    if (!fn) throw new Error(`Unknown feature "${n}"`);
    return fn;
  });
  const cols = fns.length;
  const data = new Float64Array(indices.length * cols);
  indices.forEach((i, r) => {
    for (let c = 0; c < cols; c++) data[r * cols + c] = (fns[c] as FeatureFn)(s, i);
  });
  return { names, rows: indices.length, cols, data };
}
