/**
 * Descriptive statistics used by the baseline (non-AI) reorder-point calculator.
 */
import type { SeriesData } from "./series";

export function mean(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i] as number;
  return s / xs.length;
}

/** Sample standard deviation (ddof = 1), computed with Welford's algorithm for numerical stability. */
export function sampleSd(xs: ArrayLike<number>): number {
  let m = 0;
  let m2 = 0;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] as number;
    const d = x - m;
    m += d / (i + 1);
    m2 += d * (x - m);
  }
  return xs.length > 1 ? Math.sqrt(m2 / (xs.length - 1)) : Number.NaN;
}

export interface WindowStats {
  n: number;
  mean: number;
  sd: number;
}

/**
 * Baseline D̄ and σ_D: mean and sample SD of daily sales over the trailing `days` calendar days
 * ending at `endIndex` (inclusive), store-open observed days only. Units: units/day.
 */
export function openWindowStats(s: SeriesData, endIndex: number, days = 365): WindowStats {
  const values: number[] = [];
  for (let i = Math.max(0, endIndex - days + 1); i <= endIndex; i++) {
    const v = s.sales[i];
    if (s.closed[i] === 0 && v !== undefined && Number.isFinite(v)) values.push(v);
  }
  return { n: values.length, mean: mean(values), sd: sampleSd(values) };
}
