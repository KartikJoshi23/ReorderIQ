/**
 * Log-scale demand history shared by all forecasters.
 *
 *  y[i]    = log(1 + sales_i) for observed open days;
 *            closed or unobserved days are imputed with y[i − 7] (same weekday last week), so lag
 *            and moving-average inputs stay meaningful across store closures.
 *  open[i] = 1 when day i is a usable training target (observed and the store was open).
 */
import type { SeriesData } from "../data/series";

export interface History {
  y: Float64Array;
  open: Uint8Array;
  /** Last index included (the forecast origin). */
  end: number;
}

export function buildHistory(s: SeriesData, end: number): History {
  const y = new Float64Array(end + 1);
  const open = new Uint8Array(end + 1);
  let lastOpen = 0;
  for (let i = 0; i <= end; i++) {
    const v = s.sales[i];
    if (s.closed[i] === 0 && v !== undefined && Number.isFinite(v)) {
      y[i] = Math.log1p(v);
      open[i] = 1;
      lastOpen = y[i] as number;
    } else {
      y[i] = i >= 7 ? (y[i - 7] as number) : lastOpen;
    }
  }
  return { y, open, end };
}

/** Is day i a closed day? Days beyond the calendar use the 25 Dec / 1 Jan rule. */
export function isClosed(s: SeriesData, i: number): boolean {
  return i < s.length ? s.closed[i] === 1 : false;
}

/** Autoregressive inputs at day t, computed from a log series that may contain earlier forecasts. */
export const AR_FEATURES = ["lag7", "lag14", "ma7", "ma28"] as const;

export function arFeatures(y: ArrayLike<number>, t: number, out: Float64Array, offset: number): void {
  out[offset] = y[t - 7] as number;
  out[offset + 1] = y[t - 14] as number;
  let s7 = 0;
  let s28 = 0;
  for (let k = 1; k <= 28; k++) {
    const v = y[t - k] as number;
    if (k <= 7) s7 += v;
    s28 += v;
  }
  out[offset + 2] = s7 / 7;
  out[offset + 3] = s28 / 28;
}

/** Mean of y over the 28 days before t (the level the LSTM forecasts deviations from). */
export function level28(y: ArrayLike<number>, t: number): number {
  let s = 0;
  for (let k = 1; k <= 28; k++) s += y[t - k] as number;
  return s / 28;
}

/** Duan smearing factor: mean of exp(residual) on the log scale, so expm1-back-transformed
 *  forecasts estimate the mean (not the median) of demand. */
export function smearingFactor(residuals: ArrayLike<number>): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < residuals.length; i++) {
    const r = residuals[i] as number;
    if (Number.isFinite(r)) {
      s += Math.exp(r);
      n++;
    }
  }
  return n ? s / n : 1;
}

/** Back-transform a log1p forecast to units, applying smearing and a floor at zero. */
export function toUnits(logValue: number, smear: number): number {
  return Math.max(0, Math.exp(logValue) * smear - 1);
}
