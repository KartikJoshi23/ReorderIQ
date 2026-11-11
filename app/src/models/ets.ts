/**
 * Time-series model: Holt-Winters exponential smoothing on log(1 + sales) with an additive damped
 * trend and additive weekly seasonality (m = 7). Component form, identical to statsmodels
 * ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add", seasonal_periods=7):
 *
 *   ŷ_t   = ℓ_{t−1} + φ b_{t−1} + s_{t−7}
 *   ℓ_t   = α (y_t − s_{t−7}) + (1 − α)(ℓ_{t−1} + φ b_{t−1})
 *   b_t   = β (ℓ_t − ℓ_{t−1}) + (1 − β) φ b_{t−1}
 *   s_t   = γ (y_t − ℓ_{t−1} − φ b_{t−1}) + (1 − γ) s_{t−7}
 *   ŷ_{T+h} = ℓ_T + (φ + φ² + … + φ^h) b_T + s_{T+h−7k}
 *
 * Closed / unobserved days skip the update (states propagate as forecast). Smoothing parameters are
 * fitted by minimising one-step-ahead squared error with Nelder–Mead on a transformed scale.
 */
import { dayParts } from "../data/dates";
import type { SeriesData } from "../data/series";
import { logistic, logit, nelderMead } from "../lib/optimize";
import { type History, isClosed, smearingFactor, toUnits } from "./history";

export interface EtsParams {
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
}

export interface EtsInit {
  level: number;
  trend: number;
  /** Seasonal state indexed by weekday (0 = Monday). */
  season: number[];
}

export interface EtsModel {
  params: EtsParams;
  /** State after filtering through the origin. */
  level: number;
  trend: number;
  season: Float64Array;
  origin: number;
  smear: number;
  sse: number;
  n: number;
}

export interface EtsFilterResult {
  level: number;
  trend: number;
  season: Float64Array;
  sse: number;
  n: number;
  /** One-step-ahead fitted values (log scale) for every day in [from, to]. */
  fitted: Float64Array;
}

/** Fit window: the last two years before the origin (or from the modelling start if shorter). */
export const ETS_WINDOW_DAYS = 730;
const PHI_MIN = 0.8;
const PHI_MAX = 0.98;

const dowOf = (s: SeriesData, i: number) => dayParts(s.epoch0 + i).dow;

export function etsFilter(
  y: ArrayLike<number>,
  open: ArrayLike<number>,
  dow: (i: number) => number,
  from: number,
  to: number,
  p: EtsParams,
  init: EtsInit,
): EtsFilterResult {
  let level = init.level;
  let trend = init.trend;
  const season = Float64Array.from(init.season);
  const fitted = new Float64Array(to - from + 1);
  let sse = 0;
  let n = 0;
  for (let t = from; t <= to; t++) {
    const d = dow(t);
    const sPrev = season[d] as number;
    const damped = p.phi * trend;
    const yhat = level + damped + sPrev;
    fitted[t - from] = yhat;
    if (open[t]) {
      const yt = y[t] as number;
      const e = yt - yhat;
      sse += e * e;
      n++;
      const newLevel = p.alpha * (yt - sPrev) + (1 - p.alpha) * (level + damped);
      trend = p.beta * (newLevel - level) + (1 - p.beta) * damped;
      season[d] = p.gamma * (yt - level - damped) + (1 - p.gamma) * sPrev;
      level = newLevel;
    } else {
      level = level + damped;
      trend = damped;
    }
  }
  return { level, trend, season, sse, n, fitted };
}

/** Heuristic initial state from the first two weeks of the window (weekday deviations from the mean). */
function initialState(h: History, s: SeriesData, from: number): EtsInit {
  const wk = (k: number) => {
    let sum = 0;
    for (let t = from + 7 * k; t < from + 7 * (k + 1); t++) sum += h.y[t] as number;
    return sum / 7;
  };
  const m1 = wk(0);
  const m2 = wk(1);
  const season = new Array<number>(7).fill(0);
  for (let t = from; t < from + 14; t++) season[dowOf(s, t)] = (season[dowOf(s, t)] as number) + ((h.y[t] as number) - (m1 + m2) / 2) / 2;
  return { level: m1, trend: (m2 - m1) / 7, season };
}

const toParams = (x: number[]): EtsParams => ({
  alpha: logistic(x[0] as number),
  beta: logistic(x[1] as number),
  gamma: logistic(x[2] as number),
  phi: PHI_MIN + (PHI_MAX - PHI_MIN) * logistic(x[3] as number),
});

export function fitEts(s: SeriesData, h: History, origin: number): EtsModel {
  const from = Math.max(s.modelStart, origin - ETS_WINDOW_DAYS + 1);
  const init = initialState(h, s, from);
  const dow = (i: number) => dowOf(s, i);
  const objective = (x: number[]) => etsFilter(h.y, h.open, dow, from, origin, toParams(x), init).sse;
  const x0 = [logit(0.2), logit(0.02), logit(0.1), logit((0.9 - PHI_MIN) / (PHI_MAX - PHI_MIN))];
  const best = nelderMead(objective, x0, { step: 1, maxIterations: 600, tolF: 1e-9 });
  const params = toParams(best.x);
  const run = etsFilter(h.y, h.open, dow, from, origin, params, init);
  const residuals: number[] = [];
  for (let t = from; t <= origin; t++) if (h.open[t]) residuals.push((h.y[t] as number) - (run.fitted[t - from] as number));
  return { params, level: run.level, trend: run.trend, season: run.season, origin, smear: smearingFactor(residuals), sse: run.sse, n: run.n };
}

/** Log-scale forecast for horizon k = 1..steps (statsmodels-compatible). */
export function etsForecastLog(model: Pick<EtsModel, "params" | "level" | "trend" | "season">, dowAhead: (k: number) => number, steps: number): Float64Array {
  const out = new Float64Array(steps);
  let damp = 0;
  let phiK = 1;
  for (let k = 1; k <= steps; k++) {
    phiK *= model.params.phi;
    damp += phiK;
    out[k - 1] = model.level + damp * model.trend + (model.season[dowAhead(k)] as number);
  }
  return out;
}

/** Forecast in units for days origin+1 … origin+steps (closed days → 0). */
export function forecastEts(model: EtsModel, s: SeriesData, steps: number): Float64Array {
  const logF = etsForecastLog(model, (k) => dowOf(s, model.origin + k), steps);
  return logF.map((v, k) => (isClosed(s, model.origin + 1 + k) ? 0 : toUnits(v, model.smear)));
}
