/**
 * Safety stock and reorder point — the slide's formula, implemented exactly:
 *
 *      SS  = Z · √( σ_D² · L  +  D̄² · σ_L² )          Z = Φ⁻¹(service level)
 *      ROP = expected demand during the lead time + SS  (= D̄·L + SS)
 *
 * Units: D̄ and σ_D in units/day; L and σ_L in days → SS and ROP in units.
 *
 * Two ways to feed it (the "enhancement" the exercise asks for):
 *  - Baseline (classic calculator): D̄, σ_D = mean and SD of the last 365 days of sales — static, and
 *    σ_D counts predictable swings (weekends, holidays) as if they were risk.
 *  - AI-enhanced (dynamic): D̄ = forecast demand over the coming lead time ÷ L (moves ahead of
 *    weekends, holidays, December and trend); σ_D → σ_e = RMSE of out-of-sample forecast errors over
 *    the same horizons (only the genuinely unpredictable part, bias included).
 */
import type { SeriesData } from "../data/series";
import { openWindowStats, type WindowStats } from "../data/stats";
import type { ForecastResult } from "../models/engine";
import type { LeadTimeModel } from "./leadTime";
import { inverseCdf } from "./normal";

export interface FormulaInputs {
  /** Cycle service level: target probability of no stockout during a replenishment lead time. */
  serviceLevel: number;
  /** D̄ — average demand (units/day). */
  dBar: number;
  /** σ_D — demand standard deviation (units/day). */
  sigmaD: number;
  /** L — lead time (days). */
  leadTime: number;
  /** σ_L — lead-time variability (days). */
  sigmaL: number;
}

export interface FormulaResult extends FormulaInputs {
  z: number;
  /** σ_D² · L */
  termDemand: number;
  /** D̄² · σ_L² */
  termLeadTime: number;
  /** √(σ_D² L + D̄² σ_L²) — SD of demand during the lead time. */
  sigmaLeadTimeDemand: number;
  safetyStock: number;
  /** Expected demand during the lead time (= D̄ · L). */
  leadTimeDemand: number;
  reorderPoint: number;
  /** Share of lead-time-demand variance caused by lead-time uncertainty. */
  leadTimeShare: number;
  /** Safety stock expressed in days of average demand. */
  daysOfCover: number;
}

export function slideFormula(inp: FormulaInputs): FormulaResult {
  const { serviceLevel, dBar, sigmaD, leadTime, sigmaL } = inp;
  if (!(serviceLevel > 0 && serviceLevel < 1)) throw new RangeError("Service level must be between 0 and 1 (exclusive)");
  if (!(leadTime > 0)) throw new RangeError("Lead time must be positive");
  if (dBar < 0 || sigmaD < 0 || sigmaL < 0) throw new RangeError("Demand and standard deviations cannot be negative");
  const z = inverseCdf(serviceLevel);
  const termDemand = sigmaD * sigmaD * leadTime;
  const termLeadTime = dBar * dBar * sigmaL * sigmaL;
  const sigmaLeadTimeDemand = Math.sqrt(termDemand + termLeadTime);
  const safetyStock = z * sigmaLeadTimeDemand;
  const leadTimeDemand = dBar * leadTime;
  return {
    ...inp,
    z,
    termDemand,
    termLeadTime,
    sigmaLeadTimeDemand,
    safetyStock,
    leadTimeDemand,
    reorderPoint: leadTimeDemand + safetyStock,
    leadTimeShare: termDemand + termLeadTime > 0 ? termLeadTime / (termDemand + termLeadTime) : Number.NaN,
    daysOfCover: dBar > 0 ? safetyStock / dBar : Number.NaN,
  };
}

export interface PolicyParams {
  serviceLevel: number;
  leadTime: LeadTimeModel;
}

export interface BaselinePolicy extends FormulaResult {
  method: "baseline";
  window: WindowStats;
}

/** Classic calculator at `asOf`: trailing-365-day statistics of daily sales (store-open days). */
export function baselinePolicy(s: SeriesData, asOf: number, p: PolicyParams): BaselinePolicy {
  const window = openWindowStats(s, asOf, 365);
  return {
    ...slideFormula({ serviceLevel: p.serviceLevel, dBar: window.mean, sigmaD: window.sd, leadTime: p.leadTime.mean, sigmaL: p.leadTime.sd }),
    method: "baseline",
    window,
  };
}

/** Forecast value for day k (0-based) beyond the horizon: repeat the last forecast week. */
export function forecastAt(forecast: ArrayLike<number>, k: number): number {
  let j = k;
  while (j >= forecast.length) j -= 7;
  return forecast[Math.max(0, j)] as number;
}

/** Expected demand over a (possibly fractional) lead time L starting after `offset` days. */
export function leadTimeDemand(forecast: ArrayLike<number>, L: number, offset = 0): number {
  const whole = Math.floor(L);
  let sum = 0;
  for (let k = 0; k < whole; k++) sum += forecastAt(forecast, offset + k);
  return sum + (L - whole) * forecastAt(forecast, offset + whole);
}

/** RMSE of finite out-of-sample errors at horizons fromH..toH (1-based), pooled over folds. */
export function forecastErrorRmse(foldErrors: readonly Float64Array[], fromH: number, toH: number): { rmse: number; n: number } {
  let ss = 0;
  let n = 0;
  for (const e of foldErrors) {
    for (let h = fromH; h <= Math.min(toH, e.length); h++) {
      const v = e[h - 1] as number;
      if (Number.isFinite(v)) {
        ss += v * v;
        n++;
      }
    }
  }
  return { rmse: n ? Math.sqrt(ss / n) : Number.NaN, n };
}

export interface AiPolicy extends FormulaResult {
  method: "ai";
  /** Number of out-of-sample errors behind σ_e. */
  errorCount: number;
}

/** AI-enhanced safety stock for orders placed now (lead time starts tomorrow). */
export function aiPolicy(result: ForecastResult, p: PolicyParams, offset = 0): AiPolicy {
  const L = p.leadTime.mean;
  const mu = leadTimeDemand(result.ensemble.forecast, L, offset);
  const err = forecastErrorRmse(result.foldErrors, offset + 1, offset + Math.ceil(L));
  return {
    ...slideFormula({ serviceLevel: p.serviceLevel, dBar: mu / L, sigmaD: err.rmse, leadTime: L, sigmaL: p.leadTime.sd }),
    method: "ai",
    errorCount: err.n,
  };
}

export interface DynamicPath {
  dates: string[];
  dBar: number[];
  safetyStock: number[];
  reorderPoint: number[];
}

/** How the AI safety stock and reorder point move over the coming weeks as the forecast changes. */
export function dynamicPath(result: ForecastResult, p: PolicyParams): DynamicPath {
  const last = result.horizon - Math.ceil(p.leadTime.mean);
  const out: DynamicPath = { dates: [], dBar: [], safetyStock: [], reorderPoint: [] };
  for (let t = 0; t <= Math.max(0, last); t++) {
    const pol = aiPolicy(result, p, t);
    out.dates.push(result.dates[t] as string);
    out.dBar.push(pol.dBar);
    out.safetyStock.push(pol.safetyStock);
    out.reorderPoint.push(pol.reorderPoint);
  }
  return out;
}
