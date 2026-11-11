/**
 * Monte Carlo simulation of demand during the replenishment lead time (DDLT) — slide step 2:
 * "model risk scenarios and buffer requirements".
 *
 * Each trial draws a lead time L (gamma / truncated normal / empirical) and the daily demands over
 * those L days (fractional last day), then sums them. From the simulated DDLT distribution:
 *   reorder point at the service level = its SL-quantile,  SS_MC = quantile − mean,
 *   stockout probability and expected units short for any given reorder point.
 *
 * Demand models
 *  - forecast-bootstrap (AI):  d_k = f_k · (1 + r_k), where r comes from real out-of-sample relative
 *    forecast errors resampled in weekday-aligned weekly blocks (keeps weekday structure, within-week
 *    autocorrelation and skew)
 *  - forecast-normal:          d_k ~ N(f_k, σ²) i.i.d. — the formula's own assumption
 *  - iid-normal:               d_k ~ N(D̄, σ_D²) i.i.d. — the formula's assumption with constant demand
 *  - history-bootstrap:        d_k drawn from past daily sales — the classic calculator's view
 */
import { createRng, type Rng } from "../lib/random";
import { leadTimeSampler, type LeadTimeModel } from "./leadTime";
import { forecastAt } from "./safetyStock";

export type DemandModel =
  | { kind: "forecast-bootstrap"; forecast: Float64Array; relErrors: readonly Float64Array[] }
  | { kind: "forecast-normal"; forecast: Float64Array; sigma: number }
  | { kind: "iid-normal"; mean: number; sigma: number }
  | { kind: "history-bootstrap"; values: Float64Array };

export interface Scenario {
  id: string;
  label: string;
  demandMultiplier: number;
  leadTimeShift: number;
  leadTimeSdMultiplier: number;
}

export const SCENARIOS: readonly Scenario[] = [
  { id: "base", label: "Normal conditions", demandMultiplier: 1, leadTimeShift: 0, leadTimeSdMultiplier: 1 },
  { id: "delay", label: "Supplier delay (+3 days)", demandMultiplier: 1, leadTimeShift: 3, leadTimeSdMultiplier: 1 },
  { id: "volatile", label: "Erratic supplier (σ_L × 2)", demandMultiplier: 1, leadTimeShift: 0, leadTimeSdMultiplier: 2 },
  { id: "surge", label: "Demand surge (+25%)", demandMultiplier: 1.25, leadTimeShift: 0, leadTimeSdMultiplier: 1 },
  { id: "storm", label: "Perfect storm (all three)", demandMultiplier: 1.25, leadTimeShift: 3, leadTimeSdMultiplier: 2 },
];

export interface MonteCarloInputs {
  demand: DemandModel;
  leadTime: LeadTimeModel;
  serviceLevel: number;
  /** Reorder point to stress-test (e.g. the formula's ROP). */
  reorderPoint?: number;
  /** Order quantity, for the fill-rate estimate. */
  orderQuantity?: number;
  trials: number;
  seed: number;
  scenario?: Scenario;
  bins?: number;
}

export interface MonteCarloResult {
  trials: number;
  mean: number;
  sd: number;
  skewness: number;
  quantiles: { p50: number; p90: number; p95: number; p99: number };
  /** SL-quantile of DDLT: the reorder point the simulation says is needed. */
  reorderPoint: number;
  /** reorderPoint − mean DDLT. */
  safetyStock: number;
  stockoutProbability?: number;
  /** E[(DDLT − ROP)⁺] — expected units short per replenishment cycle. */
  expectedShortage?: number;
  /** 1 − expected shortage / Q. */
  fillRate?: number;
  histogram: { edges: number[]; counts: number[] };
  leadTime: { mean: number; sd: number; p95: number };
  convergence: { n: number; safetyStock: number }[];
}

/** Lead-time model after applying a scenario's shift / volatility multiplier. */
export function scenarioLeadTime(m: LeadTimeModel, sc?: Scenario): LeadTimeModel {
  if (!sc) return m;
  const mean = m.mean + sc.leadTimeShift;
  const samples = m.samples?.map((v) => mean + (v - m.mean) * sc.leadTimeSdMultiplier);
  return { ...m, mean, sd: m.sd * sc.leadTimeSdMultiplier, samples };
}

function demandSampler(model: DemandModel, rng: Rng, multiplier: number): (days: number, frac: number) => number {
  switch (model.kind) {
    case "iid-normal":
      return (days, frac) => {
        let t = 0;
        for (let k = 0; k <= days; k++) {
          if (k === days && frac === 0) break;
          const d = Math.max(0, multiplier * model.mean + model.sigma * rng.normal());
          t += k < days ? d : frac * d;
        }
        return t;
      };
    case "forecast-normal":
      return (days, frac) => {
        let t = 0;
        for (let k = 0; k <= days; k++) {
          if (k === days && frac === 0) break;
          const f = multiplier * forecastAt(model.forecast, k);
          const d = f > 0 ? Math.max(0, f + model.sigma * rng.normal()) : 0;
          t += k < days ? d : frac * d;
        }
        return t;
      };
    case "history-bootstrap": {
      const v = model.values;
      return (days, frac) => {
        let t = 0;
        for (let k = 0; k <= days; k++) {
          if (k === days && frac === 0) break;
          const d = multiplier * (v[rng.int(v.length)] as number);
          t += k < days ? d : frac * d;
        }
        return t;
      };
    }
    case "forecast-bootstrap": {
      // Weekday-aligned weekly block bootstrap. Fold origins sit a whole number of weeks before the
      // live origin, so fold day p and live day k fall on the same weekday when p ≡ k (mod 7). Each
      // simulated week draws one real week of errors (any fold, one of the first three forecast weeks):
      // a Monday error is only ever applied to a Monday, and within-week autocorrelation is kept.
      const blocks = model.relErrors.filter((b) => b.length >= 7);
      if (!blocks.length) throw new Error("Monte Carlo needs out-of-sample forecast errors");
      const weeks = Math.max(1, Math.min(3, Math.floor((blocks[0] as Float64Array).length / 7)));
      return (days, frac) => {
        let block = blocks[0] as Float64Array;
        let week = 0;
        let t = 0;
        for (let k = 0; k <= days; k++) {
          if (k === days && frac === 0) break;
          if (k % 7 === 0) {
            block = blocks[rng.int(blocks.length)] as Float64Array;
            week = rng.int(weeks);
          }
          const f = multiplier * forecastAt(model.forecast, k);
          if (f <= 0) continue;
          const r = block[7 * week + (k % 7)] as number;
          const d = Math.max(0, f * (1 + (Number.isFinite(r) ? r : 0)));
          t += k < days ? d : frac * d;
        }
        return t;
      };
    }
  }
}

function sortedQuantile(sorted: Float64Array, q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return (sorted[lo] as number) + ((sorted[hi] as number) - (sorted[lo] as number)) * (pos - lo);
}

export function runMonteCarlo(inp: MonteCarloInputs): MonteCarloResult {
  const rng = createRng(inp.seed);
  const lt = scenarioLeadTime(inp.leadTime, inp.scenario);
  const sampleL = leadTimeSampler(lt, rng);
  const sampleD = demandSampler(inp.demand, rng, inp.scenario?.demandMultiplier ?? 1);
  const N = Math.max(100, Math.floor(inp.trials));
  const ddlt = new Float64Array(N);
  const lts = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const L = sampleL();
    lts[i] = L;
    const days = Math.floor(L);
    ddlt[i] = sampleD(days, L - days);
  }

  let m = 0;
  for (const v of ddlt) m += v / N;
  let m2 = 0;
  let m3 = 0;
  for (const v of ddlt) {
    m2 += (v - m) ** 2 / N;
    m3 += (v - m) ** 3 / N;
  }
  const sorted = Float64Array.from(ddlt).sort();
  const q = (p: number) => sortedQuantile(sorted, p);
  const rop = q(inp.serviceLevel);

  const out: MonteCarloResult = {
    trials: N,
    mean: m,
    sd: Math.sqrt(m2 * (N / (N - 1))),
    skewness: m2 > 0 ? m3 / m2 ** 1.5 : 0,
    quantiles: { p50: q(0.5), p90: q(0.9), p95: q(0.95), p99: q(0.99) },
    reorderPoint: rop,
    safetyStock: rop - m,
    histogram: histogram(sorted, inp.bins ?? 60),
    leadTime: leadTimeStats(lts),
    convergence: convergence(ddlt, inp.serviceLevel),
  };
  if (inp.reorderPoint !== undefined) {
    const R = inp.reorderPoint;
    let over = 0;
    let short = 0;
    for (const v of ddlt) {
      if (v > R) {
        over++;
        short += v - R;
      }
    }
    out.stockoutProbability = over / N;
    out.expectedShortage = short / N;
    if (inp.orderQuantity && inp.orderQuantity > 0) out.fillRate = Math.max(0, 1 - out.expectedShortage / inp.orderQuantity);
  }
  return out;
}

/** Equal-width histogram over the 0.1%–99.9% range (extreme tails folded into the end bins). */
function histogram(sorted: Float64Array, bins: number): { edges: number[]; counts: number[] } {
  const lo = sortedQuantile(sorted, 0.001);
  const hi = sortedQuantile(sorted, 0.999);
  const width = (hi - lo) / bins || 1;
  const counts = new Array<number>(bins).fill(0);
  for (const v of sorted) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[b] = (counts[b] as number) + 1;
  }
  return { edges: Array.from({ length: bins + 1 }, (_, i) => lo + i * width), counts };
}

function leadTimeStats(lts: Float64Array): MonteCarloResult["leadTime"] {
  const n = lts.length;
  let m = 0;
  for (const v of lts) m += v / n;
  let v2 = 0;
  for (const v of lts) v2 += (v - m) ** 2 / (n - 1);
  return { mean: m, sd: Math.sqrt(v2), p95: sortedQuantile(Float64Array.from(lts).sort(), 0.95) };
}

/** Safety-stock estimate from the first n trials, for n = 500, 1k, 2k, 5k, … — shows the estimate settling. */
function convergence(ddlt: Float64Array, sl: number): { n: number; safetyStock: number }[] {
  const out: { n: number; safetyStock: number }[] = [];
  const steps = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000];
  for (const n of [...steps.filter((s) => s < ddlt.length), ddlt.length]) {
    const prefix = ddlt.slice(0, n);
    let m = 0;
    for (const v of prefix) m += v / n;
    out.push({ n, safetyStock: sortedQuantile(prefix.sort(), sl) - m });
  }
  return out;
}

/** Relative forecast errors r = actual / forecast − 1 per fold (NaN where undefined), for the bootstrap. */
export function relativeErrors(folds: readonly { forecast: Float64Array; actual: Float64Array }[]): Float64Array[] {
  return folds.map((f) => f.actual.map((a, k) => {
    const p = f.forecast[k] as number;
    return Number.isFinite(a) && p > 0 ? a / p - 1 : Number.NaN;
  }));
}

export interface ScenarioResult {
  scenario: Scenario;
  result: MonteCarloResult;
}

/** Stress-test a reorder point under every preset scenario (common seed → comparable results). */
export function runScenarios(base: Omit<MonteCarloInputs, "scenario">, scenarios: readonly Scenario[] = SCENARIOS): ScenarioResult[] {
  return scenarios.map((scenario) => ({ scenario, result: runMonteCarlo({ ...base, scenario }) }));
}
