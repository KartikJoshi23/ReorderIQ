/**
 * Bayesian optimisation of the reorder policy under constraints, plus the exhaustive-grid check and
 * the AI-driven sensitivity analysis.
 *
 * Decision variables (2-D):
 *   dynamic (AI) mode  x = [buffer b, order quantity Q]  with  R_t = forecast lead-time demand_t + b
 *   static mode        x = [reorder point R, order quantity Q]
 * Objective:   expected cost per day (holding + ordering + lost-sales penalty), simulated (policySim.ts)
 * Constraints: fill rate ≥ target, 95th-pct peak stock ≤ capacity, average inventory value ≤ budget,
 *              Q ≥ MOQ (search bound), deliveries per week ≤ dock limit.
 * Feasibility margin g = min over constraints of the normalised slack (g ≥ 0 ⇔ all constraints met).
 */
import { type Constraints, type CostParams, economicOrderQuantity } from "../inventory/costs";
import { type PolicyMetrics, type PolicySpec, type ScenarioSet, simulatePolicy } from "../inventory/policySim";
import { type FormulaInputs, slideFormula } from "../inventory/safetyStock";
import { type BoIteration, type BoOptions, type BoResult, bayesianOptimize, type BoProblem } from "./bayesOpt";

export type PolicyMode = "dynamic" | "static";

export interface OptimizeSettings {
  mode: PolicyMode;
  costs: CostParams;
  constraints: Constraints;
  /** Reference safety stock (e.g. the AI formula's) used to size the search box. */
  referenceSafetyStock: number;
}

export interface Evaluated {
  policy: PolicySpec;
  metrics: PolicyMetrics;
  margins: { fill: number; capacity: number; budget: number; orders: number };
}

export function feasibilityMargins(m: PolicyMetrics, c: Constraints): Evaluated["margins"] {
  return {
    fill: (m.fillRate - c.minFillRate) / Math.max(1e-6, 1 - c.minFillRate),
    capacity: (c.capacity - m.peakOnHandP95) / c.capacity,
    budget: (c.budget - m.avgInventoryValue) / c.budget,
    orders: (c.maxOrdersPerWeek - m.ordersPerWeek) / c.maxOrdersPerWeek,
  };
}

/** Search box: buffer/R around the formula's safety stock; Q from max(MOQ, 1 day) to 28 days of demand. */
export function searchBounds(set: ScenarioSet, s: OptimizeSettings): [number, number][] {
  const D = set.meanDemand;
  const ss = Math.max(s.referenceSafetyStock, 0.5 * D);
  const meanLtd = set.expectedLeadTimeDemand.reduce((a, b) => a + b, 0) / set.expectedLeadTimeDemand.length;
  const qLo = Math.max(s.constraints.moq, D);
  const qHi = Math.max(qLo * 1.5, 28 * D);
  return s.mode === "dynamic" ? [[-0.5 * ss, 3 * ss], [qLo, qHi]] : [[Math.max(0, meanLtd - 0.5 * ss), meanLtd + 3 * ss], [qLo, qHi]];
}

export function toPolicy(mode: PolicyMode, x: readonly number[]): PolicySpec {
  return mode === "dynamic" ? { kind: "dynamic", buffer: x[0] as number, orderQty: x[1] as number } : { kind: "static", reorderPoint: x[0] as number, orderQty: x[1] as number };
}

export function evaluatePolicy(set: ScenarioSet, policy: PolicySpec, s: Pick<OptimizeSettings, "costs" | "constraints">): Evaluated {
  const metrics = simulatePolicy(set, policy, s.costs);
  return { policy, metrics, margins: feasibilityMargins(metrics, s.constraints) };
}

export function inventoryProblem(set: ScenarioSet, s: OptimizeSettings): BoProblem<Evaluated> {
  return {
    bounds: searchBounds(set, s),
    evaluate(x) {
      const ev = evaluatePolicy(set, toPolicy(s.mode, x), s);
      return { objective: ev.metrics.costPerDay, feasibility: Math.min(...Object.values(ev.margins)), details: ev };
    },
  };
}

export function optimizeInventory(set: ScenarioSet, s: OptimizeSettings, opts: BoOptions = {}): BoResult<Evaluated> {
  return bayesianOptimize(inventoryProblem(set, s), { initPoints: 8, iterations: 30, ...opts });
}

export interface GridResult {
  n: number;
  evaluations: number;
  best: { x: number[]; objective: number; details: Evaluated } | null;
}

/** Exhaustive n×n grid over the same box — the "manual parameter sweep" BO replaces, used to validate it. */
export function gridSearch(set: ScenarioSet, s: OptimizeSettings, n = 30): GridResult {
  const problem = inventoryProblem(set, s);
  const [[a0, a1], [b0, b1]] = problem.bounds as [[number, number], [number, number]];
  let best: GridResult["best"] = null;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = [a0 + ((a1 - a0) * i) / (n - 1), b0 + ((b1 - b0) * j) / (n - 1)];
      const ev = problem.evaluate(x);
      if (ev.feasibility >= 0 && (!best || ev.objective < best.objective)) best = { x, objective: ev.objective, details: ev.details as Evaluated };
    }
  }
  return { n, evaluations: n * n, best };
}

/** Baseline comparison policy: static formula ROP with the economic order quantity. */
export function baselineStaticPolicy(baselineRop: number, meanDemand: number, costs: CostParams): PolicySpec {
  return { kind: "static", reorderPoint: baselineRop, orderQty: economicOrderQuantity(meanDemand, costs) };
}

// ---------------------------------------------------------------------------------------------
// AI-driven sensitivity analysis
// ---------------------------------------------------------------------------------------------

export interface TornadoBar {
  input: string;
  label: string;
  low: { value: number; rop: number };
  high: { value: number; rop: number };
  swing: number;
}

/** Reorder-point sensitivity to ±20 % in each formula input (service level: 90 % ↔ 99 %). */
export function tornado(base: FormulaInputs, change = 0.2): { baseRop: number; bars: TornadoBar[] } {
  const rop = (inp: FormulaInputs) => slideFormula(inp).reorderPoint;
  const vary = (key: keyof FormulaInputs, label: string, lo: number, hi: number): TornadoBar => {
    const l = rop({ ...base, [key]: lo });
    const h = rop({ ...base, [key]: hi });
    return { input: key, label, low: { value: lo, rop: l }, high: { value: hi, rop: h }, swing: Math.abs(h - l) };
  };
  const bars = [
    vary("dBar", "Average demand D̄", base.dBar * (1 - change), base.dBar * (1 + change)),
    vary("sigmaD", "Demand uncertainty σ", base.sigmaD * (1 - change), base.sigmaD * (1 + change)),
    vary("leadTime", "Lead time L", base.leadTime * (1 - change), base.leadTime * (1 + change)),
    vary("sigmaL", "Lead-time variability σ_L", base.sigmaL * (1 - change), base.sigmaL * (1 + change)),
    vary("serviceLevel", "Service level", 0.9, 0.99),
  ].sort((a, b) => b.swing - a.swing);
  return { baseRop: rop(base), bars };
}

/** Reorder point over a service-level × lead-time-variability grid (formula, instant). */
export function ropHeatmap(base: FormulaInputs, serviceLevels: number[], sigmaLs: number[]): number[][] {
  return sigmaLs.map((sL) => serviceLevels.map((sl) => slideFormula({ ...base, serviceLevel: sl, sigmaL: sL }).reorderPoint));
}

export interface FrontierPoint {
  minFillRate: number;
  costPerDay: number | null;
  policy: PolicySpec | null;
  metrics: PolicyMetrics | null;
}

/** Cost–service frontier: the optimal cost for each fill-rate target (one BO run per target). */
export function costServiceFrontier(set: ScenarioSet, s: OptimizeSettings, targets = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995], opts: BoOptions = {}, onPoint?: (p: FrontierPoint) => void): FrontierPoint[] {
  return targets.map((t, i) => {
    const r = optimizeInventory(set, { ...s, constraints: { ...s.constraints, minFillRate: t } }, { iterations: 20, seed: 100 + i, ...opts });
    const p: FrontierPoint = { minFillRate: t, costPerDay: r.best?.objective ?? null, policy: r.best?.details?.policy ?? null, metrics: r.best?.details?.metrics ?? null };
    onPoint?.(p);
    return p;
  });
}

/** Shadow price of storage: daily cost saved if capacity were 10 % larger (re-optimised). */
export function capacityShadowPrice(set: ScenarioSet, s: OptimizeSettings, baseBest: number | null, opts: BoOptions = {}): { extraCapacity: number; costPerDay: number | null; savingPerDay: number | null } {
  const extra = Math.round(0.1 * s.constraints.capacity);
  const r = optimizeInventory(set, { ...s, constraints: { ...s.constraints, capacity: s.constraints.capacity + extra } }, { iterations: 20, seed: 7, ...opts });
  const c = r.best?.objective ?? null;
  return { extraCapacity: extra, costPerDay: c, savingPerDay: c !== null && baseBest !== null ? baseBest - c : null };
}

export type { BoIteration };
