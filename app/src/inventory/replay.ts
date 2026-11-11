/**
 * "Baseline vs AI" on real held-back demand — the enhancement the exercise objective asks for.
 *
 * The final weeks of real sales (never seen by the models at the start) are replayed through three
 * reorder policies with identical, randomly drawn delivery delays (500 paths, common random numbers):
 *   baseline   classic calculator: static ROP from the last 365 days' mean and SD, EOQ order size
 *   AI formula slide formula with forecast inputs, re-planned weekly as each new week of sales
 *              arrives (forecast lead-time demand + SS from forecast errors), EOQ order size
 *   AI optimised  the Bayesian-optimised buffer and order quantity on top of the weekly forecast
 * Unserved demand is lost. Metrics are averaged over the delay paths; paired per-path comparisons
 * show how often the AI policy does better under the same delays.
 *
 * Warm-up: every simulated policy starts with stock R + Q ("just after a delivery"), which is a full
 * lead time of demand above where a running policy sits after a delivery. To keep that artificial
 * starting stock out of the results, each policy first runs through the real sales of the
 * REPLAY_WARMUP_DAYS before the window (unscored), then is scored on the held-back weeks only.
 */
import type { SeriesData } from "../data/series";
import { dateAt, lastObservedIndex } from "../data/series";
import type { ForecastResult } from "../models/engine";
import { type CostParams, economicOrderQuantity } from "./costs";
import type { LeadTimeModel } from "./leadTime";
import { buildReplaySet, type PathMetrics, type PolicyMetrics, type PolicySpec, type PolicyTrace, simulatePolicy, withReorderLevels } from "./policySim";
import { aiPolicy, baselinePolicy } from "./safetyStock";

export type ReplayPolicyId = "baseline" | "aiFormula" | "optimised";

export interface ReplayPlan {
  /** First replayed day (index). */
  start: number;
  /** Last replayed day (index). */
  end: number;
  /** Forecast origins: the day before each replayed week. */
  weekOrigins: number[];
}

export function replayPlan(s: SeriesData): ReplayPlan {
  const start = s.replayStart;
  const end = Math.min(s.trainEnd, lastObservedIndex(s));
  const weekOrigins: number[] = [];
  for (let o = start - 1; o < end; o += 7) weekOrigins.push(o);
  return { start, end, weekOrigins };
}

export interface ReplayInputs {
  /** Series exactly as in the data pack (user-added data excluded). */
  series: SeriesData;
  /** Forecast issued at each week origin (same order as `replayPlan(series).weekOrigins`). */
  weekly: readonly ForecastResult[];
  serviceLevel: number;
  leadTime: LeadTimeModel;
  costs: CostParams;
  /** Bayesian-optimised buffer and order quantity (from the replay start), if available. */
  optimised?: { buffer: number; orderQty: number } | null;
  paths?: number;
  seed?: number;
}

export interface ReplayPolicyResult {
  label: string;
  spec: PolicySpec;
  /** Reorder level actually used on each replayed day (units). */
  reorderLevel: Float64Array;
  metrics: PolicyMetrics;
  /** Day-by-day trace of delay path 0. */
  trace: PolicyTrace;
  paths: PathMetrics;
}

/** Days of real sales before the window that each policy runs through, unscored, before it is evaluated. */
export const REPLAY_WARMUP_DAYS = 28;

export interface ReplayResult {
  dates: string[];
  actual: Float64Array;
  weeks: number;
  pathsSimulated: number;
  /** Unscored warm-up days of real sales simulated before the window. */
  warmupDays: number;
  policies: Partial<Record<ReplayPolicyId, ReplayPolicyResult>>;
  /** Paired comparisons across delay paths (same delays for every policy). */
  paired: {
    aiFillAtLeastBaseline: number;
    aiCheaperThanBaseline: number;
    optimisedCheaperThanBaseline: number | null;
  };
}

export const REPLAY_LABELS: Record<ReplayPolicyId, string> = {
  baseline: "Baseline (static ROP, EOQ)",
  aiFormula: "AI formula (weekly re-planned)",
  optimised: "AI optimised (Bayesian)",
};

export function runReplay(inp: ReplayInputs): ReplayResult {
  const s = inp.series;
  const plan = replayPlan(s);
  const n = plan.end - plan.start + 1;
  if (n <= 0 || !inp.weekly.length) throw new Error("Nothing to replay: no held-back days or no weekly forecasts.");
  const W = Math.max(0, Math.min(REPLAY_WARMUP_DAYS, plan.start));
  const salesAt = (i: number) => {
    const v = s.sales[i] as number;
    return s.closed[i] || !Number.isFinite(v) ? 0 : v;
  };
  // Warm-up days (real sales before the window), then the replayed window.
  const path = Float64Array.from({ length: W + n }, (_, k) => salesAt(plan.start - W + k));
  const actual = path.slice(W);
  const p = { serviceLevel: inp.serviceLevel, leadTime: inp.leadTime };

  // AI: day t belongs to replay week w; an order placed at the end of day t covers forecast offsets
  // from (t − 7w + 1) of the forecast issued at that week's origin. Warm-up days use the first
  // week's plan (offset 1): they only bring the stock to a realistic level and are not scored.
  const aiLevel = new Float64Array(W + n);
  const leadTimeDemand = new Float64Array(W + n);
  for (let t = 0; t < n; t++) {
    const w = Math.min(Math.floor(t / 7), inp.weekly.length - 1);
    const pol = aiPolicy(inp.weekly[w] as ForecastResult, p, t - 7 * w + 1);
    aiLevel[W + t] = pol.reorderPoint;
    leadTimeDemand[W + t] = pol.leadTimeDemand;
  }
  aiLevel.fill(aiLevel[W] as number, 0, W);
  leadTimeDemand.fill(leadTimeDemand[W] as number, 0, W);

  const base = baselinePolicy(s, plan.start - 1, p);
  const shared = buildReplaySet(path, inp.leadTime, leadTimeDemand, inp.paths ?? 500, inp.seed ?? 23, W);
  const run = (label: string, spec: PolicySpec, levels: Float64Array): ReplayPolicyResult => {
    const set = withReorderLevels(shared, levels);
    const r = simulatePolicy(set, spec, inp.costs, 0);
    const window = levels.slice(W);
    const reorderLevel = spec.kind === "static" ? new Float64Array(n).fill(spec.reorderPoint) : window.map((v) => v + spec.buffer);
    const tr = r.trace as PolicyTrace;
    const trace: PolicyTrace = {
      onHand: tr.onHand.slice(W),
      reorderLevel: tr.reorderLevel.slice(W),
      demand: tr.demand.slice(W),
      lost: tr.lost.slice(W),
      orders: tr.orders.filter((o) => o.day >= W).map((o) => ({ ...o, day: o.day - W, arrival: o.arrival - W })),
    };
    return { label, spec, reorderLevel, metrics: r, trace, paths: r.paths };
  };

  const policies: ReplayResult["policies"] = {
    baseline: run(REPLAY_LABELS.baseline, { kind: "static", reorderPoint: base.reorderPoint, orderQty: economicOrderQuantity(base.dBar, inp.costs) }, leadTimeDemand),
    aiFormula: run(REPLAY_LABELS.aiFormula, { kind: "dynamic", buffer: 0, orderQty: economicOrderQuantity(aiPolicy(inp.weekly[0] as ForecastResult, p).dBar, inp.costs) }, aiLevel),
  };
  if (inp.optimised) policies.optimised = run(REPLAY_LABELS.optimised, { kind: "dynamic", buffer: inp.optimised.buffer, orderQty: inp.optimised.orderQty }, leadTimeDemand);

  const share = (a: Float64Array, b: Float64Array, better: (x: number, y: number) => boolean) => {
    let k = 0;
    for (let i = 0; i < a.length; i++) if (better(a[i] as number, b[i] as number)) k++;
    return k / a.length;
  };
  const b = policies.baseline as ReplayPolicyResult;
  const a = policies.aiFormula as ReplayPolicyResult;
  return {
    dates: Array.from({ length: n }, (_, t) => dateAt(s, plan.start + t)),
    actual,
    weeks: plan.weekOrigins.length,
    pathsSimulated: shared.demand.length,
    warmupDays: W,
    policies,
    paired: {
      aiFillAtLeastBaseline: share(a.paths.fillRate, b.paths.fillRate, (x, y) => x >= y - 1e-12),
      aiCheaperThanBaseline: share(a.paths.costPerDay, b.paths.costPerDay, (x, y) => x < y),
      optimisedCheaperThanBaseline: policies.optimised ? share(policies.optimised.paths.costPerDay, b.paths.costPerDay, (x, y) => x < y) : null,
    },
  };
}
