/**
 * Stochastic inventory simulation of a continuous-review (R, Q) policy with lost sales — the
 * objective function that Bayesian optimisation minimises (slide step 3).
 *
 * Each simulated day: (1) receive orders due today, (2) serve demand from stock (unserved demand
 * is lost), (3) if inventory position (on hand + on order) ≤ the reorder level, order enough
 * multiples of Q to lift it above the level; the order arrives after a random lead time.
 *
 * Policies
 *   static   R fixed (classic calculator)
 *   dynamic  R_t = forecast demand over the coming lead time + buffer b (AI-enhanced: the reorder
 *            point moves ahead of weekends, holidays and trend; BO chooses b and Q)
 *
 * Common random numbers: every candidate policy is evaluated on the same demand and lead-time
 * paths, so differences in cost come from the policy, not from simulation noise.
 */
import { createRng } from "../lib/random";
import { type CostParams, dailyHoldingCost } from "./costs";
import { leadTimeSampler, type LeadTimeModel } from "./leadTime";
import { forecastAt, leadTimeDemand } from "./safetyStock";

export interface ScenarioSetInputs {
  /** Daily expected demand from tomorrow (ensemble forecast); repeated weekly beyond its horizon. */
  forecast: Float64Array;
  /** Out-of-sample relative forecast errors per fold (weekday-aligned weekly blocks are resampled). */
  relErrors: readonly Float64Array[];
  leadTime: LeadTimeModel;
  /** Days evaluated after the warm-up (default 91 = 13 weeks). */
  days?: number;
  warmup?: number;
  paths?: number;
  seed?: number;
  demandMultiplier?: number;
}

export interface ScenarioSet {
  warmup: number;
  days: number;
  demand: Float64Array[];
  /** Integer lead times (days) consumed in order by each path's replenishment orders. */
  leadTimes: Int32Array[];
  /** Forecast demand during a lead time for an order placed at the end of each day. */
  expectedLeadTimeDemand: Float64Array;
  meanDemand: number;
}

const MAX_ORDERS_PER_PATH = 200;

export function buildScenarioSet(inp: ScenarioSetInputs): ScenarioSet {
  const warmup = inp.warmup ?? 14;
  const days = inp.days ?? 91;
  const P = inp.paths ?? 64;
  const mult = inp.demandMultiplier ?? 1;
  const total = warmup + days;
  const rng = createRng(inp.seed ?? 17);
  const sampleL = leadTimeSampler(inp.leadTime, rng);
  const blocks = inp.relErrors.filter((b) => b.length >= 7);
  if (!blocks.length) throw new Error("Policy simulation needs out-of-sample forecast errors");
  const weeks = Math.max(1, Math.min(3, Math.floor((blocks[0] as Float64Array).length / 7)));

  const demand: Float64Array[] = [];
  const leadTimes: Int32Array[] = [];
  let sum = 0;
  for (let p = 0; p < P; p++) {
    const path = new Float64Array(total);
    let block = blocks[0] as Float64Array;
    let week = 0;
    for (let k = 0; k < total; k++) {
      if (k % 7 === 0) {
        block = blocks[rng.int(blocks.length)] as Float64Array;
        week = rng.int(weeks);
      }
      const f = mult * forecastAt(inp.forecast, k);
      const r = block[7 * week + (k % 7)] as number;
      path[k] = f > 0 ? Math.max(0, f * (1 + (Number.isFinite(r) ? r : 0))) : 0;
      if (k >= warmup) sum += path[k] as number;
    }
    demand.push(path);
    leadTimes.push(Int32Array.from({ length: MAX_ORDERS_PER_PATH }, () => Math.max(1, Math.round(sampleL()))));
  }
  const L = inp.leadTime.mean;
  return {
    warmup,
    days,
    demand,
    leadTimes,
    expectedLeadTimeDemand: Float64Array.from({ length: total }, (_, k) => mult * leadTimeDemand(inp.forecast, L, k + 1)),
    meanDemand: sum / (P * days),
  };
}

/**
 * Replay of a real demand path: every path shares the actual daily sales and only the delivery
 * delays differ. `reorderLevels[t]` is the base level a dynamic policy adds its buffer to on day t.
 * The first `warmup` days of `actual` (real sales before the evaluated window) are simulated but
 * not scored, so the artificial starting stock does not count as part of the results.
 */
export function buildReplaySet(actual: Float64Array, leadTime: LeadTimeModel, reorderLevels: Float64Array, paths = 500, seed = 23, warmup = 0): ScenarioSet {
  if (reorderLevels.length !== actual.length) throw new Error("buildReplaySet: one reorder level per day is required");
  if (warmup < 0 || warmup >= actual.length) throw new Error("buildReplaySet: warm-up must be shorter than the demand path");
  const rng = createRng(seed);
  const sampleL = leadTimeSampler(leadTime, rng);
  let sum = 0;
  for (let k = warmup; k < actual.length; k++) sum += actual[k] as number;
  return {
    warmup,
    days: actual.length - warmup,
    demand: Array.from({ length: paths }, () => actual), // shared and read-only
    leadTimes: Array.from({ length: paths }, () => Int32Array.from({ length: MAX_ORDERS_PER_PATH }, () => Math.max(1, Math.round(sampleL())))),
    expectedLeadTimeDemand: reorderLevels,
    meanDemand: sum / (actual.length - warmup),
  };
}

/** Same demand and delivery delays, different reorder-level path (common random numbers). */
export function withReorderLevels(set: ScenarioSet, reorderLevels: Float64Array): ScenarioSet {
  return { ...set, expectedLeadTimeDemand: reorderLevels };
}

export type PolicySpec =
  | { kind: "static"; reorderPoint: number; orderQty: number }
  | { kind: "dynamic"; buffer: number; orderQty: number };

export interface PolicyMetrics {
  costPerDay: number;
  holdingPerDay: number;
  orderingPerDay: number;
  stockoutPerDay: number;
  /** Units served ÷ units demanded. */
  fillRate: number;
  /** Share of replenishment cycles (order placed → order received) without any lost sales. */
  cycleServiceLevel: number;
  /** Share of open days with lost sales. */
  stockoutDayRate: number;
  avgOnHand: number;
  avgInventoryValue: number;
  /** 95th percentile (across paths) of the highest on-hand level reached. */
  peakOnHandP95: number;
  ordersPerWeek: number;
  lostUnitsPerDay: number;
}

export interface PolicyTrace {
  onHand: Float64Array;
  reorderLevel: Float64Array;
  demand: Float64Array;
  lost: Float64Array;
  orders: { day: number; qty: number; arrival: number }[];
}

export function reorderLevel(policy: PolicySpec, set: ScenarioSet, k: number): number {
  return policy.kind === "static" ? policy.reorderPoint : (set.expectedLeadTimeDemand[k] as number) + policy.buffer;
}

/** Per-path results (paths share the same random numbers across policies, so comparisons can be paired). */
export interface PathMetrics {
  fillRate: Float64Array;
  costPerDay: Float64Array;
}

export function simulatePolicy(set: ScenarioSet, policy: PolicySpec, costs: CostParams, tracePath = -1): PolicyMetrics & { trace?: PolicyTrace; paths: PathMetrics } {
  const Q = Math.max(1, policy.orderQty);
  const pathFill = new Float64Array(set.demand.length);
  const pathCost = new Float64Array(set.demand.length);
  const h = dailyHoldingCost(costs);
  const total = set.warmup + set.days;
  const arrivals = new Float64Array(total + 400);
  const peaks: number[] = [];
  let demandSum = 0;
  let soldSum = 0;
  let lostSum = 0;
  let holdingUnits = 0;
  let orders = 0;
  let openDays = 0;
  let stockoutDays = 0;
  let cycles = 0;
  let cleanCycles = 0;
  let trace: PolicyTrace | undefined;

  set.demand.forEach((path, p) => {
    arrivals.fill(0);
    const lostDay = new Uint8Array(total);
    let onHand = reorderLevel(policy, set, 0) + Q; // start just after a delivery
    let onOrder = 0;
    let nextLt = 0;
    let peak = 0;
    let pDemand = 0;
    let pSold = 0;
    let pLost = 0;
    let pHold = 0;
    let pOrders = 0;
    const pending: { day: number; arrival: number }[] = [];
    const tr: PolicyTrace | undefined = p === tracePath
      ? { onHand: new Float64Array(total), reorderLevel: new Float64Array(total), demand: new Float64Array(total), lost: new Float64Array(total), orders: [] }
      : undefined;

    for (let k = 0; k < total; k++) {
      const received = arrivals[k] as number;
      onHand += received;
      onOrder -= received;
      const d = path[k] as number;
      const sold = Math.min(onHand, d);
      const lost = d - sold;
      onHand -= sold;
      if (lost > 1e-9) lostDay[k] = 1;
      if (k >= set.warmup) {
        demandSum += d;
        soldSum += sold;
        lostSum += lost;
        holdingUnits += onHand;
        pDemand += d;
        pSold += sold;
        pLost += lost;
        pHold += onHand;
        if (d > 0) {
          openDays++;
          if (lost > 1e-9) stockoutDays++;
        }
        peak = Math.max(peak, onHand + sold); // stock available at the start of the day
      }
      const level = reorderLevel(policy, set, k);
      const position = onHand + onOrder;
      if (position <= level) {
        const qty = (Math.floor((level - position) / Q) + 1) * Q;
        const lt = (set.leadTimes[p] as Int32Array)[nextLt++ % MAX_ORDERS_PER_PATH] as number;
        arrivals[k + lt] = (arrivals[k + lt] as number) + qty;
        onOrder += qty;
        if (k >= set.warmup) {
          orders++;
          pOrders++;
          if (k + lt < total) pending.push({ day: k, arrival: k + lt });
        }
        tr?.orders.push({ day: k, qty, arrival: k + lt });
      }
      if (tr) {
        tr.onHand[k] = onHand;
        tr.reorderLevel[k] = level;
        tr.demand[k] = d;
        tr.lost[k] = lost;
      }
    }
    for (const c of pending) {
      cycles++;
      let clean = true;
      for (let j = c.day + 1; j < c.arrival; j++) if (lostDay[j]) clean = false;
      if (clean) cleanCycles++;
    }
    peaks.push(peak);
    pathFill[p] = pDemand > 0 ? pSold / pDemand : 1;
    pathCost[p] = (pHold * h + pOrders * costs.orderCost + pLost * costs.stockoutPenalty) / set.days;
    if (tr) trace = tr;
  });

  const pathDays = set.demand.length * set.days;
  const holdingPerDay = (holdingUnits * h) / pathDays;
  const orderingPerDay = (orders * costs.orderCost) / pathDays;
  const stockoutPerDay = (lostSum * costs.stockoutPenalty) / pathDays;
  peaks.sort((a, b) => a - b);
  const avgOnHand = holdingUnits / pathDays;
  return {
    costPerDay: holdingPerDay + orderingPerDay + stockoutPerDay,
    holdingPerDay,
    orderingPerDay,
    stockoutPerDay,
    fillRate: demandSum > 0 ? soldSum / demandSum : 1,
    cycleServiceLevel: cycles ? cleanCycles / cycles : 1,
    stockoutDayRate: openDays ? stockoutDays / openDays : 0,
    avgOnHand,
    avgInventoryValue: avgOnHand * costs.unitCost,
    peakOnHandP95: peaks[Math.min(peaks.length - 1, Math.floor(0.95 * peaks.length))] ?? 0,
    ordersPerWeek: (orders / pathDays) * 7,
    lostUnitsPerDay: lostSum / pathDays,
    trace,
    paths: { fillRate: pathFill, costPerDay: pathCost },
  };
}
