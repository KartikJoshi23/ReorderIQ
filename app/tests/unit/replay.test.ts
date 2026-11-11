import { describe, expect, it } from "vitest";
import { DEMO_SERIES_ID, generateDemoPack } from "../../src/data/demo";
import { seriesFromPack } from "../../src/data/series";
import { DEFAULT_COSTS } from "../../src/inventory/costs";
import { buildReplaySet, simulatePolicy, withReorderLevels } from "../../src/inventory/policySim";
import { replayPlan, runReplay } from "../../src/inventory/replay";
import { aiPolicy } from "../../src/inventory/safetyStock";
import { DEFAULT_ENGINE_SETTINGS, type ForecastResult, ForecastEngine } from "../../src/models/engine";

const s = seriesFromPack(generateDemoPack(2026), DEMO_SERIES_ID);

describe("replay set", () => {
  it("shares the real demand path and varies only the delivery delays (common random numbers)", () => {
    const actual = Float64Array.from({ length: 21 }, (_, k) => 100 + (k % 7) * 10);
    const set = buildReplaySet(actual, { kind: "gamma", mean: 5, sd: 2 }, new Float64Array(21).fill(500), 50, 1);
    expect(set.demand.every((d) => d === actual)).toBe(true);
    expect(new Set(set.leadTimes.map((l) => l.join(","))).size).toBeGreaterThan(40);
    const other = withReorderLevels(set, new Float64Array(21).fill(900));
    expect(other.leadTimes).toBe(set.leadTimes);
    const r = simulatePolicy(set, { kind: "dynamic", buffer: 0, orderQty: 700 }, DEFAULT_COSTS);
    expect(r.paths.fillRate).toHaveLength(50);
    const mean = r.paths.fillRate.reduce((a, b) => a + b, 0) / 50;
    expect(mean).toBeCloseTo(r.fillRate, 2);
  });

  it("keeps the artificial starting stock out of the scored days (warm-up)", () => {
    // 100 units/day, fixed 5-day lead time, static R = 1,000, Q = 700. The simulation starts at R + Q = 1,700;
    // a running policy peaks at R − 5·100 + Q + 100 (stock at the start of a delivery day) = 1,300.
    const actual = new Float64Array(35).fill(100);
    const levels = new Float64Array(35).fill(500);
    const lt = { kind: "gamma" as const, mean: 5, sd: 0 };
    const policy = { kind: "static" as const, reorderPoint: 1000, orderQty: 700 };
    const cold = simulatePolicy(buildReplaySet(actual, lt, levels, 4, 1, 0), policy, DEFAULT_COSTS);
    const warm = simulatePolicy(buildReplaySet(actual, lt, levels, 4, 1, 14), policy, DEFAULT_COSTS);
    expect(cold.peakOnHandP95).toBe(1700);
    expect(warm.peakOnHandP95).toBe(1300);
    expect(warm.fillRate).toBe(1);
    expect(buildReplaySet(actual, lt, levels, 4, 1, 14).days).toBe(21);
    expect(() => buildReplaySet(actual, lt, levels.subarray(1), 4, 1, 0)).toThrow();
  });
});

describe("baseline vs AI replay on held-back demand", () => {
  it(
    "replays every held-back week with weekly re-planned AI reorder levels and paired comparisons",
    async () => {
      const plan = replayPlan(s);
      expect(plan.end - plan.start + 1).toBe(56);
      expect(plan.weekOrigins).toHaveLength(8);
      const engine = new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, useLstm: false });
      const weekly: ForecastResult[] = [];
      for (const o of plan.weekOrigins) weekly.push(await engine.run(s, o));
      const r = runReplay({ series: s, weekly, serviceLevel: 0.95, leadTime: { kind: "gamma", mean: 7, sd: 2 }, costs: DEFAULT_COSTS, optimised: { buffer: 60_000, orderQty: 60_000 }, paths: 200 });
      expect(r.dates).toHaveLength(56);
      expect(r.weeks).toBe(8);
      expect(r.warmupDays).toBe(28);
      expect(r.actual[0]).toBe(s.sales[plan.start]); // the scored window starts at the held-back day
      const base = r.policies.baseline;
      const ai = r.policies.aiFormula;
      expect(base?.reorderLevel.every((v) => v === base.reorderLevel[0])).toBe(true); // static
      expect(new Set(Array.from(ai?.reorderLevel ?? [])).size).toBeGreaterThan(20); // moves day by day
      // Day 8 (index 7) is in replay week 2: its AI level must come from the forecast issued at week 2's
      // origin, which only knew sales up to that origin (no look-ahead).
      const p = { serviceLevel: 0.95, leadTime: { kind: "gamma" as const, mean: 7, sd: 2 } };
      expect(ai?.reorderLevel[7]).toBeCloseTo(aiPolicy(weekly[1] as ForecastResult, p, 1).reorderPoint, 9);
      expect(ai?.reorderLevel[6]).toBeCloseTo(aiPolicy(weekly[0] as ForecastResult, p, 7).reorderPoint, 9);
      expect(weekly[1]?.asOf).toBe(plan.weekOrigins[1]);
      for (const p of [base, ai, r.policies.optimised]) {
        expect(p?.metrics.fillRate).toBeGreaterThan(0.5);
        expect(p?.metrics.fillRate).toBeLessThanOrEqual(1);
        expect(p?.trace.onHand).toHaveLength(56);
      }
      expect(r.paired.aiFillAtLeastBaseline).toBeGreaterThanOrEqual(0);
      expect(r.paired.aiFillAtLeastBaseline).toBeLessThanOrEqual(1);
      expect(r.paired.optimisedCheaperThanBaseline).not.toBeNull();
    },
    120_000,
  );
});
