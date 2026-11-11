import { describe, expect, it } from "vitest";
import { DEFAULT_COSTS, defaultConstraints, economicOrderQuantity } from "../../src/inventory/costs";
import { buildScenarioSet, simulatePolicy } from "../../src/inventory/policySim";
import { createRng } from "../../src/lib/random";
import { bayesianOptimize, expectedImprovement, haltonPoints } from "../../src/optimize/bayesOpt";
import { fitGp, fitGpAuto, predictGp } from "../../src/optimize/gp";
import { capacityShadowPrice, gridSearch, optimizeInventory, ropHeatmap, tornado } from "../../src/optimize/inventoryOpt";
import { paths, readJson } from "../helpers";

interface GpGolden {
  gp: { X: number[][]; y: number[]; hyper: { lengthscales: number[]; signalVar: number; noiseVar: number }; Xs: number[][]; mean: number[]; sdWithNoise: number[]; lml: number };
}
const g = readJson<GpGolden>(paths.golden.replace("favorita-golden.json", "parity.json")).gp;

describe("Gaussian process matches scikit-learn (Matérn 5/2 ARD + white noise, fixed hyperparameters)", () => {
  const m = fitGp(g.X, g.y, g.hyper);
  it("posterior mean, posterior SD and log marginal likelihood", () => {
    g.Xs.forEach((x, i) => {
      const p = predictGp(m, x);
      expect(p.mean).toBeCloseTo(g.mean[i] as number, 9);
      expect(Math.sqrt(p.variance + g.hyper.noiseVar)).toBeCloseTo(g.sdWithNoise[i] as number, 9);
    });
    expect(m.logMarginalLikelihood).toBeCloseTo(g.lml, 9);
  });

  it("automatic hyperparameters interpolate the data and grow uncertain away from it", () => {
    const auto = fitGpAuto(g.X, g.y);
    const atData = predictGp(auto, g.X[0] as number[]);
    expect(Math.abs(atData.mean - (g.y[0] as number))).toBeLessThan(0.15);
    expect(predictGp(auto, [3, 3]).sd).toBeGreaterThan(atData.sd);
  });
});

describe("Bayesian optimisation", () => {
  it("expected improvement is zero far above the incumbent and positive below it", () => {
    expect(expectedImprovement(10, 0.01, 0)).toBeCloseTo(0, 12);
    expect(expectedImprovement(-1, 0.5, 0)).toBeGreaterThan(1);
  });

  it("Halton design fills the unit square", () => {
    const pts = haltonPoints(64, 2, createRng(1));
    for (const q of [[0, 0.5], [0.5, 1]]) for (const r of [[0, 0.5], [0.5, 1]])
      expect(pts.filter((p) => (p[0] as number) >= (q[0] as number) && (p[0] as number) < (q[1] as number) && (p[1] as number) >= (r[0] as number) && (p[1] as number) < (r[1] as number)).length).toBeGreaterThan(10);
  });

  it("finds the Branin global minimum (0.3979) within 38 evaluations", () => {
    const branin = ([x1, x2]: number[]) => {
      const a = x1 as number;
      const b = x2 as number;
      return (b - (5.1 / (4 * Math.PI ** 2)) * a * a + (5 / Math.PI) * a - 6) ** 2 + 10 * (1 - 1 / (8 * Math.PI)) * Math.cos(a) + 10;
    };
    const r = bayesianOptimize({ bounds: [[-5, 10], [0, 15]], evaluate: (x) => ({ objective: branin(x), feasibility: 1 }) }, { initPoints: 8, iterations: 30, seed: 3 });
    expect(r.evaluations).toBe(38);
    expect(r.best?.objective).toBeLessThan(0.3979 + 0.05);
    expect(r.history.at(-1)?.bestObjective).toBe(r.best?.objective);
  }, 60_000);

  it("respects constraints: min (x−0.8)²+(y−0.8)² s.t. x + y ≤ 1 → (0.5, 0.5), value 0.18", () => {
    const r = bayesianOptimize(
      { bounds: [[0, 1], [0, 1]], evaluate: ([x, y]) => ({ objective: ((x as number) - 0.8) ** 2 + ((y as number) - 0.8) ** 2, feasibility: 1 - (x as number) - (y as number) }) },
      { initPoints: 6, iterations: 24, seed: 5 },
    );
    expect((r.best?.x[0] as number) + (r.best?.x[1] as number)).toBeLessThanOrEqual(1 + 1e-12);
    expect(r.best?.objective).toBeLessThan(0.2);
    expect(r.surface?.mean).toHaveLength(40);
  }, 60_000);
});

describe("inventory policy simulation", () => {
  const forecast = Float64Array.from({ length: 28 }, (_, k) => (k % 7 >= 5 ? 1500 : 1000));
  const rng = createRng(11);
  const relErrors = Array.from({ length: 6 }, () => Float64Array.from({ length: 28 }, () => 0.15 * rng.normal()));
  const set = buildScenarioSet({ forecast, relErrors, leadTime: { kind: "gamma", mean: 7, sd: 2 }, paths: 48, seed: 21 });
  const settings = { mode: "dynamic" as const, costs: DEFAULT_COSTS, constraints: defaultConstraints(set.meanDemand), referenceSafetyStock: 3500 };

  it("conserves stock: start + receipts − sales = end, and sales + lost = demand, on every day", () => {
    const r = simulatePolicy(set, { kind: "dynamic", buffer: 2000, orderQty: 6000 }, DEFAULT_COSTS, 0);
    const tr = r.trace;
    expect(tr).toBeDefined();
    let onHand = (tr?.reorderLevel[0] ?? 0) + 6000;
    const arrivals = new Map<number, number>();
    for (const o of tr?.orders ?? []) arrivals.set(o.arrival, (arrivals.get(o.arrival) ?? 0) + o.qty);
    for (let k = 0; k < (tr?.onHand.length ?? 0); k++) {
      onHand += arrivals.get(k) ?? 0;
      const sold = Math.min(onHand, tr?.demand[k] as number);
      expect(sold + (tr?.lost[k] as number)).toBeCloseTo(tr?.demand[k] as number, 9);
      onHand -= sold;
      expect(tr?.onHand[k]).toBeCloseTo(onHand, 6);
    }
    expect(r.fillRate).toBeGreaterThan(0);
    expect(r.fillRate).toBeLessThanOrEqual(1);
    expect(r.cycleServiceLevel).toBeLessThanOrEqual(1);
  });

  it("with common random numbers, a larger buffer never loses more sales and holds more stock", () => {
    const small = simulatePolicy(set, { kind: "dynamic", buffer: 500, orderQty: 6000 }, DEFAULT_COSTS);
    const large = simulatePolicy(set, { kind: "dynamic", buffer: 4000, orderQty: 6000 }, DEFAULT_COSTS);
    expect(large.lostUnitsPerDay).toBeLessThanOrEqual(small.lostUnitsPerDay);
    expect(large.avgOnHand).toBeGreaterThan(small.avgOnHand);
    expect(large.fillRate).toBeGreaterThanOrEqual(small.fillRate);
  });

  it("EOQ = √(2DS/H)", () => {
    expect(economicOrderQuantity(1000, DEFAULT_COSTS)).toBeCloseTo(Math.sqrt((2 * 365_000 * 120) / (0.7 * 0.25)), 9);
  });

  it("Bayesian optimisation reaches the 900-point grid optimum within 2% using 38 evaluations", () => {
    const bo = optimizeInventory(set, settings, { seed: 9 });
    const grid = gridSearch(set, settings, 30);
    expect(bo.best).not.toBeNull();
    expect(grid.best).not.toBeNull();
    expect(bo.best?.objective as number).toBeLessThan((grid.best?.objective as number) * 1.02);
    expect(bo.best?.details?.metrics.fillRate).toBeGreaterThanOrEqual(settings.constraints.minFillRate);
    expect(bo.evaluations).toBe(38);
    const shadow = capacityShadowPrice(set, settings, bo.best?.objective ?? null, { seed: 9 });
    expect(shadow.extraCapacity).toBeGreaterThan(0);
  }, 120_000);
});

describe("formula sensitivity views", () => {
  const base = { serviceLevel: 0.95, dBar: 10_865, sigmaD: 3_587, leadTime: 7, sigmaL: 2 };
  it("tornado ranks inputs by reorder-point swing; demand and lead time dominate", () => {
    const t = tornado(base);
    expect(t.bars).toHaveLength(5);
    expect(t.bars[0]?.swing).toBeGreaterThanOrEqual(t.bars[1]?.swing as number);
    expect(["dBar", "leadTime"]).toContain(t.bars[0]?.input);
    expect(t.bars.find((b) => b.input === "sigmaL")?.swing).toBeGreaterThan(t.bars.find((b) => b.input === "sigmaD")?.swing as number);
  });
  it("heatmap rises with service level and with lead-time variability", () => {
    const h = ropHeatmap(base, [0.9, 0.95, 0.99], [1, 2, 3]);
    expect(h[0]?.[2]).toBeGreaterThan(h[0]?.[0] as number);
    expect(h[2]?.[1]).toBeGreaterThan(h[0]?.[1] as number);
  });
});
