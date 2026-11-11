import { describe, expect, it } from "vitest";
import { leadTimeSampler, parseLeadTimeCsv, sampleGamma } from "../../src/inventory/leadTime";
import { relativeErrors, runMonteCarlo, runScenarios, SCENARIOS } from "../../src/inventory/monteCarlo";
import { leadTimeDemand, slideFormula } from "../../src/inventory/safetyStock";
import { createRng } from "../../src/lib/random";
import { type Golden, paths, readJson } from "../helpers";

const golden = readJson<Golden & { monteCarlo: { normalL: { ssDiff: number }; gammaL: { ssDiff: number; pStockoutAtFormulaRop: number } } }>(paths.golden);
const gam = readJson<{ gamma: { shape: number; scale: number; mean: number; sd: number; p95: number } }>(paths.golden.replace("favorita-golden.json", "parity.json")).gamma;
const { mean: dBar, sd: sigmaD } = golden.baselineWindow;

describe("lead-time models", () => {
  it("Marsaglia–Tsang gamma sampler matches scipy moments (k = 12.25, θ = 4/7)", () => {
    const rng = createRng(9);
    const N = 400_000;
    const x = Float64Array.from({ length: N }, () => sampleGamma(rng, gam.shape, gam.scale));
    const m = x.reduce((a, b) => a + b, 0) / N;
    const sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (N - 1));
    expect(m / gam.mean - 1).toBeLessThan(0.003);
    expect(Math.abs(sd / gam.sd - 1)).toBeLessThan(0.005);
    const p95 = x.sort()[Math.floor(0.95 * N)] as number;
    expect(Math.abs(p95 / gam.p95 - 1)).toBeLessThan(0.005);
    expect(Math.abs(sampleGamma(createRng(1), 0.5, 2))).toBeGreaterThan(0); // shape < 1 branch
  });

  it("truncated-normal and empirical samplers stay positive and respect their inputs", () => {
    const rng = createRng(4);
    const tn = leadTimeSampler({ kind: "normal", mean: 2, sd: 2 }, rng);
    for (let i = 0; i < 1000; i++) expect(tn()).toBeGreaterThan(0);
    const emp = leadTimeSampler({ kind: "empirical", mean: 5, sd: 1, samples: [4, 5, 6] }, rng);
    for (let i = 0; i < 100; i++) expect([4, 5, 6]).toContain(emp());
    expect(leadTimeSampler({ kind: "gamma", mean: 7, sd: 0 }, rng)()).toBe(7);
  });

  it("parses supplier lead-time records (dates or days)", () => {
    const r = parseLeadTimeCsv("order_date,receipt_date\n2017-01-01,2017-01-08\n2017-01-05,2017-01-14\n2017-01-09,bad\n");
    expect(r.samples).toEqual([7, 9]);
    expect(r.mean).toBe(8);
    expect(r.sd).toBeCloseTo(Math.SQRT2, 12);
    expect(r.errors).toHaveLength(1);
    expect(parseLeadTimeCsv("lead_time_days\n6\n8\n").mean).toBe(7);
    expect(parseLeadTimeCsv("supplier,qty\nA,3").errors[0]?.message).toMatch(/order_date/);
  });
});

describe("Monte Carlo reproduces the notebook's cross-check of the slide formula (Store 44 BEVERAGES)", () => {
  const formula = slideFormula({ serviceLevel: 0.95, dBar, sigmaD, leadTime: 7, sigmaL: 2 });
  const base = { demand: { kind: "iid-normal" as const, mean: dBar, sigma: sigmaD }, serviceLevel: 0.95, reorderPoint: formula.reorderPoint, trials: 400_000, seed: 2026 };

  it("under the formula's own assumptions (normal L, i.i.d. normal demand) SS_MC ≈ formula +2% (notebook: +2.1%)", () => {
    const mc = runMonteCarlo({ ...base, leadTime: { kind: "normal", mean: 7, sd: 2 } });
    const diff = mc.safetyStock / formula.safetyStock - 1;
    expect(diff).toBeGreaterThan(golden.monteCarlo.normalL.ssDiff - 0.012);
    expect(diff).toBeLessThan(golden.monteCarlo.normalL.ssDiff + 0.012);
    expect(mc.mean).toBeCloseTo(dBar * 7, -2); // E[DDLT] = D̄·L (to within ±50 units)
  });

  it("with right-skewed Gamma lead times the formula under-protects (notebook: SS +8.3%, stockout 6.2%)", () => {
    const mc = runMonteCarlo({ ...base, leadTime: { kind: "gamma", mean: 7, sd: 2 } });
    const diff = mc.safetyStock / formula.safetyStock - 1;
    expect(diff).toBeGreaterThan(golden.monteCarlo.gammaL.ssDiff - 0.012);
    expect(diff).toBeLessThan(golden.monteCarlo.gammaL.ssDiff + 0.012);
    expect(mc.stockoutProbability).toBeGreaterThan(golden.monteCarlo.gammaL.pStockoutAtFormulaRop - 0.004);
    expect(mc.stockoutProbability).toBeLessThan(golden.monteCarlo.gammaL.pStockoutAtFormulaRop + 0.004);
    expect(mc.skewness).toBeGreaterThan(0.2); // right-skewed
  });
});

describe("Monte Carlo mechanics", () => {
  const forecast = Float64Array.from({ length: 28 }, (_, k) => (k % 7 >= 5 ? 150 : 100));
  const lt = { kind: "gamma" as const, mean: 7, sd: 2 };

  it("with zero forecast errors and a fixed lead time, DDLT equals the forecast lead-time demand exactly", () => {
    const zero = [new Float64Array(28), new Float64Array(28)];
    const mc = runMonteCarlo({ demand: { kind: "forecast-bootstrap", forecast, relErrors: zero }, leadTime: { kind: "gamma", mean: 6.5, sd: 0 }, serviceLevel: 0.95, trials: 200, seed: 1 });
    expect(mc.mean).toBeCloseTo(leadTimeDemand(forecast, 6.5), 9);
    expect(mc.safetyStock).toBeCloseTo(0, 9);
  });

  it("applies each forecast error only to the same weekday (weekday-aligned weekly blocks)", () => {
    // Errors are +100% on weekday position 5 only. With L fixed at 7 days, every simulated cycle
    // contains exactly one such day, so DDLT is constant = lead-time demand + forecast[5].
    const spike = Array.from({ length: 6 }, () => Float64Array.from({ length: 28 }, (_, p) => (p % 7 === 5 ? 1 : 0)));
    const mc = runMonteCarlo({ demand: { kind: "forecast-bootstrap", forecast, relErrors: spike }, leadTime: { kind: "gamma", mean: 7, sd: 0 }, serviceLevel: 0.95, trials: 5_000, seed: 2 });
    expect(mc.quantiles.p50).toBeCloseTo(leadTimeDemand(forecast, 7) + (forecast[5] as number), 9);
    expect(mc.safetyStock).toBeCloseTo(0, 9);
  });

  it("is deterministic per seed; histogram counts every trial; convergence ends at the final estimate", () => {
    const rel = relativeErrors([{ forecast, actual: forecast.map((f, k) => f * (1 + 0.1 * Math.sin(k))) }]);
    const inp = { demand: { kind: "forecast-bootstrap" as const, forecast, relErrors: rel }, leadTime: lt, serviceLevel: 0.95, reorderPoint: 900, orderQuantity: 700, trials: 20_000, seed: 3 };
    const a = runMonteCarlo(inp);
    expect(runMonteCarlo(inp)).toEqual(a);
    expect(a.histogram.counts.reduce((x, y) => x + y, 0)).toBe(20_000);
    expect(a.convergence.at(-1)?.safetyStock).toBeCloseTo(a.safetyStock, 9);
    expect(a.fillRate).toBeGreaterThan(0.9);
    expect(a.leadTime.mean).toBeCloseTo(7, 1);
  });

  it("riskier scenarios need more buffer and raise the stockout risk of a fixed reorder point", () => {
    const res = runScenarios({ demand: { kind: "forecast-normal", forecast, sigma: 15 }, leadTime: lt, serviceLevel: 0.95, reorderPoint: 1000, trials: 30_000, seed: 8 });
    const by = Object.fromEntries(res.map((r) => [r.scenario.id, r.result]));
    expect(res).toHaveLength(SCENARIOS.length);
    for (const id of ["delay", "volatile", "surge", "storm"]) {
      expect(by[id]?.stockoutProbability as number).toBeGreaterThan(by.base?.stockoutProbability as number);
    }
    expect(by.storm?.reorderPoint as number).toBeGreaterThan(by.delay?.reorderPoint as number);
    expect(by.volatile?.leadTime.sd as number).toBeCloseTo(4, 1);
  });

  it("relative errors are actual/forecast − 1, undefined on closed days", () => {
    const r = relativeErrors([{ forecast: Float64Array.from([100, 0, 50]), actual: Float64Array.from([110, Number.NaN, 40]) }]);
    expect(r[0]?.[0]).toBeCloseTo(0.1, 12);
    expect(Number.isNaN(r[0]?.[1] as number)).toBe(true);
    expect(r[0]?.[2]).toBeCloseTo(-0.2, 12);
  });
});
