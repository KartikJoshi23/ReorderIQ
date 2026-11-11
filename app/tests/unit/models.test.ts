import { describe, expect, it } from "vitest";
import { DEMO_SERIES_ID, generateDemoPack } from "../../src/data/demo";
import { extendCalendar, initialAsOf, type SeriesData, seriesFromPack } from "../../src/data/series";
import { actualsAfter, foldMetrics, foldOrigins, forecastBaseline } from "../../src/models/backtest";
import { combine, ensembleFolds, inverseWapeWeights } from "../../src/models/ensemble";
import { fitEts, forecastEts } from "../../src/models/ets";
import { buildHistory } from "../../src/models/history";
import { applyIntervals, coverage, fitIntervals } from "../../src/models/intervals";
import { computeMetrics } from "../../src/models/metrics";
import { fitRidge, forecastRidge, rawCoefficient } from "../../src/models/ridge";
import { scoreIssued } from "../../src/models/tracking";

const base = seriesFromPack(generateDemoPack(2026), DEMO_SERIES_ID);
const asOf = initialAsOf(base);
const s = extendCalendar(base, asOf + 1 + 28);
const H = 28;

function wapeAt(origin: number, f: Float64Array) {
  return computeMetrics(actualsAfter(s, origin, H), f, 1).wape;
}

describe("ETS and Ridge on the synthetic demo series", () => {
  const h = buildHistory(s, asOf);

  it("history imputes closed days with last week's value and flags them as unusable", () => {
    const closedIdx = Array.from(s.closed.subarray(0, asOf + 1)).findIndex((c, i) => c === 1 && i > 30);
    expect(h.open[closedIdx]).toBe(0);
    expect(h.y[closedIdx]).toBe(h.y[closedIdx - 7]);
  });

  it("fits sensible smoothing parameters and beats the 28-day moving average", () => {
    const m = fitEts(s, h, asOf);
    for (const v of [m.params.alpha, m.params.beta, m.params.gamma]) expect(v).toBeGreaterThan(0), expect(v).toBeLessThan(1);
    expect(m.params.phi).toBeGreaterThanOrEqual(0.8);
    const f = forecastEts(m, s, H);
    expect(wapeAt(asOf, f)).toBeLessThan(wapeAt(asOf, forecastBaseline("ma28", s, asOf, H)));
  });

  it("Ridge recovers the weekend uplift built into the demo generator and forecasts well", () => {
    const m = fitRidge(s, h, asOf);
    // Demo weekend effects are β_sat ≈ 0.44, β_sun ≈ 0.51 on log scale (conditional on the AR terms here).
    expect(rawCoefficient(m, "dow_sun")).toBeGreaterThan(0.2);
    expect(rawCoefficient(m, "dow_sat")).toBeGreaterThan(0.2);
    const f = forecastRidge(m, s, h, H);
    expect(wapeAt(asOf, f)).toBeLessThan(wapeAt(asOf, forecastBaseline("ma28", s, asOf, H)));
  });

  it("has no look-ahead: changing data after the origin leaves the forecast unchanged", () => {
    const origin = asOf - 56;
    const poisoned: SeriesData = { ...s, sales: s.sales.map((v, i) => (i > origin ? v * 10 : v)) };
    const fA = forecastRidge(fitRidge(s, buildHistory(s, origin), origin), s, buildHistory(s, origin), H);
    const fB = forecastRidge(fitRidge(poisoned, buildHistory(poisoned, origin), origin), poisoned, buildHistory(poisoned, origin), H);
    expect(Array.from(fB)).toEqual(Array.from(fA));
    const eA = forecastEts(fitEts(s, buildHistory(s, origin), origin), s, H);
    const eB = forecastEts(fitEts(poisoned, buildHistory(poisoned, origin), origin), poisoned, H);
    expect(Array.from(eB)).toEqual(Array.from(eA));
  });
});

describe("backtest utilities", () => {
  it("places 6 × 28-day folds before the as-of date", () => {
    const o = foldOrigins(s, asOf);
    expect(o).toEqual([asOf - 168, asOf - 140, asOf - 112, asOf - 84, asOf - 56, asOf - 28]);
  });

  it("seasonal naive repeats the same weekday and skips closures; closed days forecast 0", () => {
    const f = forecastBaseline("snaive", s, asOf, 14);
    expect(f[0]).toBe(s.sales[asOf - 6]);
    expect(f[7]).toBe(s.sales[asOf - 6]);
  });
});

describe("ensemble and intervals", () => {
  it("inverse-WAPE weights favour the more accurate model and sum to 1", () => {
    const w = inverseWapeWeights({ ets: 0.2, ridge: 0.1 });
    expect(w.ridge).toBeCloseTo(2 / 3, 12);
    expect((w.ets ?? 0) + (w.ridge ?? 0)).toBeCloseTo(1, 12);
    expect(Array.from(combine({ ets: Float64Array.from([3]), ridge: Float64Array.from([6]) }, w))).toEqual([5]);
  });

  it("ensemble fold k only uses weights learned from earlier folds", () => {
    const mk = (origin: number, f: number, a: number) => ({ origin, forecast: Float64Array.from([f]), actual: Float64Array.from([a]) });
    const byModel = { ets: [mk(0, 120, 100), mk(1, 100, 100)], ridge: [mk(0, 100, 100), mk(1, 200, 100)] };
    const { folds, weightsByFold } = ensembleFolds(byModel, 1);
    expect(weightsByFold[0]).toEqual({ ets: 0.5, ridge: 0.5 });
    // Fold 1 weights come from fold 0 only: ridge was perfect there (WAPE 0) → essentially all weight.
    expect(weightsByFold[1]?.ridge).toBeCloseTo(1, 6);
    expect(folds[1]?.forecast[0]).toBeCloseTo(200, 4);
    expect(foldMetrics(folds, 1).n).toBe(2);
  });

  it("conformal bands contain the calibration errors at the intended rates", () => {
    const folds = Array.from({ length: 10 }, (_, k) => ({
      origin: k,
      forecast: new Float64Array(28).fill(100),
      actual: Float64Array.from({ length: 28 }, (_, j) => 100 * (1 + 0.2 * Math.sin(k * 28 + j))),
    }));
    const iv = fitIntervals(folds, 28);
    const bands = applyIntervals(new Float64Array(28).fill(100), iv);
    const all = folds.flatMap((f) => Array.from(f.actual));
    const lo = Array.from({ length: all.length }, (_, i) => bands.lo80[i % 28] as number);
    const hi = Array.from({ length: all.length }, (_, i) => bands.hi80[i % 28] as number);
    const c = coverage(all, lo, hi);
    expect(c.rate).toBeGreaterThan(0.75);
    expect(c.rate).toBeLessThan(0.85);
    expect(bands.lo95[0]).toBeLessThanOrEqual(bands.lo80[0] as number);
  });
});

describe("live accuracy tracking", () => {
  it("scores issued forecasts only on days that have since been observed", () => {
    const f = Float64Array.from({ length: 7 }, (_, k) => s.sales[asOf - 7 + 1 + k] as number);
    const scores = scoreIssued([{ version: 1, asOf: asOf - 7, forecast: f, lo80: f.map((v) => v - 1), hi80: f.map((v) => v + 1) }], s, asOf - 4);
    expect(scores[0]?.n).toBeLessThanOrEqual(3);
    expect(scores[0]?.wape).toBe(0);
    expect(scores[0]?.coverage80).toBe(1);
  });
});
