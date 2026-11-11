/** Python ↔ TypeScript parity on synthetic data (scripts/make_parity_golden.py). */
import { describe, expect, it } from "vitest";
import { etsFilter, etsForecastLog } from "../../src/models/ets";
import { computeMetrics, quantile } from "../../src/models/metrics";
import { ridgePredictRow, ridgeSolve } from "../../src/models/ridge";
import { paths, readJson } from "../helpers";

interface ParityGolden {
  ets: { y: number[]; params: { alpha: number; beta: number; gamma: number; phi: number }; init: { level: number; trend: number; season: number[] }; fitted: number[]; finalLevel: number; finalTrend: number; forecast: number[]; sse: number };
  ridge: { X: number[][]; y: number[]; lambda: number; coef: number[]; intercept: number; predFirst: number };
  metrics: { actual: number[]; forecast: number[]; mae: number; rmse: number; wape: number; bias: number };
  quantile: { values: number[]; q: number[]; expected: number[] };
}

const g = readJson<ParityGolden>(paths.golden.replace("favorita-golden.json", "parity.json"));

describe("Holt-Winters matches statsmodels ExponentialSmoothing (damped add trend, add weekly season)", () => {
  const n = g.ets.y.length;
  const run = etsFilter(g.ets.y, new Array<number>(n).fill(1), (i) => i % 7, 0, n - 1, g.ets.params, g.ets.init);

  it("one-step fitted values and SSE", () => {
    g.ets.fitted.forEach((v, i) => expect(run.fitted[i]).toBeCloseTo(v, 10));
    expect(run.sse).toBeCloseTo(g.ets.sse, 9);
  });

  it("final level/trend and 14-day forecast", () => {
    expect(run.level).toBeCloseTo(g.ets.finalLevel, 10);
    expect(run.trend).toBeCloseTo(g.ets.finalTrend, 10);
    const f = etsForecastLog({ params: g.ets.params, level: run.level, trend: run.trend, season: run.season }, (k) => (n - 1 + k) % 7, 14);
    // Textbook forecast (Hyndman & Athanasopoulos, FPP3 §8.3) uses the latest seasonal estimate s_T
    // for h = 7, 14. statsmodels uses the previous week's estimate s_{T−7} at exact multiples of the
    // period, so there the two differ by exactly s_T − s_{T−7}; every other step must match.
    const before = etsFilter(g.ets.y, new Array<number>(n).fill(1), (i) => i % 7, 0, n - 2, g.ets.params, g.ets.init);
    const lastPos = (n - 1) % 7;
    const staleGap = (run.season[lastPos] as number) - (before.season[lastPos] as number);
    g.ets.forecast.forEach((v, k) => {
      const h = k + 1;
      expect(f[k]).toBeCloseTo(h % 7 === 0 ? v + staleGap : v, 10);
    });
    expect(Math.abs(staleGap)).toBeGreaterThan(1e-3);
  });
});

describe("Ridge matches scikit-learn Ridge on standardised inputs", () => {
  it("coefficients, intercept and prediction", () => {
    const rows = g.ridge.X.length;
    const cols = (g.ridge.X[0] as number[]).length;
    const fit = ridgeSolve(Float64Array.from(g.ridge.X.flat()), Float64Array.from(g.ridge.y), rows, cols, g.ridge.lambda);
    g.ridge.coef.forEach((c, j) => expect(fit.beta[j]).toBeCloseTo(c, 9));
    expect(fit.intercept).toBeCloseTo(g.ridge.intercept, 10);
    expect(ridgePredictRow(fit, g.ridge.X[0] as number[])).toBeCloseTo(g.ridge.predFirst, 9);
  });
});

describe("metrics and quantiles match scikit-learn / numpy", () => {
  it("MAE, RMSE, WAPE, bias", () => {
    const m = computeMetrics(g.metrics.actual, g.metrics.forecast, 1);
    expect(m.mae).toBeCloseTo(g.metrics.mae, 10);
    expect(m.rmse).toBeCloseTo(g.metrics.rmse, 10);
    expect(m.wape).toBeCloseTo(g.metrics.wape, 12);
    expect(m.bias).toBeCloseTo(g.metrics.bias, 12);
    expect(m.mase).toBeCloseTo(g.metrics.mae, 10);
  });

  it("linear-interpolation quantiles (numpy default)", () => {
    g.quantile.q.forEach((q, i) => expect(quantile(g.quantile.values, q)).toBeCloseTo(g.quantile.expected[i] as number, 10));
  });
});
