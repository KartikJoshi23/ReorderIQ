import { describe, expect, it } from "vitest";
import { DEMO_SERIES_ID, generateDemoPack } from "../../src/data/demo";
import { initialAsOf, seriesFromPack } from "../../src/data/series";
import { openWindowStats } from "../../src/data/stats";
import { aiPolicy, baselinePolicy, dynamicPath, forecastAt, forecastErrorRmse, leadTimeDemand, slideFormula } from "../../src/inventory/safetyStock";
import type { ForecastResult } from "../../src/models/engine";
import { type Golden, paths, readJson } from "../helpers";

const golden = readJson<Golden>(paths.golden);
const { mean: dBar, sd: sigmaD } = golden.baselineWindow;

describe("slide formula SS = Z·√(σ_D²·L + D̄²·σ_L²) reproduces the notebook hand calculation", () => {
  it.each(golden.formula)("L=$L σ_L=$sigmaL SL=$serviceLevel", (row) => {
    const r = slideFormula({ serviceLevel: row.serviceLevel, dBar, sigmaD, leadTime: row.L, sigmaL: row.sigmaL });
    expect(r.z).toBeCloseTo(row.z, 12);
    expect(r.termDemand).toBeCloseTo(row.termDemand, 4);
    expect(r.termLeadTime).toBeCloseTo(row.termLeadTime, 4);
    expect(r.safetyStock).toBeCloseTo(row.ss, 6);
    expect(r.reorderPoint).toBeCloseTo(row.rop, 6);
  });

  it("matches the worked example: SS = 39,004.15, ROP = 115,061.63 (L = 7, σ_L = 2, 95%)", () => {
    const r = slideFormula({ serviceLevel: 0.95, dBar, sigmaD, leadTime: 7, sigmaL: 2 });
    expect(r.safetyStock).toBeCloseTo(39004.15, 2);
    expect(r.reorderPoint).toBeCloseTo(115061.63, 2);
    expect(r.leadTimeShare).toBeCloseTo(0.84, 2);
    expect(r.daysOfCover).toBeCloseTo(3.59, 2);
  });

  it("reduces to the textbook special cases and validates inputs", () => {
    const noLtVar = slideFormula({ serviceLevel: 0.95, dBar: 100, sigmaD: 20, leadTime: 4, sigmaL: 0 });
    expect(noLtVar.safetyStock).toBeCloseTo(1.6448536269514722 * 20 * 2, 10); // Z σ_D √L
    const noDemandVar = slideFormula({ serviceLevel: 0.95, dBar: 100, sigmaD: 0, leadTime: 4, sigmaL: 1 });
    expect(noDemandVar.safetyStock).toBeCloseTo(1.6448536269514722 * 100, 10); // Z D̄ σ_L
    expect(slideFormula({ serviceLevel: 0.5, dBar: 100, sigmaD: 20, leadTime: 4, sigmaL: 1 }).safetyStock).toBeCloseTo(0, 10);
    expect(() => slideFormula({ serviceLevel: 1, dBar: 1, sigmaD: 1, leadTime: 1, sigmaL: 1 })).toThrow(RangeError);
    expect(() => slideFormula({ serviceLevel: 0.9, dBar: 1, sigmaD: 1, leadTime: 0, sigmaL: 1 })).toThrow(RangeError);
  });
});

describe("baseline and AI inputs", () => {
  const s = seriesFromPack(generateDemoPack(2026), DEMO_SERIES_ID);
  const asOf = initialAsOf(s);
  const p = { serviceLevel: 0.95, leadTime: { kind: "gamma" as const, mean: 7, sd: 2 } };

  it("baseline uses trailing-365-day statistics", () => {
    const b = baselinePolicy(s, asOf, p);
    const w = openWindowStats(s, asOf, 365);
    expect(b.dBar).toBe(w.mean);
    expect(b.sigmaD).toBe(w.sd);
    expect(b.method).toBe("baseline");
  });

  it("expected lead-time demand handles fractional lead times and horizons beyond the forecast", () => {
    const f = Float64Array.from([10, 20, 30, 40, 50, 60, 70]);
    expect(leadTimeDemand(f, 3)).toBe(60);
    expect(leadTimeDemand(f, 2.5)).toBe(10 + 20 + 0.5 * 30);
    expect(leadTimeDemand(f, 2, 3)).toBe(40 + 50);
    expect(forecastAt(f, 9)).toBe(30); // repeats the last forecast week
  });

  it("σ_e is the RMSE of finite out-of-sample errors over the lead-time horizons", () => {
    const e = [Float64Array.from([3, -4, Number.NaN, 100]), Float64Array.from([0, 0, 5, 100])];
    const r = forecastErrorRmse(e, 1, 3);
    expect(r.n).toBe(5);
    expect(r.rmse).toBeCloseTo(Math.sqrt((9 + 16 + 25) / 5), 12);
  });

  it("AI policy feeds forecast demand and forecast-error RMSE into the same formula", () => {
    const forecast = Float64Array.from({ length: 28 }, (_, k) => (k % 7 >= 5 ? 150 : 100));
    const foldErrors = Array.from({ length: 6 }, (_, j) => Float64Array.from({ length: 28 }, (_, k) => ((j + k) % 2 ? 10 : -10)));
    const result = { ensemble: { forecast }, foldErrors, horizon: 28, dates: Array.from({ length: 28 }, (_, k) => `d${k}`) } as unknown as ForecastResult;
    const ai = aiPolicy(result, p);
    expect(ai.leadTimeDemand).toBeCloseTo(leadTimeDemand(forecast, 7), 10);
    expect(ai.sigmaD).toBeCloseTo(10, 12);
    expect(ai.errorCount).toBe(42);
    expect(ai.safetyStock).toBeCloseTo(slideFormula({ serviceLevel: 0.95, dBar: ai.dBar, sigmaD: 10, leadTime: 7, sigmaL: 2 }).safetyStock, 10);
    const path = dynamicPath(result, p);
    expect(path.reorderPoint).toHaveLength(28 - 7 + 1);
    expect(path.dates[0]).toBe("d0");
  });
});
