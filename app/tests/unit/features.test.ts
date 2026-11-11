import { describe, expect, it } from "vitest";
import { DEFAULT_FEATURES, designMatrix, featureValue } from "../../src/data/features";
import type { Pack } from "../../src/data/schema";
import { indexOfDate, seriesFromPack } from "../../src/data/series";
import { type Golden, paths, readJson } from "../helpers";

const s = seriesFromPack(readJson<Pack>(paths.fixtureJson), "1-TEST"); // 2016-12-20 .. 2017-01-09
const golden = readJson<Golden>(paths.golden);
const at = (iso: string) => indexOfDate(s, iso);

describe("calendar features (match scripts/favorita_data.py)", () => {
  it("payday = 15th or month-end; Christmas ramp = day/24 in 1–24 Dec", () => {
    expect(featureValue("payday", s, at("2016-12-31"))).toBe(1);
    expect(featureValue("payday", s, at("2016-12-30"))).toBe(0);
    expect(featureValue("xmas", s, at("2016-12-24"))).toBe(golden.spotChecks.xmasRamp["2016-12-24"]);
    expect(featureValue("xmas", s, at("2016-12-26"))).toBe(golden.spotChecks.xmasRamp["2016-12-26"]);
    expect(featureValue("xmas", s, at("2016-12-20"))).toBeCloseTo(20 / 24, 12);
  });

  it("weekday dummies use Monday as reference", () => {
    const mon = at("2016-12-26");
    for (const d of ["tue", "wed", "thu", "fri", "sat", "sun"]) expect(featureValue(`dow_${d}`, s, mon)).toBe(0);
    expect(featureValue("dow_sun", s, at("2017-01-01"))).toBe(1);
  });

  it("yearly Fourier terms use day-of-year / 365.25", () => {
    const i = at("2017-01-01"); // day-of-year 1
    expect(featureValue("sin1", s, i)).toBeCloseTo(Math.sin((2 * Math.PI) / 365.25), 12);
    expect(featureValue("cos3", s, i)).toBeCloseTo(Math.cos((2 * Math.PI * 3) / 365.25), 12);
  });

  it("trend is years since the modelling start; promotions enter as log(1 + count)", () => {
    expect(featureValue("trend", s, s.modelStart)).toBe(0);
    expect(featureValue("trend", s, s.modelStart + 365)).toBeCloseTo(365 / 365.25, 12);
    expect(featureValue("log_promo", s, 3)).toBeCloseTo(Math.log1p(s.promo[3] as number), 12);
  });

  it("oil_lag7 uses only the previous 7 days (no same-day information)", () => {
    const i = 10;
    let sum = 0;
    for (let j = 3; j < 10; j++) sum += s.oil[j] as number;
    expect(featureValue("oil_lag7", s, i)).toBeCloseTo(sum / 7, 12);
  });

  it("builds a row-major design matrix", () => {
    const d = designMatrix(s, [0, 1, 2]);
    expect(d.cols).toBe(DEFAULT_FEATURES.length);
    expect(d.data.length).toBe(3 * d.cols);
    expect(d.data[1 * d.cols + d.names.indexOf("holiday")]).toBe(s.holiday[1]);
    expect(() => designMatrix(s, [0], ["nope"])).toThrow(/Unknown feature/);
  });
});
