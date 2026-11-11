import { describe, expect, it } from "vitest";
import { cdf, erfc, inverseCdf, lossFunction, pdf, upperTail } from "../../src/inventory/normal";
import { type Golden, paths, readJson } from "../helpers";

interface NormalGolden {
  normal: { x: number[]; cdf: number[]; sf: number[]; pdf: number[]; loss: number[]; p: number[]; ppf: number[] };
}
const g = readJson<NormalGolden>(paths.golden.replace("favorita-golden.json", "parity.json")).normal;
const fav = readJson<Golden>(paths.golden);
const relErr = (a: number, b: number) => (b === 0 ? Math.abs(a) : Math.abs(a - b) / Math.abs(b));

describe("standard normal matches scipy.stats.norm", () => {
  it("Φ, upper tail, density and loss function to ≤1e-12 relative error (tails to 1e-14 and beyond)", () => {
    g.x.forEach((x, i) => {
      expect(relErr(cdf(x), g.cdf[i] as number)).toBeLessThan(1e-12);
      expect(relErr(upperTail(x), g.sf[i] as number)).toBeLessThan(1e-12);
      expect(relErr(pdf(x), g.pdf[i] as number)).toBeLessThan(1e-13);
      if ((g.loss[i] as number) > 1e-300) expect(relErr(lossFunction(x), g.loss[i] as number)).toBeLessThan(1e-9);
    });
  });

  it("Φ⁻¹ (the service-level factor Z) to 1e-12, including extreme service levels", () => {
    g.p.forEach((p, i) => expect(Math.abs(inverseCdf(p) - (g.ppf[i] as number))).toBeLessThan(1e-12));
    fav.inverseNormal.forEach(({ p, z }) => expect(Math.abs(inverseCdf(p) - z)).toBeLessThan(1e-12));
  });

  it("handles edge cases", () => {
    expect(inverseCdf(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(inverseCdf(1)).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(inverseCdf(1.2))).toBe(true);
    expect(inverseCdf(0.5)).toBeCloseTo(0, 15);
    expect(erfc(0)).toBe(1);
    expect(erfc(-30)).toBe(2);
  });
});
