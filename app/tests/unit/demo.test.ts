import { describe, expect, it } from "vitest";
import params from "../../src/data/demo-params.json";
import { DEMO_SERIES_ID, generateDemoPack } from "../../src/data/demo";
import { PackSchema } from "../../src/data/schema";
import { dateAt, lastObservedIndex, seriesFromPack } from "../../src/data/series";
import { createRng } from "../../src/lib/random";

describe("synthetic demo pack", () => {
  const pack = generateDemoPack(2026);
  const s = seriesFromPack(pack, DEMO_SERIES_ID);

  it("is schema-valid, labelled synthetic and deterministic for a seed", () => {
    expect(PackSchema.safeParse(pack).success).toBe(true);
    expect(pack.meta.synthetic).toBe(true);
    expect(s.source).toBe("demo");
    expect(generateDemoPack(2026)).toEqual(pack);
    expect(generateDemoPack(7).series[0]?.sales).not.toEqual(pack.series[0]?.sales);
  });

  it("mirrors the real pack's calendar offsets (model start, 8-week replay, 16-day horizon)", () => {
    expect(s.length).toBe(1704);
    expect(s.trainEnd - s.modelStart).toBe(1687 - 455);
    expect(s.trainEnd - s.replayStart + 1).toBe(56);
    expect(lastObservedIndex(s)).toBe(s.trainEnd);
    expect(dateAt(s, s.trainEnd)).toBe("2025-08-15");
  });

  it("reproduces the estimated weekly pattern and level", () => {
    const sums = new Array<number>(7).fill(0);
    const counts = new Array<number>(7).fill(0);
    let logSum = 0;
    let n = 0;
    for (let i = s.modelStart; i <= s.trainEnd; i++) {
      if (s.closed[i]) continue;
      const v = s.sales[i] as number;
      const dow = (((s.epoch0 + i + 3) % 7) + 7) % 7; // 1970-01-01 was a Thursday (dow 3)
      sums[dow] = (sums[dow] as number) + v;
      counts[dow] = (counts[dow] as number) + 1;
      logSum += Math.log1p(v);
      n++;
    }
    const means = sums.map((x, d) => x / (counts[d] as number));
    // Sunday/Monday ratio ≈ exp(β_sun) from the fitted coefficients (allow simulation noise)
    expect(Math.log((means[6] as number) / (means[0] as number))).toBeCloseTo(params.coefficients.dow_sun, 1);
    expect(logSum / n).toBeGreaterThan(8.5);
    expect(logSum / n).toBeLessThan(9.6);
  });
});

describe("seeded RNG", () => {
  it("is reproducible and has standard-normal moments", () => {
    const a = createRng(1);
    const b = createRng(1);
    expect([a.next(), a.next()]).toEqual([b.next(), b.next()]);
    const r = createRng(42);
    let m = 0;
    let m2 = 0;
    const N = 200_000;
    for (let i = 0; i < N; i++) {
      const z = r.normal();
      m += z;
      m2 += z * z;
    }
    expect(m / N).toBeCloseTo(0, 2);
    expect(m2 / N).toBeCloseTo(1, 2);
    const u = createRng(3);
    for (let i = 0; i < 1000; i++) {
      const x = u.next();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});
