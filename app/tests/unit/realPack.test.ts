/**
 * Real-data parity checks against the notebook's golden values. They run locally where the
 * git-ignored build output exists and are skipped elsewhere.
 */
import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { PackSchema } from "../../src/data/schema";
import { indexOfDate, seriesFromPack } from "../../src/data/series";
import { openWindowStats } from "../../src/data/stats";
import { type Golden, paths, readJson, realPlainPack } from "../helpers";

const plain = realPlainPack();
const golden = readJson<Golden>(paths.golden);

describe.skipIf(!plain)("real Favorita pack (local only)", () => {
  const pack = PackSchema.parse(plain);
  const s = seriesFromPack(pack, golden.series);

  it("baseline D̄ and σ_D match the notebook (trailing 365 days, open days, ddof = 1)", () => {
    const st = openWindowStats(s, indexOfDate(s, golden.baselineWindow.end), 365);
    expect(st.n).toBe(golden.baselineWindow.n);
    expect(st.mean).toBeCloseTo(golden.baselineWindow.mean, 8);
    expect(st.sd).toBeCloseTo(golden.baselineWindow.sd, 6);
  });

  it("calendar flags and promotion days match the notebook", () => {
    const sum = (a: Uint8Array) => a.reduce((x, y) => x + y, 0);
    expect(s.length).toBe(golden.calendar.days);
    expect(sum(s.holiday)).toBe(golden.calendar.store44.holiday);
    expect(sum(s.event)).toBe(golden.calendar.store44.event);
    expect(sum(s.quake)).toBe(golden.calendar.store44.quake);
    expect(sum(s.workDay)).toBe(golden.calendar.store44.work_day);
    expect(sum(s.closed)).toBe(golden.calendar.store44.closed);
    let promoDays = 0;
    for (let i = s.modelStart; i <= s.trainEnd; i++) if ((s.promo[i] as number) > 0) promoDays++;
    expect(promoDays).toBe(golden.promoDaysSinceModelStart);
  });

  it.skipIf(!existsSync(paths.publicPack))("the deployed data file is schema-valid and identical to the build output", () => {
    const deployed = PackSchema.parse(readJson(paths.publicPack));
    expect(deployed).toEqual(pack);
  });
});
