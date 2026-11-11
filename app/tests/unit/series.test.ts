import { describe, expect, it } from "vitest";
import type { Pack } from "../../src/data/schema";
import {
  applyObservations,
  dateAt,
  initialAsOf,
  lastObservedIndex,
  listSeries,
  ObservationError,
  ORIGIN_ADDED,
  releaseDays,
  seriesFromPack,
  titleCase,
} from "../../src/data/series";
import { paths, readJson } from "../helpers";

const fixture = () => readJson<Pack>(paths.fixtureJson);

describe("seriesFromPack", () => {
  it("maps pack arrays to typed arrays with NaN for unobserved days", () => {
    const s = seriesFromPack(fixture(), "1-TEST");
    expect(s.length).toBe(21);
    expect(dateAt(s, 0)).toBe("2016-12-20");
    expect(dateAt(s, s.trainEnd)).toBe("2017-01-05");
    expect(Number.isNaN(s.sales[17])).toBe(true);
    expect(s.closed[5]).toBe(1); // 25 Dec
    expect(s.names.get(12)).toBe("Primer dia del ano");
    expect(lastObservedIndex(s)).toBe(16);
    expect(s.source).toBe("real");
  });

  it("lists series with readable labels", () => {
    expect(listSeries(fixture())[0]).toMatchObject({ id: "1-TEST", label: "Store 1 · Test", city: "Testville" });
    expect(titleCase("LIQUOR,WINE,BEER")).toBe("Liquor,Wine,Beer");
    expect(titleCase("PERSONAL CARE")).toBe("Personal Care");
  });
});

describe("replay window", () => {
  it("starts with the replay window hidden and releases days up to the last observation", () => {
    const s = seriesFromPack(fixture(), "1-TEST");
    const asOf = initialAsOf(s);
    expect(dateAt(s, asOf)).toBe("2016-12-31");
    expect(releaseDays(s, asOf, 3)).toBe(asOf + 3);
    expect(releaseDays(s, asOf, 999)).toBe(lastObservedIndex(s));
  });
});

describe("applyObservations (adding new data)", () => {
  const base = () => seriesFromPack(fixture(), "1-TEST");

  it("appends contiguous new days and extends the calendar past the pack end", () => {
    const s = base();
    const last = lastObservedIndex(s); // 2017-01-05
    const rows = [];
    for (let k = 1; k <= 20; k++) rows.push({ date: dateAt(s, last + k), sales: 100 + k, promo: 2 });
    const r = applyObservations(s, last, rows);
    expect(r.added).toBe(20);
    expect(r.asOf).toBe(last + 20);
    expect(r.extendedBy).toBe(last + 21 - s.length);
    expect(r.series.sales[last + 20]).toBe(120);
    expect(r.series.origin[last + 1]).toBe(ORIGIN_ADDED);
    expect(r.series.promo[last + 20]).toBe(2);
    expect(Number.isNaN(s.sales[last + 1])).toBe(true); // input not mutated
  });

  it("treats past dates as corrections and re-opens a closed day with sales", () => {
    const s = base();
    const r = applyObservations(s, 10, [{ date: dateAt(s, 5), sales: 7 }]);
    expect(r.revised).toBe(1);
    expect(r.asOf).toBe(10);
    expect(r.series.sales[5]).toBe(7);
    expect(r.series.closed[5]).toBe(0);
  });

  it("rejects gaps, duplicates, negative and fractional units with clear messages", () => {
    const s = base();
    const run = (rows: { date: string; sales: number; promo?: number }[]) => {
      try {
        applyObservations(s, 10, rows);
        return [];
      } catch (e) {
        expect(e).toBeInstanceOf(ObservationError);
        return (e as ObservationError).problems;
      }
    };
    expect(run([{ date: dateAt(s, 12), sales: 5 }])[0]).toMatch(/Missing data for 2016-12-31/);
    expect(run([{ date: dateAt(s, 11), sales: 5 }, { date: dateAt(s, 11), sales: 6 }]).join()).toMatch(/more than once/);
    expect(run([{ date: dateAt(s, 11), sales: -1 }]).join()).toMatch(/≥ 0/);
    expect(run([{ date: dateAt(s, 11), sales: 2.5 }]).join()).toMatch(/whole units/);
    expect(run([{ date: dateAt(s, 11), sales: 2, promo: 1.5 }]).join()).toMatch(/promotions/);
    expect(run([{ date: "2017-13-01", sales: 2 }]).join()).toMatch(/Invalid/);
  });
});
