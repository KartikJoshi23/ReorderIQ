import { describe, expect, it } from "vitest";
import { addDays, dayParts, daysBetween, epochDay, isoDate } from "../../src/data/dates";

describe("dates", () => {
  it("round-trips ISO dates through epoch days", () => {
    for (const iso of ["1970-01-01", "2013-01-01", "2016-02-29", "2017-08-15", "2025-12-31"]) {
      expect(isoDate(epochDay(iso))).toBe(iso);
    }
    expect(epochDay("1970-01-02")).toBe(1);
  });

  it("rejects malformed and impossible dates", () => {
    expect(() => epochDay("2017-02-30")).toThrow();
    expect(() => epochDay("17-01-01")).toThrow();
    expect(() => epochDay("2017/01/01")).toThrow();
  });

  it("does arithmetic across month, leap-year and year boundaries", () => {
    expect(addDays("2016-02-28", 1)).toBe("2016-02-29");
    expect(addDays("2016-12-31", 1)).toBe("2017-01-01");
    expect(daysBetween("2013-01-01", "2017-08-15")).toBe(1687);
    expect(daysBetween("2013-01-01", "2014-04-01")).toBe(455);
  });

  it("matches pandas conventions for weekday, day-of-year and month-end", () => {
    const p = dayParts(epochDay("2017-08-15")); // Tuesday
    expect(p.dow).toBe(1);
    expect(dayParts(epochDay("2017-08-13")).dow).toBe(6); // Sunday
    expect(dayParts(epochDay("2016-12-31")).dayOfYear).toBe(366);
    expect(dayParts(epochDay("2017-01-01")).dayOfYear).toBe(1);
    expect(dayParts(epochDay("2017-02-28")).isMonthEnd).toBe(true);
    expect(dayParts(epochDay("2016-02-28")).isMonthEnd).toBe(false);
  });
});
