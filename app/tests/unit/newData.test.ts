import { describe, expect, it } from "vitest";
import { parseObservationsCsv, splitCsvLine } from "../../src/data/newData";

describe("parseObservationsCsv", () => {
  it("reads the simple format with optional promotions", () => {
    const r = parseObservationsCsv("date,sales,onpromotion\n2017-08-16,10500,40\n2017-08-17, 9800 ,\n");
    expect(r.errors).toEqual([]);
    expect(r.rows).toEqual([{ date: "2017-08-16", sales: 10500, promo: 40 }, { date: "2017-08-17", sales: 9800 }]);
  });

  it("accepts header synonyms, semicolons and a BOM", () => {
    const r = parseObservationsCsv("﻿Date;Units;Promo\n2017-08-16;12;1\n");
    expect(r.rows).toEqual([{ date: "2017-08-16", sales: 12, promo: 1 }]);
  });

  it("filters Kaggle train.csv rows to the selected store and family (quoted family names)", () => {
    const csv = [
      "id,date,store_nbr,family,sales,onpromotion",
      '1,2017-08-16,44,BEVERAGES,10000,30',
      '2,2017-08-16,44,"LIQUOR,WINE,BEER",50,0',
      '3,2017-08-16,45,BEVERAGES,9000,25',
    ].join("\n");
    const r = parseObservationsCsv(csv, { store: 44, family: "BEVERAGES" });
    expect(r.rows).toEqual([{ date: "2017-08-16", sales: 10000, promo: 30 }]);
    expect(r.skipped).toBe(2);
  });

  it("reports line-level errors and a missing header", () => {
    const r = parseObservationsCsv("date,sales\n2017-02-30,5\n2017-08-16,-3\n2017-08-17,abc\n");
    expect(r.rows).toEqual([]);
    expect(r.errors.map((e) => e.line)).toEqual([2, 3, 4]);
    expect(parseObservationsCsv("day,qty\n1,2").errors[0]?.message).toMatch(/"date" and "sales"/);
    expect(parseObservationsCsv("").errors[0]?.message).toMatch(/empty/);
  });

  it("splits quoted CSV fields with escaped quotes", () => {
    expect(splitCsvLine('a,"b,c","d ""e"""', ",")).toEqual(["a", "b,c", 'd "e"']);
  });
});
