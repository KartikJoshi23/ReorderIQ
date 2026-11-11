/**
 * Parse new daily observations from CSV text (dashboard "Add data" upload).
 *
 * Accepted layouts (header row required, comma or semicolon separated, quotes supported):
 *   date,sales[,onpromotion]                       — simple format ("units"/"quantity" and "promo" also accepted)
 *   id,date,store_nbr,family,sales,onpromotion     — Kaggle train.csv format; rows for other
 *                                                     stores/families are skipped
 */
import { epochDay } from "./dates";
import type { NewObservation } from "./series";

export interface ParseIssue {
  line: number;
  message: string;
}

export interface ParseResult {
  rows: NewObservation[];
  errors: ParseIssue[];
  /** Rows skipped because they belong to another store/family (Kaggle format). */
  skipped: number;
}

const SALES_HEADERS = ["sales", "units", "quantity", "units_sold"];
const PROMO_HEADERS = ["onpromotion", "promo", "promotions"];

export function splitCsvLine(line: string, sep: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === sep) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((v) => v.trim());
}

export function parseObservationsCsv(text: string, target?: { store: number; family: string }): ParseResult {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/);
  const headerLine = lines.findIndex((l) => l.trim() !== "");
  if (headerLine < 0) return { rows: [], errors: [{ line: 1, message: "The file is empty." }], skipped: 0 };

  const rawHeader = lines[headerLine] as string;
  const sep = rawHeader.includes(";") && !rawHeader.includes(",") ? ";" : ",";
  const header = splitCsvLine(rawHeader, sep).map((h) => h.toLowerCase());
  const col = (names: string[]) => header.findIndex((h) => names.includes(h));
  const iDate = col(["date"]);
  const iSales = col(SALES_HEADERS);
  const iPromo = col(PROMO_HEADERS);
  const iStore = col(["store_nbr", "store"]);
  const iFamily = col(["family"]);

  const errors: ParseIssue[] = [];
  if (iDate < 0 || iSales < 0) {
    errors.push({ line: headerLine + 1, message: `Header must contain "date" and "sales" columns (found: ${header.join(", ")})` });
    return { rows: [], errors, skipped: 0 };
  }

  const rows: NewObservation[] = [];
  let skipped = 0;
  for (let n = headerLine + 1; n < lines.length; n++) {
    const line = lines[n] as string;
    if (line.trim() === "") continue;
    const cells = splitCsvLine(line, sep);
    const lineNo = n + 1;
    if (target && iStore >= 0 && iFamily >= 0) {
      if (Number(cells[iStore]) !== target.store || (cells[iFamily] ?? "").toUpperCase() !== target.family.toUpperCase()) {
        skipped++;
        continue;
      }
    }
    const date = cells[iDate] ?? "";
    try {
      epochDay(date);
    } catch {
      errors.push({ line: lineNo, message: `"${date}" is not a valid date (use YYYY-MM-DD)` });
      continue;
    }
    const sales = Number(cells[iSales]);
    if ((cells[iSales] ?? "") === "" || !Number.isFinite(sales) || sales < 0) {
      errors.push({ line: lineNo, message: `sales "${cells[iSales] ?? ""}" must be a number ≥ 0` });
      continue;
    }
    const row: NewObservation = { date, sales };
    if (iPromo >= 0 && (cells[iPromo] ?? "") !== "") {
      const promo = Number(cells[iPromo]);
      if (!Number.isInteger(promo) || promo < 0) {
        errors.push({ line: lineNo, message: `promotions "${cells[iPromo]}" must be a whole number ≥ 0` });
        continue;
      }
      row.promo = promo;
    }
    rows.push(row);
  }
  if (rows.length === 0 && errors.length === 0) {
    errors.push({ line: headerLine + 1, message: skipped ? "No rows match the selected store and product family." : "No data rows found." });
  }
  return { rows, errors, skipped };
}
