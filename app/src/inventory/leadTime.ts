/**
 * Replenishment lead-time models (days). The Favorita data has no supplier lead times, so L and σ_L
 * are assumptions set on the dashboard (defaults 7 d and 2 d) — or estimated from uploaded
 * order/receipt records, which turns the assumption into data.
 *
 *   gamma      moment-matched: shape k = (L/σ_L)², scale θ = σ_L²/L  → positive, right-skewed delays (default)
 *   normal     N(L, σ_L²) truncated at 0                             → the formula's implicit assumption
 *   empirical  resampling of observed lead times                      → from uploaded records
 */
import type { Rng } from "../lib/random";

export type LeadTimeKind = "gamma" | "normal" | "empirical";

export interface LeadTimeModel {
  kind: LeadTimeKind;
  /** Mean lead time L (days). */
  mean: number;
  /** Lead-time standard deviation σ_L (days). */
  sd: number;
  /** Observed lead times (days) for the empirical model. */
  samples?: readonly number[];
}

export const DEFAULT_LEAD_TIME: LeadTimeModel = { kind: "gamma", mean: 7, sd: 2 };

/** Gamma(shape, scale) by Marsaglia & Tsang (2000); shape < 1 handled by the boost u^{1/shape}. */
export function sampleGamma(rng: Rng, shape: number, scale: number): number {
  if (shape < 1) return sampleGamma(rng, shape + 1, scale) * Math.pow(rng.next() || Number.MIN_VALUE, 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = rng.normal();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}

/** Returns a sampler of lead times (days, always > 0). */
export function leadTimeSampler(model: LeadTimeModel, rng: Rng): () => number {
  const { mean, sd } = model;
  if (sd <= 0) return () => mean;
  switch (model.kind) {
    case "gamma": {
      const shape = (mean / sd) ** 2;
      const scale = (sd * sd) / mean;
      return () => sampleGamma(rng, shape, scale);
    }
    case "normal":
      return () => {
        let x: number;
        do x = mean + sd * rng.normal();
        while (x <= 0);
        return x;
      };
    case "empirical": {
      const s = model.samples?.filter((v) => v > 0) ?? [];
      if (!s.length) throw new Error("Empirical lead-time model needs observed lead times");
      return () => s[rng.int(s.length)] as number;
    }
  }
}

export interface LeadTimeRecords {
  samples: number[];
  mean: number;
  sd: number;
  errors: { line: number; message: string }[];
}

/**
 * Parse supplier lead-time records. Accepted headers:
 *   order_date,receipt_date   (dates YYYY-MM-DD; lead time = receipt − order in days)
 *   lead_time_days            (or lead_time / days)
 */
export function parseLeadTimeCsv(text: string): LeadTimeRecords {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim() !== "");
  const errors: LeadTimeRecords["errors"] = [];
  const samples: number[] = [];
  if (!lines.length) return { samples, mean: Number.NaN, sd: Number.NaN, errors: [{ line: 1, message: "The file is empty." }] };
  const sep = (lines[0] as string).includes(";") && !(lines[0] as string).includes(",") ? ";" : ",";
  const header = (lines[0] as string).split(sep).map((h) => h.trim().toLowerCase());
  const iOrder = header.indexOf("order_date");
  const iReceipt = header.indexOf("receipt_date");
  const iDays = header.findIndex((h) => ["lead_time_days", "lead_time", "days"].includes(h));
  if (iDays < 0 && (iOrder < 0 || iReceipt < 0)) {
    return { samples, mean: Number.NaN, sd: Number.NaN, errors: [{ line: 1, message: 'Need "order_date" and "receipt_date" columns, or a "lead_time_days" column.' }] };
  }
  const toDay = (s: string) => {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s.trim());
    return m ? Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])) / 86_400_000 : Number.NaN;
  };
  lines.slice(1).forEach((line, k) => {
    const cells = line.split(sep);
    const v = iDays >= 0 ? Number(cells[iDays]) : toDay(cells[iReceipt] ?? "") - toDay(cells[iOrder] ?? "");
    if (!Number.isFinite(v) || v <= 0 || v > 365) errors.push({ line: k + 2, message: `invalid lead time "${line.trim()}"` });
    else samples.push(v);
  });
  const n = samples.length;
  const mean = n ? samples.reduce((a, b) => a + b, 0) / n : Number.NaN;
  const sd = n > 1 ? Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : Number.NaN;
  return { samples, mean, sd, errors };
}
