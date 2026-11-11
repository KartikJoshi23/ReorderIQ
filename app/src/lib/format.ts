/** Number and date formatting for the dashboard (en-US grouping, compact on axes). */
const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const nfCompact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const fmt = {
  int: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : nf0.format(v)),
  one: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : nf1.format(v)),
  compact: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : nfCompact.format(v)),
  /** Axis ticks: one style per axis — full below 1,000, compact from 1,000 (5K, 10K, 15K…). */
  axis: (v: number) => (Math.abs(v) >= 1_000 ? nfCompact.format(v) : nf0.format(v)),
  pct: (v: number | null | undefined, digits = 1) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`),
  signedPct: (v: number | null | undefined, digits = 1) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`),
  usd: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : usd0.format(v)),
  days: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${nf1.format(v)} days`),
  date: (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }),
  shortDate: (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" }),
  weekday: (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { weekday: "short", timeZone: "UTC" }),
};
