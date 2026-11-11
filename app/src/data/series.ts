/**
 * Daily time-series container for one store × product-family "planning item", plus the logic for
 * adding new observations (slide step 1: "keep the possibility of adding new data").
 *
 * Conventions
 *  - index i  <->  calendar day epoch0 + i  (continuous, no gaps)
 *  - sales[i] = NaN  ->  not (yet) observed
 *  - `asOf`   = index of the last day whose actual sales the system currently knows. The held-back
 *               replay window (replayStart..trainEnd) "arrives" by moving asOf forward.
 */
import { dayParts, epochDay, isoDate } from "./dates";
import type { Pack } from "./schema";

export type DataSource = "real" | "demo";

/** Provenance of each day's sales value. */
export const ORIGIN_PACK = 0;
export const ORIGIN_ADDED = 1;

export interface StoreMeta {
  store: number;
  city: string;
  state: string;
  type: string;
  cluster: number;
}

export interface SeriesData {
  readonly id: string;
  readonly family: string;
  readonly fractional: boolean;
  readonly source: DataSource;
  readonly storeMeta: StoreMeta;
  /** Epoch day of index 0. */
  readonly epoch0: number;
  readonly length: number;
  readonly sales: Float64Array;
  readonly promo: Int32Array;
  readonly closed: Uint8Array;
  readonly holiday: Uint8Array;
  readonly event: Uint8Array;
  readonly quake: Uint8Array;
  readonly workDay: Uint8Array;
  readonly oil: Float64Array;
  readonly origin: Uint8Array;
  readonly names: ReadonlyMap<number, string>;
  readonly modelStart: number;
  readonly replayStart: number;
  readonly trainEnd: number;
  readonly promoKnownFrom: number;
}

export interface SeriesInfo {
  id: string;
  store: number;
  family: string;
  city: string;
  label: string;
}

export function listSeries(pack: Pack): SeriesInfo[] {
  const stores = new Map(pack.stores.map((s) => [s.store, s]));
  return pack.series.map((s) => {
    const city = stores.get(s.store)?.city ?? "";
    return { id: s.id, store: s.store, family: s.family, city, label: `Store ${s.store} · ${titleCase(s.family)}` };
  });
}

export function seriesFromPack(pack: Pack, id: string): SeriesData {
  const ps = pack.series.find((s) => s.id === id);
  if (!ps) throw new Error(`Series "${id}" is not in the data pack`);
  const st = pack.stores.find((s) => s.store === ps.store);
  if (!st) throw new Error(`Store ${ps.store} is not in the data pack`);
  const cal = pack.calendar;
  const epoch0 = epochDay(cal.start);
  const at = (iso: string) => epochDay(iso) - epoch0;
  const n = cal.days;
  return {
    id: ps.id,
    family: ps.family,
    fractional: ps.fractional,
    source: pack.meta.synthetic ? "demo" : "real",
    storeMeta: { store: st.store, city: st.city, state: st.state, type: st.type, cluster: st.cluster },
    epoch0,
    length: n,
    sales: Float64Array.from(ps.sales, (v) => (v === null ? Number.NaN : v)),
    promo: Int32Array.from(ps.promo),
    closed: Uint8Array.from(st.closed),
    holiday: Uint8Array.from(st.holiday),
    event: Uint8Array.from(st.event),
    quake: Uint8Array.from(st.quake),
    workDay: Uint8Array.from(st.workDay),
    oil: Float64Array.from(cal.oil),
    origin: new Uint8Array(n),
    names: new Map(Object.entries(st.names).map(([k, v]) => [Number(k), v])),
    modelStart: at(cal.modelStart),
    replayStart: at(cal.replayStart),
    trainEnd: at(cal.trainEnd),
    promoKnownFrom: at(cal.promoKnownFrom),
  };
}

export function dateAt(s: SeriesData, i: number): string {
  return isoDate(s.epoch0 + i);
}

export function indexOfDate(s: SeriesData, iso: string): number {
  return epochDay(iso) - s.epoch0;
}

/** The replay starts with the final 8 weeks hidden: the system knows actuals up to replayStart − 1. */
export function initialAsOf(s: SeriesData): number {
  return s.replayStart - 1;
}

/** Last index that holds an observed (finite) sales value. */
export function lastObservedIndex(s: SeriesData): number {
  for (let i = s.length - 1; i >= 0; i--) if (Number.isFinite(s.sales[i])) return i;
  return -1;
}

/** Move `asOf` forward by `days`, never past the last day that actually has an observation. */
export function releaseDays(s: SeriesData, asOf: number, days: number): number {
  let target = Math.min(asOf + Math.max(0, Math.floor(days)), lastObservedIndex(s));
  while (target > asOf && !Number.isFinite(s.sales[target])) target--;
  return Math.max(asOf, target);
}

// ---------------------------------------------------------------------------------------------
// Adding new data
// ---------------------------------------------------------------------------------------------

export interface NewObservation {
  date: string;
  sales: number;
  /** Items on promotion that day; keeps the planned value when omitted. */
  promo?: number;
}

export interface ApplyResult {
  series: SeriesData;
  asOf: number;
  added: number;
  revised: number;
  extendedBy: number;
}

export class ObservationError extends Error {
  constructor(readonly problems: string[]) {
    super(problems.join("\n"));
    this.name = "ObservationError";
  }
}

/**
 * Apply new or corrected daily observations and return a new SeriesData (inputs are not mutated).
 * Rules:
 *  - dates on or before `asOf` are corrections (overwrite);
 *  - dates after `asOf` are new and must continue the history without gaps (asOf+1, asOf+2, …);
 *  - dates may extend beyond the pack's calendar; new days get default calendar features
 *    (closed on 25 Dec / 1 Jan, no holiday flags, oil carried forward);
 *  - a positive sale on a day flagged closed re-opens that day.
 */
export function applyObservations(s: SeriesData, asOf: number, rows: NewObservation[]): ApplyResult {
  const problems: string[] = [];
  const seen = new Set<number>();
  const parsed: { i: number; row: NewObservation }[] = [];
  for (const row of rows) {
    let i: number;
    try {
      i = indexOfDate(s, row.date);
    } catch (e) {
      problems.push((e as Error).message);
      continue;
    }
    if (i < s.modelStart) problems.push(`${row.date}: before the modelling window (${dateAt(s, s.modelStart)})`);
    if (seen.has(i)) problems.push(`${row.date}: appears more than once`);
    seen.add(i);
    if (!Number.isFinite(row.sales) || row.sales < 0) problems.push(`${row.date}: sales must be a number ≥ 0`);
    else if (!s.fractional && !Number.isInteger(row.sales)) problems.push(`${row.date}: ${titleCase(s.family)} is sold in whole units`);
    if (row.promo !== undefined && (!Number.isInteger(row.promo) || row.promo < 0)) {
      problems.push(`${row.date}: promotions must be a whole number ≥ 0`);
    }
    parsed.push({ i, row });
  }
  const fresh = parsed.filter((p) => p.i > asOf).map((p) => p.i).sort((a, b) => a - b);
  fresh.forEach((idx, k) => {
    const expected = asOf + 1 + k;
    if (idx !== expected) problems.push(`Missing data for ${dateAt(s, expected)} — new days must follow on from ${dateAt(s, asOf)} without gaps`);
  });
  if (problems.length) throw new ObservationError([...new Set(problems)]);

  const maxIdx = Math.max(s.length - 1, ...parsed.map((p) => p.i));
  const out = extendCalendar(s, maxIdx + 1);
  let revised = 0;
  for (const { i, row } of parsed) {
    if (i <= asOf) revised++;
    (out.sales as Float64Array)[i] = row.sales;
    (out.origin as Uint8Array)[i] = ORIGIN_ADDED;
    if (row.promo !== undefined) (out.promo as Int32Array)[i] = row.promo;
    if (row.sales > 0) (out.closed as Uint8Array)[i] = 0;
  }
  return {
    series: out,
    asOf: fresh.length ? Math.max(asOf, fresh[fresh.length - 1] ?? asOf) : asOf,
    added: fresh.length,
    revised,
    extendedBy: out.length - s.length,
  };
}

/**
 * Copy of `s` with arrays grown to `length` days. New days get default calendar features:
 * closed on 25 Dec / 1 Jan, no holiday flags, oil carried forward, and promotions set to the
 * median of the last 28 known days (future promotion plans beyond the data are unknown).
 */
export function extendCalendar(s: SeriesData, length: number): SeriesData {
  const n = Math.max(length, s.length); // always returns fresh arrays, so callers may mutate the copy
  const grow = <T extends Float64Array | Int32Array | Uint8Array>(a: T, fill: number): T => {
    const b = new (a.constructor as new (len: number) => T)(n);
    b.set(a);
    if (n > a.length) b.fill(fill as never, a.length);
    return b;
  };
  const recentPromo = Array.from(s.promo.subarray(Math.max(0, s.length - 28))).sort((a, b) => a - b);
  const promoFill = recentPromo.length ? (recentPromo[Math.floor((recentPromo.length - 1) / 2)] as number) : 0;
  const closed = grow(s.closed, 0);
  for (let i = s.length; i < n; i++) {
    const p = dayParts(s.epoch0 + i);
    closed[i] = (p.month === 12 && p.day === 25) || (p.month === 1 && p.day === 1) ? 1 : 0;
  }
  return {
    ...s,
    length: n,
    sales: grow(s.sales, Number.NaN),
    promo: grow(s.promo, promoFill),
    closed,
    holiday: grow(s.holiday, 0),
    event: grow(s.event, 0),
    quake: grow(s.quake, 0),
    workDay: grow(s.workDay, 0),
    oil: grow(s.oil, s.oil[s.oil.length - 1] ?? 0),
    origin: grow(s.origin, ORIGIN_PACK),
  };
}

export function titleCase(text: string): string {
  return text.toLowerCase().replace(/(^|[\s,/])([a-z])/g, (_, sep: string, c: string) => sep + c.toUpperCase());
}
