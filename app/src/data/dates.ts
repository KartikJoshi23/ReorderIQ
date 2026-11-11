/**
 * Calendar-day arithmetic on UTC epoch days (days since 1970-01-01).
 * Using integer day numbers avoids every time-zone / daylight-saving pitfall of `Date`.
 */
const MS_PER_DAY = 86_400_000;

/** "YYYY-MM-DD" -> epoch day. Throws on malformed or impossible dates. */
export function epochDay(iso: string): number {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) throw new Error(`Invalid date "${iso}" (expected YYYY-MM-DD)`);
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const ms = Date.UTC(y, mo - 1, d);
  const back = new Date(ms);
  if (back.getUTCFullYear() !== y || back.getUTCMonth() !== mo - 1 || back.getUTCDate() !== d) {
    throw new Error(`Invalid calendar date "${iso}"`);
  }
  return Math.round(ms / MS_PER_DAY);
}

/** Epoch day -> "YYYY-MM-DD". */
export function isoDate(day: number): string {
  return new Date(day * MS_PER_DAY).toISOString().slice(0, 10);
}

export function addDays(iso: string, n: number): string {
  return isoDate(epochDay(iso) + n);
}

/** Whole days from `a` to `b` (positive when b is later). */
export function daysBetween(a: string, b: string): number {
  return epochDay(b) - epochDay(a);
}

export interface DayParts {
  year: number;
  /** 1..12 */
  month: number;
  /** 1..31 */
  day: number;
  /** 0 = Monday … 6 = Sunday (same convention as pandas `dayofweek`). */
  dow: number;
  /** 1..366 (same as pandas `dayofyear`). */
  dayOfYear: number;
  isMonthEnd: boolean;
}

export function dayParts(day: number): DayParts {
  const d = new Date(day * MS_PER_DAY);
  const year = d.getUTCFullYear();
  const month = d.getUTCMonth() + 1;
  const dom = d.getUTCDate();
  const startOfYear = Math.round(Date.UTC(year, 0, 1) / MS_PER_DAY);
  const tomorrow = new Date((day + 1) * MS_PER_DAY);
  return {
    year,
    month,
    day: dom,
    dow: (d.getUTCDay() + 6) % 7,
    dayOfYear: day - startOfYear + 1,
    isMonthEnd: tomorrow.getUTCDate() === 1,
  };
}
