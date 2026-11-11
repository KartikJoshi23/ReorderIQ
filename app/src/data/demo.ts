/**
 * Synthetic demo pack — an alternative data source that exercises every dashboard feature.
 * It contains NO real data: demand is simulated from aggregate statistics (demo-params.json,
 * estimated by scripts/favorita_data.py::estimate_demo_params) on a made-up calendar:
 *
 *   log(1 + sales_t) = Σ β_k · x_k(t) + ε_t,   ε_t = φ ε_{t−1} + σ z_t        (AR(1) noise)
 *   promotions: two-state Markov chain (on/off) with log-normal promoted-item counts
 *   oil:        bounded random walk
 *
 * The calendar offsets mirror the real pack (model start, replay window, horizon) so every code
 * path behaves identically in demo and real mode.
 */
import params from "./demo-params.json";
import { addDays, dayParts, epochDay } from "./dates";
import { FEATURES } from "./features";
import type { Pack } from "./schema";
import type { SeriesData } from "./series";
import { createRng } from "../lib/random";

export const DEMO_SERIES_ID = "DEMO-BEVERAGES";

const START = "2021-01-01";
const DAYS = 1704;
const MODEL_START_OFFSET = 455; // same offsets as the real pack (2013-01-01 → 2014-04-01)
const TRAIN_END_OFFSET = 1687;
const REPLAY_DAYS = 56;

/** Fixed-date public holidays used by the synthetic calendar (month-day, name). */
const HOLIDAYS: ReadonlyArray<[string, string]> = [
  ["01-01", "New Year's Day"],
  ["05-01", "Labour Day"],
  ["05-24", "Battle of Pichincha"],
  ["08-10", "Independence Day"],
  ["10-09", "Independence of Guayaquil"],
  ["11-02", "All Souls' Day"],
  ["11-03", "Independence of Cuenca"],
  ["12-25", "Christmas Day"],
];
const CLOSED = new Set(["01-01", "12-25"]);

export function generateDemoPack(seed = 2026): Pack {
  const rng = createRng(seed);
  const epoch0 = epochDay(START);
  const coef = params.coefficients as Record<string, number>;

  const holiday: (0 | 1)[] = [];
  const event: (0 | 1)[] = [];
  const closed: (0 | 1)[] = [];
  const names: Record<string, string> = {};
  const oil: number[] = [];
  const promo: number[] = [];

  let price = 62;
  let on = false;
  for (let i = 0; i < DAYS; i++) {
    const p = dayParts(epoch0 + i);
    const md = `${String(p.month).padStart(2, "0")}-${String(p.day).padStart(2, "0")}`;
    const hol = HOLIDAYS.find(([d]) => d === md);
    const blackFriday = p.month === 11 && p.dow === 4 && p.day >= 23 && p.day <= 29;
    holiday.push(hol ? 1 : 0);
    event.push(blackFriday ? 1 : 0);
    closed.push(CLOSED.has(md) ? 1 : 0);
    if (hol) names[String(i)] = hol[1];
    if (blackFriday) names[String(i)] = "Black Friday";

    price = Math.min(110, Math.max(30, price + 0.9 * rng.normal()));
    oil.push(Math.round(price * 100) / 100);

    on = rng.next() < (on ? params.promo.pOnGivenOn : params.promo.pOnGivenOff);
    promo.push(on ? Math.max(1, Math.round(Math.exp(params.promo.logCountMean + params.promo.logCountSd * rng.normal()))) : 0);
  }

  const zeros = new Array<0 | 1>(DAYS).fill(0);
  // Skeleton series so demand is built from exactly the same feature functions the models use.
  const skeleton: SeriesData = {
    id: DEMO_SERIES_ID, family: "BEVERAGES", fractional: false, source: "demo",
    storeMeta: { store: 0, city: "Demo City", state: "Demo", type: "A", cluster: 0 },
    epoch0, length: DAYS,
    sales: new Float64Array(DAYS), promo: Int32Array.from(promo),
    closed: Uint8Array.from(closed), holiday: Uint8Array.from(holiday), event: Uint8Array.from(event),
    quake: new Uint8Array(DAYS), workDay: new Uint8Array(DAYS), oil: Float64Array.from(oil),
    origin: new Uint8Array(DAYS), names: new Map(),
    modelStart: MODEL_START_OFFSET, replayStart: TRAIN_END_OFFSET - REPLAY_DAYS + 1,
    trainEnd: TRAIN_END_OFFSET, promoKnownFrom: 0,
  };

  const sales: (number | null)[] = [];
  let eps = 0;
  for (let i = 0; i < DAYS; i++) {
    eps = params.noise.ar1 * eps + params.noise.innovationSd * rng.normal();
    if (i > TRAIN_END_OFFSET) {
      sales.push(null);
      continue;
    }
    if (closed[i]) {
      sales.push(0);
      continue;
    }
    let eta = eps;
    for (const [name, beta] of Object.entries(coef)) {
      const fn = FEATURES[name];
      if (!fn) throw new Error(`demo-params.json uses unknown feature "${name}"`);
      eta += beta * fn(skeleton, i);
    }
    sales.push(Math.max(0, Math.round(Math.expm1(eta))));
  }

  return {
    schema: "favorita-pack/1",
    meta: {
      source: "SYNTHETIC DEMO — simulated in your browser from aggregate statistics; contains no real sales data",
      url: "",
      terms: "Synthetic data for demonstration only.",
      synthetic: true,
    },
    calendar: {
      start: START,
      days: DAYS,
      trainEnd: addDays(START, TRAIN_END_OFFSET),
      modelStart: addDays(START, MODEL_START_OFFSET),
      replayStart: addDays(START, TRAIN_END_OFFSET - REPLAY_DAYS + 1),
      promoKnownFrom: START,
      oil,
    },
    stores: [{
      store: 0, city: "Demo City", state: "Demo", type: "A", cluster: 0,
      holiday, event, quake: zeros, workDay: zeros, closed, names,
    }],
    series: [{ id: DEMO_SERIES_ID, store: 0, family: "BEVERAGES", unit: "units", fractional: false, sales, promo }],
    defaultSeries: DEMO_SERIES_ID,
  };
}
