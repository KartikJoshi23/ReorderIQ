/**
 * Runtime schema for the data pack (real Favorita pack or synthetic demo pack).
 * Mirrors scripts/favorita_data.py::build_pack — every daily array has `calendar.days` entries,
 * index 0 = `calendar.start`.
 */
import { z } from "zod";

const isoDate = z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "expected YYYY-MM-DD");
const flags = z.array(z.union([z.literal(0), z.literal(1)]));

export const StoreSchema = z.object({
  store: z.number().int(),
  city: z.string(),
  state: z.string(),
  type: z.string(),
  cluster: z.number().int(),
  holiday: flags,
  event: flags,
  quake: flags,
  workDay: flags,
  closed: flags,
  /** Sparse map: day index (as string key) -> holiday/event description(s). */
  names: z.record(z.string(), z.string()),
});

export const SeriesSchema = z.object({
  id: z.string(),
  store: z.number().int(),
  family: z.string(),
  unit: z.string(),
  fractional: z.boolean(),
  /** Units sold per day; null = not observed (future). */
  sales: z.array(z.number().nonnegative().nullable()),
  /** Number of items in the family on promotion (planned in advance). */
  promo: z.array(z.number().int().nonnegative()),
});

export const PackSchema = z
  .object({
    schema: z.literal("favorita-pack/1"),
    meta: z.object({
      source: z.string(),
      url: z.string(),
      terms: z.string(),
      synthetic: z.boolean().optional(),
    }),
    calendar: z.object({
      start: isoDate,
      days: z.number().int().positive(),
      trainEnd: isoDate,
      modelStart: isoDate,
      replayStart: isoDate,
      promoKnownFrom: isoDate,
      oil: z.array(z.number()),
    }),
    stores: z.array(StoreSchema).min(1),
    series: z.array(SeriesSchema).min(1),
    defaultSeries: z.string(),
  })
  .superRefine((pack, ctx) => {
    const n = pack.calendar.days;
    const check = (len: number, path: (string | number)[]) => {
      if (len !== n) ctx.addIssue({ code: z.ZodIssueCode.custom, path, message: `expected ${n} daily values, got ${len}` });
    };
    check(pack.calendar.oil.length, ["calendar", "oil"]);
    pack.stores.forEach((s, i) => {
      for (const key of ["holiday", "event", "quake", "workDay", "closed"] as const) check(s[key].length, ["stores", i, key]);
    });
    const storeIds = new Set(pack.stores.map((s) => s.store));
    pack.series.forEach((s, i) => {
      check(s.sales.length, ["series", i, "sales"]);
      check(s.promo.length, ["series", i, "promo"]);
      if (!storeIds.has(s.store)) ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["series", i, "store"], message: `unknown store ${s.store}` });
    });
    if (!pack.series.some((s) => s.id === pack.defaultSeries)) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["defaultSeries"], message: "default series not in pack" });
    }
  });

export type Pack = z.infer<typeof PackSchema>;
export type PackStore = z.infer<typeof StoreSchema>;
export type PackSeries = z.infer<typeof SeriesSchema>;
