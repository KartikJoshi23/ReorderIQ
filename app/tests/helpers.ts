/** Test helpers: fixture/golden loading. Real-data checks are skipped when the local files are absent. */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Pack } from "../src/data/schema";

const here = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));

export const paths = {
  fixtureJson: here("./fixtures/fixture-pack.json"),
  golden: here("./golden/favorita-golden.json"),
  publicPack: here("../public/data/favorita-pack.json"),
  realPlain: here("../../data/processed/favorita-pack.json"),
};

export function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf-8")) as T;
}

/** Real pack written by `scripts/build_data_pack.py` (git-ignored copy); null if absent. */
export function realPlainPack(): Pack | null {
  return existsSync(paths.realPlain) ? readJson<Pack>(paths.realPlain) : null;
}

export interface Golden {
  series: string;
  baselineWindow: { start: string; end: string; n: number; mean: number; sd: number };
  formula: { L: number; sigmaL: number; serviceLevel: number; z: number; termDemand: number; termLeadTime: number; ss: number; rop: number }[];
  inverseNormal: { p: number; z: number }[];
  calendar: { days: number; store44: { holiday: number; event: number; quake: number; work_day: number; closed: number } };
  promoDaysSinceModelStart: number;
  spotChecks: { payday: Record<string, number>; xmasRamp: Record<string, number> };
}
