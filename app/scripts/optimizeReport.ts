/**
 * Offline validation of the Bayesian optimiser on the real data (local only).
 * Run from app/:   npx vite-node scripts/optimizeReport.ts
 * For each shortlist series at the replay start: forecast (ETS + Ridge ensemble; the LSTM is omitted
 * here only to keep this offline sweep fast), 64 simulated 13-week demand paths with Gamma lead
 * times (common random numbers), then on the SAME paths:
 *   baseline static policy (classic formula ROP, EOQ) · AI formula policy (dynamic, b = AI SS, EOQ)
 *   · Bayesian optimisation (38 evaluations) · exhaustive 30×30 grid (900 evaluations).
 * Writes ../data/processed/ts-optimize.json (+ the default series' paths for the notebook's NumPy check).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { PackSchema } from "../src/data/schema";
import { initialAsOf, seriesFromPack } from "../src/data/series";
import { DEFAULT_COSTS, defaultConstraints, economicOrderQuantity } from "../src/inventory/costs";
import { DEFAULT_LEAD_TIME } from "../src/inventory/leadTime";
import { relativeErrors } from "../src/inventory/monteCarlo";
import { buildScenarioSet, type PolicySpec } from "../src/inventory/policySim";
import { aiPolicy, baselinePolicy } from "../src/inventory/safetyStock";
import { DEFAULT_ENGINE_SETTINGS, ForecastEngine } from "../src/models/engine";
import { evaluatePolicy, gridSearch, optimizeInventory, type OptimizeSettings } from "../src/optimize/inventoryOpt";

const here = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const PLAIN = here("../../data/processed/favorita-pack.json");
const OUT = here("../../data/processed/ts-optimize.json");
if (!existsSync(PLAIN)) throw new Error(`Missing ${PLAIN}`);
const pack = PackSchema.parse(JSON.parse(readFileSync(PLAIN, "utf-8")));
const params = { serviceLevel: 0.95, leadTime: DEFAULT_LEAD_TIME };

const rows: Record<string, unknown>[] = [];
let detail: Record<string, unknown> | null = null;
for (const info of pack.series) {
  const s = seriesFromPack(pack, info.id);
  const asOf = initialAsOf(s);
  const r = await new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, useLstm: false }).run(s, asOf);
  const base = baselinePolicy(s, asOf, params);
  const ai = aiPolicy(r, params);
  const scenario = { forecast: r.ensemble.forecast, relErrors: relativeErrors(r.ensemble.folds), leadTime: params.leadTime, seed: 17 };
  const set = buildScenarioSet(scenario);
  const settings: OptimizeSettings = { mode: "dynamic", costs: DEFAULT_COSTS, constraints: defaultConstraints(set.meanDemand), referenceSafetyStock: ai.safetyStock };
  const eoq = economicOrderQuantity(set.meanDemand, DEFAULT_COSTS);
  const policies: Record<string, PolicySpec> = {
    baseline: { kind: "static", reorderPoint: base.reorderPoint, orderQty: eoq },
    aiFormula: { kind: "dynamic", buffer: ai.safetyStock, orderQty: eoq },
  };
  const evals = Object.fromEntries(Object.entries(policies).map(([k, p]) => [k, evaluatePolicy(set, p, settings)]));

  let t0 = performance.now();
  const bo = optimizeInventory(set, settings, { seed: 9 });
  const boMs = performance.now() - t0;
  const boStatic = optimizeInventory(set, { ...settings, mode: "static" }, { seed: 9 });
  t0 = performance.now();
  const grid = gridSearch(set, settings, 30);
  const gridMs = performance.now() - t0;

  const summary = (e: { metrics: { costPerDay: number; fillRate: number; cycleServiceLevel: number; avgOnHand: number; ordersPerWeek: number; peakOnHandP95: number } } | undefined) =>
    e ? { cost: e.metrics.costPerDay, fill: e.metrics.fillRate, csl: e.metrics.cycleServiceLevel, avgOnHand: e.metrics.avgOnHand, ordersPerWeek: e.metrics.ordersPerWeek, peak: e.metrics.peakOnHandP95 } : null;
  const row = {
    series: info.id,
    meanDemand: set.meanDemand,
    constraints: settings.constraints,
    eoq,
    baseline: { ...summary(evals.baseline), feasible: Math.min(...Object.values(evals.baseline?.margins ?? {})) >= 0 },
    aiFormula: { ...summary(evals.aiFormula), feasible: Math.min(...Object.values(evals.aiFormula?.margins ?? {})) >= 0 },
    bo: { x: bo.best?.x, ...summary(bo.best?.details), evaluations: bo.evaluations, ms: boMs, convergence: bo.history.map((h) => h.bestObjective) },
    boStatic: { x: boStatic.best?.x, ...summary(boStatic.best?.details) },
    grid: { x: grid.best?.x, cost: grid.best?.objective, evaluations: grid.evaluations, ms: gridMs },
  };
  rows.push(row);
  const pct = (v?: number) => `${(100 * (v ?? Number.NaN)).toFixed(1)}%`;
  console.log(`${info.id.padEnd(17)} baseline $${row.baseline.cost?.toFixed(0)}/d fill ${pct(row.baseline.fill)} | AI-formula $${row.aiFormula.cost?.toFixed(0)}/d fill ${pct(row.aiFormula.fill)} | BO $${row.bo.cost?.toFixed(0)}/d fill ${pct(row.bo.fill)} (${bo.evaluations} evals, ${(boMs / 1000).toFixed(1)}s) | grid $${row.grid.cost?.toFixed(0)}/d (900 evals, ${(gridMs / 1000).toFixed(1)}s) | gap ${(100 * ((row.bo.cost ?? 0) / (row.grid.cost ?? 1) - 1)).toFixed(2)}%`);
  if (info.id === pack.defaultSeries) {
    detail = { series: info.id, warmup: set.warmup, days: set.days, demand: set.demand.map((d) => Array.from(d)), leadTimes: set.leadTimes.map((l) => Array.from(l)), expectedLeadTimeDemand: Array.from(set.expectedLeadTimeDemand), costs: DEFAULT_COSTS, policies: { ...policies, bo: bo.best?.details?.policy }, metrics: { baseline: evals.baseline?.metrics, aiFormula: evals.aiFormula?.metrics, bo: bo.best?.details?.metrics }, surface: bo.surface, history: bo.history.map(({ details: _d, ...h }) => h) };
  }
}
writeFileSync(OUT, JSON.stringify({ generatedAt: new Date().toISOString(), note: "Forecasts: ETS + Ridge ensemble (LSTM omitted for speed in this offline sweep)", rows, detail }, null, 1));
console.log("wrote", OUT);
