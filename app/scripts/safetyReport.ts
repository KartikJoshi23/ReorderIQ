/**
 * Offline safety-stock report on the real data (local only; reads the git-ignored unencrypted pack).
 * Run from app/:   npx vite-node scripts/safetyReport.ts
 * For each shortlist series, at the replay start (orders placed on 2017-06-20):
 *   baseline (classic) vs AI-enhanced (forecast-driven) safety stock and reorder point from the slide
 *   formula, and the Monte Carlo view of both, under default settings (95%, L = 7 d, σ_L = 2 d, gamma).
 * Writes ../data/processed/ts-safety.json for the notebook's §9 cross-check.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { PackSchema } from "../src/data/schema";
import { initialAsOf, seriesFromPack } from "../src/data/series";
import { DEFAULT_LEAD_TIME } from "../src/inventory/leadTime";
import { relativeErrors, runMonteCarlo, runScenarios } from "../src/inventory/monteCarlo";
import { aiPolicy, baselinePolicy, dynamicPath, slideFormula } from "../src/inventory/safetyStock";
import { ForecastEngine } from "../src/models/engine";

const here = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const PLAIN = here("../../data/processed/favorita-pack.json");
const OUT = here("../../data/processed/ts-safety.json");
if (!existsSync(PLAIN)) throw new Error(`Missing ${PLAIN}`);
const pack = PackSchema.parse(JSON.parse(readFileSync(PLAIN, "utf-8")));
const params = { serviceLevel: 0.95, leadTime: DEFAULT_LEAD_TIME };
const TRIALS = 200_000;

// Formula cross-check under its own assumptions (golden inputs), as reported in VERIFICATION.md
const g = JSON.parse(readFileSync(here("../tests/golden/favorita-golden.json"), "utf-8"));
const f0 = slideFormula({ serviceLevel: 0.95, dBar: g.baselineWindow.mean, sigmaD: g.baselineWindow.sd, leadTime: 7, sigmaL: 2 });
const check = Object.fromEntries((["normal", "gamma"] as const).map((kind) => {
  const mc = runMonteCarlo({ demand: { kind: "iid-normal", mean: g.baselineWindow.mean, sigma: g.baselineWindow.sd }, leadTime: { kind, mean: 7, sd: 2 }, serviceLevel: 0.95, reorderPoint: f0.reorderPoint, trials: 400_000, seed: 2026 });
  return [kind, { ssMc: mc.safetyStock, ssFormula: f0.safetyStock, diff: mc.safetyStock / f0.safetyStock - 1, pStockout: mc.stockoutProbability }];
}));
console.log("Formula check (400k trials):", JSON.stringify(check));

const rows: Record<string, unknown>[] = [];
let detail: Record<string, unknown> | null = null;
for (const info of pack.series) {
  const s = seriesFromPack(pack, info.id);
  const asOf = initialAsOf(s);
  const engine = new ForecastEngine();
  const r = await engine.run(s, asOf);
  engine.dispose();
  const base = baselinePolicy(s, asOf, params);
  const ai = aiPolicy(r, params);
  const rel = relativeErrors(r.ensemble.folds);
  const mcInp = { demand: { kind: "forecast-bootstrap" as const, forecast: r.ensemble.forecast, relErrors: rel }, leadTime: params.leadTime, serviceLevel: 0.95, trials: TRIALS, seed: 11 };
  const atAi = runMonteCarlo({ ...mcInp, reorderPoint: ai.reorderPoint });
  const atBase = runMonteCarlo({ ...mcInp, reorderPoint: base.reorderPoint });
  rows.push({
    series: info.id,
    baseline: { dBar: base.dBar, sigmaD: base.sigmaD, ss: base.safetyStock, rop: base.reorderPoint, leadTimeShare: base.leadTimeShare },
    ai: { dBar: ai.dBar, sigmaE: ai.sigmaD, errorCount: ai.errorCount, ss: ai.safetyStock, rop: ai.reorderPoint, leadTimeDemand: ai.leadTimeDemand, leadTimeShare: ai.leadTimeShare },
    mc: { ropNeeded: atAi.reorderPoint, ssNeeded: atAi.safetyStock, meanDdlt: atAi.mean, skewness: atAi.skewness, pStockoutAtAiRop: atAi.stockoutProbability, pStockoutAtBaselineRop: atBase.stockoutProbability },
  });
  console.log(`${info.id.padEnd(17)} baseline SS ${base.safetyStock.toFixed(0)} ROP ${base.reorderPoint.toFixed(0)} | AI SS ${ai.safetyStock.toFixed(0)} ROP ${ai.reorderPoint.toFixed(0)} (σ_e ${ai.sigmaD.toFixed(0)} vs σ_D ${base.sigmaD.toFixed(0)}) | MC ROP ${atAi.reorderPoint.toFixed(0)}  P(stockout) AI ${(100 * (atAi.stockoutProbability ?? 0)).toFixed(1)}% baseline ${(100 * (atBase.stockoutProbability ?? 0)).toFixed(1)}%`);
  if (info.id === pack.defaultSeries) {
    detail = {
      series: info.id,
      forecast: Array.from(r.ensemble.forecast),
      relErrors: rel.map((b) => Array.from(b, (v) => (Number.isFinite(v) ? v : null))),
      mc: { ssNeeded: atAi.safetyStock, ropNeeded: atAi.reorderPoint, mean: atAi.mean, quantiles: atAi.quantiles, histogram: atAi.histogram, convergence: atAi.convergence },
      scenarios: runScenarios({ ...mcInp, reorderPoint: ai.reorderPoint, trials: 100_000 }).map((x) => ({ id: x.scenario.id, label: x.scenario.label, ropNeeded: x.result.reorderPoint, pStockout: x.result.stockoutProbability, expectedShortage: x.result.expectedShortage })),
      path: dynamicPath(r, params),
    };
  }
}
writeFileSync(OUT, JSON.stringify({ generatedAt: new Date().toISOString(), params, trials: TRIALS, formulaCheck: check, rows, detail }, null, 1));
console.log("wrote", OUT);
