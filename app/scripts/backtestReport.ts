/**
 * Offline evaluation of the browser models on the real data (local only; reads the git-ignored
 * unencrypted pack). Run from app/:
 *     npx vite-node scripts/backtestReport.ts
 * Writes ../data/processed/ts-backtest.json, read by the notebook's model-comparison section:
 *   1. Ridge λ selection by rolling-origin validation across the shortlist;
 *   2. full engine (ETS, Ridge, LSTM, ensemble, benchmarks) per series at the replay start.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { PackSchema } from "../src/data/schema";
import { dateAt, extendCalendar, initialAsOf, seriesFromPack } from "../src/data/series";
import { actualsAfter, foldMetrics, foldOrigins, type FoldForecast } from "../src/models/backtest";
import { DEFAULT_ENGINE_SETTINGS, ForecastEngine } from "../src/models/engine";
import { buildHistory } from "../src/models/history";
import { seasonalNaiveScale } from "../src/models/metrics";
import { fitRidge, forecastRidge } from "../src/models/ridge";

const here = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const PLAIN = here("../../data/processed/favorita-pack.json");
const OUT = here("../../data/processed/ts-backtest.json");
const LAMBDAS = [0.1, 1, 3, 10, 30, 100, 300];

if (!existsSync(PLAIN)) throw new Error(`Missing ${PLAIN} — run scripts/build_data_pack.py --write-plain first`);
const pack = PackSchema.parse(JSON.parse(readFileSync(PLAIN, "utf-8")));

const lambdaRows: { series: string; lambda: number; wape: number }[] = [];
const results: Record<string, unknown>[] = [];

for (const info of pack.series) {
  const base = seriesFromPack(pack, info.id);
  const asOf = initialAsOf(base);
  const s = extendCalendar(base, asOf + 29);
  const origins = foldOrigins(s, asOf);
  const scale = seasonalNaiveScale(s.sales, (i) => s.closed[i] === 0 && Number.isFinite(s.sales[i] as number), s.modelStart, origins[0] as number);
  for (const lambda of LAMBDAS) {
    const folds: FoldForecast[] = origins.map((o) => {
      const h = buildHistory(s, o);
      return { origin: o, forecast: forecastRidge(fitRidge(s, h, o, lambda), s, h, 28), actual: actualsAfter(s, o, 28) };
    });
    lambdaRows.push({ series: info.id, lambda, wape: foldMetrics(folds, scale).wape });
  }
}
const meanWape = LAMBDAS.map((lambda) => {
  const w = lambdaRows.filter((r) => r.lambda === lambda).map((r) => r.wape);
  return { lambda, meanWape: w.reduce((a, b) => a + b, 0) / w.length };
});
const best = meanWape.reduce((a, b) => (b.meanWape < a.meanWape ? b : a));
console.log("Ridge λ selection (mean WAPE across series):", meanWape.map((r) => `${r.lambda}: ${(r.meanWape * 100).toFixed(2)}%`).join(" | "), "→ best", best.lambda);

for (const info of pack.series) {
  const s = seriesFromPack(pack, info.id);
  const asOf = initialAsOf(s);
  const engine = new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, ridgeLambda: best.lambda });
  const t0 = performance.now();
  const r = await engine.run(s, asOf);
  const secs = (performance.now() - t0) / 1000;
  engine.dispose();
  const row = (id: string, m: { metrics: ReturnType<typeof foldMetrics> }) => ({ model: id, ...m.metrics });
  const models = [
    ...(["ets", "ridge", "lstm"] as const).filter((id) => r.models[id]).map((id) => row(id, r.models[id] as { metrics: ReturnType<typeof foldMetrics> })),
    row("ensemble", r.ensemble),
    row("ma28", r.baselines.ma28),
    row("snaive", r.baselines.snaive),
  ];
  results.push({
    series: info.id,
    asOfDate: dateAt(s, asOf),
    origins: r.ensemble.folds.map((f) => dateAt(s, f.origin)),
    seconds: secs,
    lstmEpochs: r.lstm?.epochsRun,
    weights: r.ensemble.weights,
    drivers: r.drivers,
    models,
    foldForecasts: Object.fromEntries(
      [...(["ets", "ridge", "lstm"] as const).filter((id) => r.models[id]).map((id) => [id, r.models[id]?.folds]), ["ensemble", r.ensemble.folds]].map(([id, folds]) => [
        id as string,
        (folds as FoldForecast[]).map((f) => ({ origin: dateAt(s, f.origin), forecast: Array.from(f.forecast) })),
      ]),
    ),
  });
  console.log(`${info.id.padEnd(18)} ${secs.toFixed(0)}s  ` + models.map((m) => `${m.model} ${(m.wape * 100).toFixed(1)}%`).join("  "));
}

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify({ generatedAt: new Date().toISOString(), lambdaSelection: { grid: meanWape, best: best.lambda, rows: lambdaRows }, results }, null, 1));
console.log("wrote", OUT);
