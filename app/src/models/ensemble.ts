/**
 * Adaptive ensemble: a weighted average of the forecasters with weights ∝ 1 / WAPE on recent
 * out-of-sample folds. Weights are re-estimated every time new data arrives (adaptive
 * decision-making). In the backtest, fold k is combined with weights learned from folds < k only
 * (equal weights for the first fold), so the ensemble's reported accuracy is not optimistic.
 */
import type { FoldForecast, ModelId } from "./backtest";
import { foldMetrics } from "./backtest";

export type Weights = Partial<Record<ModelId, number>>;

/** WAPE floor so a (near-)perfect model receives (almost) all the weight instead of dividing by zero. */
const WAPE_FLOOR = 1e-9;

export function inverseWapeWeights(wape: Partial<Record<ModelId, number>>): Weights {
  const entries = Object.entries(wape).filter(([, w]) => Number.isFinite(w) && (w as number) >= 0) as [ModelId, number][];
  if (!entries.length) return {};
  const inv = entries.map(([id, w]) => [id, 1 / Math.max(w, WAPE_FLOOR)] as const);
  const total = inv.reduce((s, [, v]) => s + v, 0);
  return Object.fromEntries(inv.map(([id, v]) => [id, v / total])) as Weights;
}

export function combine(forecasts: Partial<Record<ModelId, Float64Array>>, weights: Weights): Float64Array {
  const ids = Object.keys(weights) as ModelId[];
  const len = forecasts[ids[0] as ModelId]?.length ?? 0;
  const out = new Float64Array(len);
  for (const id of ids) {
    const f = forecasts[id];
    const w = weights[id] ?? 0;
    if (!f) continue;
    for (let k = 0; k < len; k++) out[k] = (out[k] as number) + w * (f[k] as number);
  }
  return out;
}

/** Ensemble backtest with weights from earlier folds only. Inputs must share fold origins. */
export function ensembleFolds(byModel: Partial<Record<ModelId, FoldForecast[]>>, naiveScale: number): { folds: FoldForecast[]; weightsByFold: Weights[] } {
  const ids = (Object.keys(byModel) as ModelId[]).filter((id) => byModel[id]?.length);
  const nFolds = Math.min(...ids.map((id) => (byModel[id] as FoldForecast[]).length));
  const folds: FoldForecast[] = [];
  const weightsByFold: Weights[] = [];
  for (let k = 0; k < nFolds; k++) {
    let w: Weights;
    if (k === 0) w = Object.fromEntries(ids.map((id) => [id, 1 / ids.length])) as Weights;
    else w = inverseWapeWeights(Object.fromEntries(ids.map((id) => [id, foldMetrics((byModel[id] as FoldForecast[]).slice(0, k), naiveScale).wape])));
    const first = (byModel[ids[0] as ModelId] as FoldForecast[])[k] as FoldForecast;
    folds.push({
      origin: first.origin,
      actual: first.actual,
      forecast: combine(Object.fromEntries(ids.map((id) => [id, ((byModel[id] as FoldForecast[])[k] as FoldForecast).forecast])), w),
    });
    weightsByFold.push(w);
  }
  return { folds, weightsByFold };
}
