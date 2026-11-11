/**
 * Forecast engine — runs inside the forecast Web Worker.
 *
 * `run(series, asOf)` produces, for the item and "as of" date:
 *   1. a rolling-origin backtest of every model (ETS, Ridge, LSTM, two benchmarks);
 *   2. live forecasts for the next `horizon` days;
 *   3. the adaptive ensemble (weights ∝ 1/WAPE), conformal 80/95% bands and the out-of-sample
 *      errors that the safety-stock and Monte Carlo modules use;
 *   4. driver effects (weekday, promotions, holidays, paydays, Christmas, oil) from the regression.
 *
 * Continuous updating: when called again for the same item with a later `asOf` (new data arrived),
 * ETS and Ridge are refitted from scratch (milliseconds) and the LSTM is fine-tuned from its current
 * weights on the newest data instead of retrained; its backtest is reused and flagged as such.
 * A full LSTM retrain happens on request, on a new item, or if data before its training point changed.
 */
import { dateAt, extendCalendar, type SeriesData } from "../data/series";
import {
  actualsAfter,
  type BacktestSettings,
  type BaselineId,
  DEFAULT_BACKTEST,
  type FoldForecast,
  foldMetrics,
  foldOrigins,
  forecastBaseline,
  MODEL_IDS,
  type ModelId,
} from "./backtest";
import { combine, ensembleFolds, inverseWapeWeights, type Weights } from "./ensemble";
import { fitEts, forecastEts } from "./ets";
import { buildHistory } from "./history";
import { applyIntervals, type Bands, fitIntervals } from "./intervals";
import { DEFAULT_LSTM_CONFIG, disposeLstm, fineTuneLstm, forecastLstm, initTensorflow, type LstmConfig, type LstmModel, type LstmProgress, trainLstm } from "./lstm";
import type { ForecastMetrics } from "./metrics";
import { seasonalNaiveScale } from "./metrics";
import { DEFAULT_RIDGE_LAMBDA, fitRidge, forecastRidge, type RidgeModel, rawCoefficient } from "./ridge";

export interface EngineSettings {
  backtest: BacktestSettings;
  ridgeLambda: number;
  useLstm: boolean;
  lstm: LstmConfig;
}

export const DEFAULT_ENGINE_SETTINGS: EngineSettings = {
  backtest: DEFAULT_BACKTEST,
  ridgeLambda: DEFAULT_RIDGE_LAMBDA,
  useLstm: true,
  lstm: DEFAULT_LSTM_CONFIG,
};

export interface ModelRun {
  forecast: Float64Array;
  folds: FoldForecast[];
  metrics: ForecastMetrics;
}

export interface LstmInfo {
  backend: string;
  action: "trained" | "fine-tuned";
  epochsRun: number;
  trainedThrough: string;
  /** Version whose rolling backtest the LSTM metrics come from. */
  backtestVersion: number;
}

export interface Drivers {
  /** Sales multiplier by weekday relative to Monday (index 0 = Monday). */
  weekday: number[];
  /** Relative effects (e.g. 0.20 = +20 %), conditional on recent sales level. */
  promoPerDoubling: number;
  holiday: number;
  event: number;
  earthquake: number;
  payday: number;
  christmasEve: number;
  oilPer10: number;
}

export interface ForecastResult {
  version: number;
  seriesId: string;
  source: string;
  asOf: number;
  asOfDate: string;
  /** Index of the first forecast day (asOf + 1). */
  start: number;
  dates: string[];
  horizon: number;
  models: Partial<Record<ModelId, ModelRun>>;
  baselines: Record<BaselineId, ModelRun>;
  ensemble: ModelRun & { weights: Weights };
  bands: Bands;
  /** Ensemble out-of-sample errors (actual − forecast) per fold, NaN on closed days — for σ_e and Monte Carlo. */
  foldErrors: Float64Array[];
  naiveScale: number;
  drivers: Drivers;
  lstm?: LstmInfo;
  timingsMs: Record<string, number>;
}

export interface EngineProgress {
  stage: string;
  fraction: number;
  lstm?: LstmProgress;
}

export interface RunOptions {
  /** Force a full LSTM retrain (otherwise the LSTM is fine-tuned when possible). */
  retrainLstm?: boolean;
  /** Override the LSTM setting for this run (the dashboard shows a fast ETS + Ridge result first). */
  useLstm?: boolean;
  /** Cancellation generation the run belongs to (set when the run is queued). */
  generation?: number;
}

/** Thrown when a run is abandoned because a newer request superseded it (e.g. another series was selected). */
export class CancelledError extends Error {
  constructor() {
    super("Forecast run cancelled");
    this.name = "CancelledError";
  }
}

interface LstmState {
  key: string;
  model: LstmModel;
  /**
   * Every LSTM forecast issued so far, keyed by origin: the rolling-backtest folds plus the final
   * forecast of each later run. Each used only data up to its origin, so each is out-of-sample.
   */
  issued: Map<number, Float64Array>;
  backtestVersion: number;
  backend: string;
}

/**
 * Backtest origins after a fine-tune. All models must share fold origins (the ensemble combines
 * fold k of every model), but the LSTM cannot be re-backtested at new origins without retraining.
 * So the folds are drawn from origins where the LSTM really issued a forecast, whose full horizon
 * of actuals is known by `asOf`: the latest one, then each next one at least a horizon earlier
 * (non-overlapping), up to `count` folds, oldest first.
 */
export function rollingOrigins(issued: Iterable<number>, asOf: number, horizon: number, count: number): number[] {
  const complete = [...issued].filter((o) => o + horizon <= asOf).sort((a, b) => b - a);
  const out: number[] = [];
  for (const o of complete) {
    if (out.length === count) break;
    if (!out.length || o <= (out[out.length - 1] as number) - horizon) out.push(o);
  }
  return out.reverse();
}

export class ForecastEngine {
  private version = 0;
  private lstmState: LstmState | null = null;
  private generation = 0;

  constructor(private readonly settings: EngineSettings = DEFAULT_ENGINE_SETTINGS) {}

  /** Current cancellation generation; runs queued under an older generation are abandoned. */
  get currentGeneration(): number {
    return this.generation;
  }

  /** Abandon every queued or running forecast (an LSTM stops at the end of its current epoch). */
  cancelAll(): void {
    this.generation++;
  }

  async run(series: SeriesData, asOf: number, onProgress?: (p: EngineProgress) => void, opts: RunOptions = {}): Promise<ForecastResult> {
    const gen = opts.generation ?? this.generation;
    const stale = () => gen !== this.generation;
    const check = () => {
      if (stale()) throw new CancelledError();
    };
    check();
    const H = this.settings.backtest.horizon;
    const timings: Record<string, number> = {};
    const tick = (name: string, t0: number) => (timings[name] = Math.round(performance.now() - t0));
    const report = (stage: string, fraction: number, lstm?: LstmProgress) => onProgress?.({ stage, fraction, lstm });

    const s = extendCalendar(series, Math.max(series.length, asOf + 1 + H));
    const useLstm = opts.useLstm ?? this.settings.useLstm;
    const key = `${s.source}:${s.id}`;
    const st = this.lstmState;
    const canFineTune = useLstm && !opts.retrainLstm && st !== null && st.key === key && asOf >= st.model.origin;
    // Fine-tune: fold origins come from the LSTM's issued forecasts so every model shares them.
    const rolled = canFineTune && st ? rollingOrigins(st.issued.keys(), asOf, H, this.settings.backtest.folds) : [];
    const origins = rolled.length ? rolled : foldOrigins(s, asOf, this.settings.backtest);
    if (!origins.length) throw new Error("Not enough history before the selected date to backtest the models.");
    const allOrigins = [...origins, asOf];
    const naiveScale = seasonalNaiveScale(s.sales, (i) => s.closed[i] === 0 && Number.isFinite(s.sales[i] as number), s.modelStart, origins[0] as number);
    const fold = (origin: number, forecast: Float64Array): FoldForecast => ({ origin, forecast, actual: actualsAfter(s, origin, H) });

    // --- Statistical and regression models + benchmarks (refitted at every origin) ---------------
    report("Fitting Holt-Winters and Ridge regression", 0.02);
    let t0 = performance.now();
    const etsFolds: FoldForecast[] = [];
    const ridgeFolds: FoldForecast[] = [];
    const baseFolds: Record<BaselineId, FoldForecast[]> = { ma28: [], snaive: [] };
    let etsFinal = new Float64Array(H);
    let ridgeFinal = new Float64Array(H);
    let ridgeModel: RidgeModel | null = null;
    const baseFinal = { ma28: new Float64Array(H), snaive: new Float64Array(H) };
    for (const origin of allOrigins) {
      const h = buildHistory(s, origin);
      const ets = forecastEts(fitEts(s, h, origin), s, H);
      const rm = fitRidge(s, h, origin, this.settings.ridgeLambda);
      const ridge = forecastRidge(rm, s, h, H);
      const ma = forecastBaseline("ma28", s, origin, H);
      const sn = forecastBaseline("snaive", s, origin, H);
      if (origin === asOf) {
        etsFinal = ets;
        ridgeFinal = ridge;
        ridgeModel = rm;
        baseFinal.ma28 = ma;
        baseFinal.snaive = sn;
      } else {
        etsFolds.push(fold(origin, ets));
        ridgeFolds.push(fold(origin, ridge));
        baseFolds.ma28.push(fold(origin, ma));
        baseFolds.snaive.push(fold(origin, sn));
      }
    }
    tick("etsRidge", t0);
    check();

    const models: Partial<Record<ModelId, ModelRun>> = {
      ets: { forecast: etsFinal, folds: etsFolds, metrics: foldMetrics(etsFolds, naiveScale) },
      ridge: { forecast: ridgeFinal, folds: ridgeFolds, metrics: foldMetrics(ridgeFolds, naiveScale) },
    };

    // --- LSTM: full rolling train or warm-start fine-tune --------------------------------------
    let lstmInfo: LstmInfo | undefined;
    if (useLstm) {
      t0 = performance.now();
      const backend = await initTensorflow();
      const lstmProgress = (stage: string, base: number, span: number) => (p: LstmProgress) =>
        report(stage, base + span * (p.epoch / p.epochs), p);

      if (canFineTune && st) {
        // A cancelled fine-tune leaves a partly updated model, so the cached state is dropped.
        if (asOf > st.model.origin) {
          await fineTuneLstm(st.model, s, buildHistory(s, asOf), asOf, lstmProgress("Fine-tuning LSTM on new data", 0.3, 0.6), stale);
          if (stale()) {
            disposeLstm(st.model);
            this.lstmState = null;
            check();
          }
        }
        const forecast = forecastLstm(st.model, s, buildHistory(s, asOf), H);
        // The first forecast issued at an origin is kept (a later what-if at the same date does not replace it).
        if (!st.issued.has(asOf)) st.issued.set(asOf, forecast);
        const folds = origins.map((o) => fold(o, st.issued.get(o) as Float64Array));
        models.lstm = { forecast, folds, metrics: foldMetrics(folds, naiveScale) };
        lstmInfo = { backend: st.backend, action: "fine-tuned", epochsRun: st.model.epochsRun, trainedThrough: dateAt(s, asOf), backtestVersion: st.backtestVersion };
      } else {
        disposeLstm(st?.model);
        this.lstmState = null;
        const first = origins[0] as number;
        const model = await trainLstm(s, buildHistory(s, first), first, this.settings.lstm, lstmProgress("Training LSTM", 0.1, 0.5), stale);
        try {
          check();
          const folds: FoldForecast[] = [fold(first, forecastLstm(model, s, buildHistory(s, first), H))];
          for (let k = 1; k < allOrigins.length; k++) {
            const origin = allOrigins[k] as number;
            const h = buildHistory(s, origin);
            await fineTuneLstm(model, s, h, origin, lstmProgress(`LSTM rolling update ${k}/${allOrigins.length - 1}`, 0.6 + (0.3 * (k - 1)) / (allOrigins.length - 1), 0.3 / (allOrigins.length - 1)), stale);
            check();
            const f = forecastLstm(model, s, h, H);
            if (origin === asOf) models.lstm = { forecast: f, folds, metrics: foldMetrics(folds, naiveScale) };
            else folds.push(fold(origin, f));
          }
          const issued = new Map<number, Float64Array>(folds.map((f) => [f.origin, f.forecast]));
          if (models.lstm) issued.set(asOf, models.lstm.forecast);
          this.lstmState = { key, model, issued, backtestVersion: this.version + 1, backend };
          lstmInfo = { backend, action: "trained", epochsRun: model.epochsRun, trainedThrough: dateAt(s, asOf), backtestVersion: this.version + 1 };
        } catch (e) {
          disposeLstm(model);
          throw e;
        }
      }
      tick("lstm", t0);
    }

    // --- Adaptive ensemble, intervals, errors, drivers ------------------------------------------
    check();
    report("Combining models and calibrating intervals", 0.95);
    const active = MODEL_IDS.filter((id) => models[id]);
    const weights = inverseWapeWeights(Object.fromEntries(active.map((id) => [id, (models[id] as ModelRun).metrics.wape])));
    const ens = ensembleFolds(Object.fromEntries(active.map((id) => [id, (models[id] as ModelRun).folds])), naiveScale);
    const ensForecast = combine(Object.fromEntries(active.map((id) => [id, (models[id] as ModelRun).forecast])), weights);
    const intervals = fitIntervals(ens.folds, H);
    const foldErrors = ens.folds.map((f) => f.actual.map((a, k) => a - (f.forecast[k] as number)));

    this.version++;
    report("Done", 1);
    return {
      version: this.version,
      seriesId: s.id,
      source: s.source,
      asOf,
      asOfDate: dateAt(s, asOf),
      start: asOf + 1,
      dates: Array.from({ length: H }, (_, k) => dateAt(s, asOf + 1 + k)),
      horizon: H,
      models,
      baselines: {
        ma28: { forecast: baseFinal.ma28, folds: baseFolds.ma28, metrics: foldMetrics(baseFolds.ma28, naiveScale) },
        snaive: { forecast: baseFinal.snaive, folds: baseFolds.snaive, metrics: foldMetrics(baseFolds.snaive, naiveScale) },
      },
      ensemble: { forecast: ensForecast, folds: ens.folds, metrics: foldMetrics(ens.folds, naiveScale), weights },
      bands: applyIntervals(ensForecast, intervals),
      foldErrors,
      naiveScale,
      drivers: ridgeModel ? drivers(ridgeModel) : emptyDrivers(),
      lstm: lstmInfo,
      timingsMs: timings,
    };
  }

  dispose(): void {
    disposeLstm(this.lstmState?.model);
    this.lstmState = null;
  }
}

/** Driver effects from the final Ridge fit (effects on sales, holding recent sales level fixed). */
function drivers(m: RidgeModel): Drivers {
  const eff = (name: string, delta = 1) => Math.exp(rawCoefficient(m, name) * delta) - 1;
  const days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
  return {
    weekday: days.map((d, i) => (i === 0 ? 1 : Math.exp(rawCoefficient(m, `dow_${d}`)))),
    promoPerDoubling: eff("log_promo", Math.log(2)),
    holiday: eff("holiday"),
    event: eff("event"),
    earthquake: eff("quake"),
    payday: eff("payday"),
    christmasEve: eff("xmas"),
    oilPer10: eff("oil_lag7", 10),
  };
}

function emptyDrivers(): Drivers {
  return { weekday: [1, 1, 1, 1, 1, 1, 1], promoPerDoubling: 0, holiday: 0, event: 0, earthquake: 0, payday: 0, christmasEve: 0, oilPer10: 0 };
}
