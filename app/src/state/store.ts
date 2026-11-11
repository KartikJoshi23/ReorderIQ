/**
 * Application state (Zustand). Heavy work is delegated to Web Workers:
 *   forecast worker  — ETS / Ridge / LSTM, backtests, ensemble (serialised queue)
 *   risk worker      — Monte Carlo and scenario stress tests
 *   optimiser worker — Bayesian optimisation, grid validation, frontier, policy simulation
 * Every async result carries a token so results for an outdated state are discarded.
 */
import * as Comlink from "comlink";
import { create } from "zustand";
import { generateDemoPack } from "../data/demo";
import { type Pack, PackSchema } from "../data/schema";
import {
  applyObservations,
  dateAt,
  initialAsOf,
  lastObservedIndex,
  listSeries,
  type NewObservation,
  ObservationError,
  releaseDays as release,
  type SeriesData,
  type SeriesInfo,
  seriesFromPack,
} from "../data/series";
import { type Constraints, type CostParams, DEFAULT_COSTS, defaultConstraints, economicOrderQuantity } from "../inventory/costs";
import { DEFAULT_LEAD_TIME, type LeadTimeKind, type LeadTimeModel } from "../inventory/leadTime";
import { type MonteCarloResult, relativeErrors, type ScenarioResult } from "../inventory/monteCarlo";
import type { PolicyMetrics, PolicySpec, PolicyTrace, ScenarioSetInputs } from "../inventory/policySim";
import { replayPlan, type ReplayResult, runReplay } from "../inventory/replay";
import { aiPolicy, baselinePolicy } from "../inventory/safetyStock";
import type { EngineProgress, ForecastResult } from "../models/engine";
import { createForecastClient, type ForecastClient } from "../models/forecastClient";
import type { IssuedForecast } from "../models/tracking";
import type { BoIteration, BoResult } from "../optimize/bayesOpt";
import type { Evaluated, FrontierPoint, GridResult, OptimizeSettings } from "../optimize/inventoryOpt";
import { createSimClient, type SimClient } from "../inventory/simClient";

export interface VersionEntry {
  version: number;
  asOfDate: string;
  time: string;
  action: string;
  wape: number;
  ms: number;
}

export type RunStatus = "idle" | "running" | "error";

export interface SimResults {
  baseline?: PolicyMetrics & { trace?: PolicyTrace };
  aiFormula?: PolicyMetrics & { trace?: PolicyTrace };
  optimised?: PolicyMetrics & { trace?: PolicyTrace };
}

export interface AppState {
  phase: "loading" | "error" | "ready";
  loadError?: string;
  pack: Pack | null;
  source: "real" | "demo" | null;
  seriesOptions: SeriesInfo[];
  seriesId: string | null;
  series: SeriesData | null;
  asOf: number;
  dataMessage?: { tone: "good" | "critical"; text: string };

  serviceLevel: number;
  leadTime: LeadTimeModel;
  costs: CostParams;
  constraints: Constraints | null;

  forecast: ForecastResult | null;
  forecastStatus: RunStatus;
  forecastProgress: EngineProgress | null;
  forecastError?: string;
  issued: IssuedForecast[];
  versionLog: VersionEntry[];

  mc: MonteCarloResult | null;
  mcBaselineStockout: number | null;
  scenarios: ScenarioResult[] | null;
  riskStatus: RunStatus;

  boStatus: RunStatus;
  boIterations: BoIteration[];
  bo: BoResult<Evaluated> | null;
  grid: GridResult | null;
  gridStatus: RunStatus;
  frontier: FrontierPoint[];
  frontierStatus: RunStatus;
  shadow: { extraCapacity: number; savingPerDay: number | null } | null;
  sim: SimResults;
  simPath: number;

  /** Baseline vs AI replay on held-back real demand. */
  impact: ReplayResult | null;
  impactStatus: RunStatus;
  impactProgress: string | null;
  impactStale: boolean;
  /** Multiplier applied to planned promotions after the as-of date (1 = as planned). */
  promoPlan: number;

  loadReal(): Promise<void>;
  useDemo(): void;
  selectSeries(id: string): void;
  releaseDays(days: number): void;
  addObservations(rows: NewObservation[]): boolean;
  resetData(): void;
  retrainLstm(): void;
  setServiceLevel(v: number): void;
  setLeadTime(patch: Partial<LeadTimeModel>): void;
  setLeadTimeKind(kind: LeadTimeKind): void;
  setCosts(patch: Partial<CostParams>): void;
  setConstraints(patch: Partial<Constraints>): void;
  setSimPath(p: number): void;
  runOptimisation(): void;
  runGridCheck(): void;
  runFrontier(): void;
  runImpact(): void;
  setPromoPlan(multiplier: number): void;
}

// ---------------------------------------------------------------------------------------------
let forecastClient: ForecastClient | null = null;
let impactClient: ForecastClient | null = null;
let riskClient: SimClient | null = null;
let optClient: SimClient | null = null;
const fc = () => (forecastClient ??= createForecastClient());
/** Separate forecast worker for the replay study, so live updates are never queued behind it. */
const impactFc = () => (impactClient ??= createForecastClient());
const risk = () => (riskClient ??= createSimClient());
const opt = () => (optClient ??= createSimClient());

let forecastToken = 0;
let riskToken = 0;
let optToken = 0;
let impactToken = 0;

/** Copy of the series with planned promotions after `asOf` scaled by `multiplier` (what-if planning). */
function withPromoPlan(s: SeriesData, asOf: number, multiplier: number): SeriesData {
  if (multiplier === 1) return s;
  const promo = Int32Array.from(s.promo, (v, i) => (i > asOf ? Math.round(v * multiplier) : v));
  return { ...s, promo };
}
const timers: Record<string, ReturnType<typeof setTimeout>> = {};
function debounce(key: string, ms: number, fn: () => void) {
  clearTimeout(timers[key]);
  timers[key] = setTimeout(fn, ms);
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Policy inputs shared by every consumer. */
export function policyParams(s: Pick<AppState, "serviceLevel" | "leadTime">) {
  return { serviceLevel: s.serviceLevel, leadTime: s.leadTime };
}

function scenarioInputs(f: ForecastResult, leadTime: LeadTimeModel): ScenarioSetInputs {
  return { forecast: f.ensemble.forecast, relErrors: relativeErrors(f.ensemble.folds), leadTime, seed: 17 };
}

export const useApp = create<AppState>()((set, get) => {
  /** Forecast run: fast ETS + Ridge first (initial load), then the full engine with the LSTM. */
  async function runForecast(kind: "initial" | "update" | "retrain" | "whatif") {
    const token = ++forecastToken;
    const { series: stored, asOf, promoPlan } = get();
    if (!stored) return;
    const series = withPromoPlan(stored, asOf, promoPlan);
    set({ forecastStatus: "running", forecastError: undefined });
    const onProgress = Comlink.proxy((p: EngineProgress) => {
      if (token === forecastToken) set({ forecastProgress: p });
    });
    try {
      if (kind === "initial") {
        const t0 = performance.now();
        const quick = await fc().run(series, asOf, { useLstm: false }, onProgress);
        if (token !== forecastToken) return;
        accept(quick, "Fast start: Holt-Winters + Ridge", performance.now() - t0);
      }
      const t0 = performance.now();
      const full = await fc().run(series, asOf, { retrainLstm: kind === "retrain" }, onProgress);
      if (token !== forecastToken) return;
      const action =
        kind === "whatif"
          ? `Promotion plan ${get().promoPlan.toFixed(2)}× → forecasts recomputed`
          : full.lstm?.action === "fine-tuned"
            ? "New data → LSTM fine-tuned, ETS + Ridge refitted"
            : "LSTM trained (rolling backtest)";
      accept(full, action, performance.now() - t0);
      set({ forecastStatus: "idle", forecastProgress: null });
    } catch (e) {
      if (token === forecastToken) set({ forecastStatus: "error", forecastError: errorText(e), forecastProgress: null });
    }
  }

  function accept(f: ForecastResult, action: string, ms: number) {
    const s = get();
    const issued: IssuedForecast = { version: f.version, asOf: f.asOf, forecast: f.ensemble.forecast, lo80: f.bands.lo80, hi80: f.bands.hi80 };
    const meanDemand = f.ensemble.forecast.reduce((a, b) => a + b, 0) / f.ensemble.forecast.length;
    set({
      forecast: f,
      issued: [...s.issued.filter((i) => i.asOf !== f.asOf), issued],
      versionLog: [...s.versionLog, { version: f.version, asOfDate: f.asOfDate, time: new Date().toLocaleTimeString(), action, wape: f.ensemble.metrics.wape, ms }],
      constraints: s.constraints ?? defaultConstraints(meanDemand, s.costs),
    });
    scheduleRisk(0);
    scheduleOptimisation(300);
  }

  function scheduleRisk(ms = 250) {
    debounce("risk", ms, () => void runRisk());
  }

  async function runRisk() {
    const s = get();
    if (!s.forecast || !s.series) return;
    const token = ++riskToken;
    set({ riskStatus: "running" });
    try {
      const p = policyParams(s);
      const ai = aiPolicy(s.forecast, p);
      const base = baselinePolicy(s.series, s.asOf, p);
      const demand = { kind: "forecast-bootstrap" as const, forecast: s.forecast.ensemble.forecast, relErrors: relativeErrors(s.forecast.ensemble.folds) };
      const eoq = economicOrderQuantity(ai.dBar, s.costs);
      const common = { demand, leadTime: s.leadTime, serviceLevel: s.serviceLevel, orderQuantity: eoq, seed: 7 };
      const mc = await risk().monteCarlo({ ...common, reorderPoint: ai.reorderPoint, trials: 50_000 });
      const mcBase = await risk().monteCarlo({ ...common, reorderPoint: base.reorderPoint, trials: 50_000, bins: 10 });
      const scenarios = await risk().scenarios({ ...common, reorderPoint: ai.reorderPoint, trials: 20_000, bins: 30 });
      if (token !== riskToken) return;
      set({ mc, mcBaselineStockout: mcBase.stockoutProbability ?? null, scenarios, riskStatus: "idle" });
    } catch {
      if (token === riskToken) set({ riskStatus: "error" });
    }
  }

  function scheduleOptimisation(ms = 900) {
    debounce("opt", ms, () => void runOptimisation());
  }

  function optimiseSettings(s: AppState, f: ForecastResult): OptimizeSettings | null {
    if (!s.constraints) return null;
    return { mode: "dynamic", costs: s.costs, constraints: s.constraints, referenceSafetyStock: aiPolicy(f, policyParams(s)).safetyStock };
  }

  async function runOptimisation() {
    const s = get();
    if (!s.forecast || !s.series) return;
    const settings = optimiseSettings(s, s.forecast);
    if (!settings) return;
    const token = ++optToken;
    set({ boStatus: "running", boIterations: [], grid: null, frontier: [], shadow: null });
    const scenario = scenarioInputs(s.forecast, s.leadTime);
    const onIteration = Comlink.proxy((it: BoIteration) => {
      if (token === optToken) set((st) => ({ boIterations: [...st.boIterations, { ...it, details: undefined } as BoIteration] }));
    });
    try {
      const bo = await opt().optimize({ scenario, settings, seed: 9 }, onIteration);
      if (token !== optToken) return;
      set({ bo, boStatus: "idle" });
      await runSimulations(token, scenario, settings, bo.best?.details?.policy);
      const shadow = await opt().shadowPrice({ scenario, settings, seed: 9 }, bo.best?.objective ?? null);
      if (token === optToken) set({ shadow: { extraCapacity: shadow.extraCapacity, savingPerDay: shadow.savingPerDay } });
    } catch {
      if (token === optToken) set({ boStatus: "error" });
    }
  }

  async function runSimulations(token: number, scenario: ScenarioSetInputs, settings: OptimizeSettings, optimised?: PolicySpec) {
    const s = get();
    if (!s.forecast || !s.series) return;
    const p = policyParams(s);
    const ai = aiPolicy(s.forecast, p);
    const base = baselinePolicy(s.series, s.asOf, p);
    const eoq = economicOrderQuantity(ai.dBar, s.costs);
    const path = s.simPath;
    const [baseline, aiFormula, opti] = await Promise.all([
      opt().simulate(scenario, { kind: "static", reorderPoint: base.reorderPoint, orderQty: eoq }, settings, path),
      opt().simulate(scenario, { kind: "dynamic", buffer: ai.safetyStock, orderQty: eoq }, settings, path),
      optimised ? opt().simulate(scenario, optimised, settings, path) : Promise.resolve(undefined),
    ]);
    if (token === optToken) set({ sim: { baseline, aiFormula, optimised: opti } });
  }

  function loadPack(pack: Pack) {
    const options = listSeries(pack);
    set({ pack, source: pack.meta.synthetic ? "demo" : "real", seriesOptions: options, phase: "ready" });
    get().selectSeries(pack.defaultSeries);
  }

  function resetAnalyses() {
    riskToken++;
    optToken++;
    impactToken++;
    return {
      mc: null, mcBaselineStockout: null, scenarios: null, bo: null, boIterations: [], grid: null, frontier: [], shadow: null, sim: {},
      boStatus: "idle" as const, riskStatus: "idle" as const, impact: null, impactStatus: "idle" as const, impactProgress: null, impactStale: false, promoPlan: 1,
    };
  }

  /** Settings that the replay depends on changed: keep the old result visible but flag it. */
  function markImpactStale() {
    if (get().impact) set({ impactStale: true });
  }

  async function runImpact() {
    const s = get();
    if (!s.pack || !s.seriesId) return;
    const token = ++impactToken;
    const series = seriesFromPack(s.pack, s.seriesId); // as in the data pack: user-added data excluded
    const plan = replayPlan(series);
    const p = policyParams(s);
    set({ impactStatus: "running", impactStale: false, impactProgress: "Preparing replay" });
    try {
      const weekly: ForecastResult[] = [];
      for (const [k, origin] of plan.weekOrigins.entries()) {
        if (token !== impactToken) return;
        set({ impactProgress: `Re-forecasting with data up to ${dateAt(series, origin)} (week ${k + 1} of ${plan.weekOrigins.length})` });
        weekly.push(await impactFc().run(series, origin, { useLstm: false }));
      }
      const first = weekly[0] as ForecastResult;
      const meanDemand = first.ensemble.forecast.reduce((a, b) => a + b, 0) / first.ensemble.forecast.length;
      const settings: OptimizeSettings = {
        mode: "dynamic",
        costs: s.costs,
        constraints: s.constraints ?? defaultConstraints(meanDemand, s.costs),
        referenceSafetyStock: aiPolicy(first, p).safetyStock,
      };
      if (token !== impactToken) return;
      set({ impactProgress: "Optimising buffer and order quantity at the replay start" });
      const bo = await opt().optimize({ scenario: scenarioInputs(first, s.leadTime), settings, seed: 9 });
      const best = bo.best?.details?.policy;
      if (token !== impactToken) return;
      set({ impactProgress: "Replaying real demand through each policy" });
      const impact = runReplay({
        series, weekly, serviceLevel: s.serviceLevel, leadTime: s.leadTime, costs: s.costs,
        optimised: best?.kind === "dynamic" ? { buffer: best.buffer, orderQty: best.orderQty } : null,
      });
      if (token === impactToken) set({ impact, impactStatus: "idle", impactProgress: null });
    } catch (e) {
      if (token === impactToken) set({ impactStatus: "error", impactProgress: errorText(e) });
    }
  }

  return {
    phase: "loading",
    pack: null,
    source: null,
    seriesOptions: [],
    seriesId: null,
    series: null,
    asOf: 0,
    serviceLevel: 0.95,
    leadTime: DEFAULT_LEAD_TIME,
    costs: DEFAULT_COSTS,
    constraints: null,
    forecast: null,
    forecastStatus: "idle",
    forecastProgress: null,
    issued: [],
    versionLog: [],
    mc: null,
    mcBaselineStockout: null,
    scenarios: null,
    riskStatus: "idle",
    boStatus: "idle",
    boIterations: [],
    bo: null,
    grid: null,
    gridStatus: "idle",
    frontier: [],
    frontierStatus: "idle",
    shadow: null,
    sim: {},
    simPath: 0,
    impact: null,
    impactStatus: "idle",
    impactProgress: null,
    impactStale: false,
    promoPlan: 1,

    async loadReal() {
      set({ phase: "loading", loadError: undefined });
      try {
        const res = await fetch(`${import.meta.env.BASE_URL}data/favorita-pack.json`);
        if (!res.ok) throw new Error(`Could not load the Favorita data (HTTP ${res.status}).`);
        const parsed = PackSchema.safeParse(await res.json());
        if (!parsed.success) {
          const first = parsed.error.issues[0];
          throw new Error(`The data file failed validation: ${first?.path.join(".")} ${first?.message}`);
        }
        loadPack(parsed.data);
      } catch (e) {
        set({ phase: "error", loadError: errorText(e) });
      }
    },

    useDemo() {
      loadPack(generateDemoPack());
    },

    selectSeries(id) {
      const pack = get().pack;
      if (!pack) return;
      const series = seriesFromPack(pack, id);
      forecastToken++;
      set({ seriesId: id, series, asOf: initialAsOf(series), forecast: null, issued: [], versionLog: [], constraints: null, dataMessage: undefined, ...resetAnalyses() });
      // Abandon forecasts for the previous series (incl. an LSTM still training), then start fresh.
      void fc().cancel();
      void fc().reset();
      void runForecast("initial");
    },

    releaseDays(days) {
      const { series, asOf } = get();
      if (!series) return;
      const next = release(series, asOf, days);
      if (next === asOf) {
        set({ dataMessage: { tone: "critical", text: "No more held-back data — add new observations to go further." } });
        return;
      }
      set({ asOf: next, dataMessage: { tone: "good", text: `Released ${next - asOf} day(s) of actual sales up to ${dateAt(series, next)}. Models are updating…` } });
      void runForecast("update");
    },

    addObservations(rows) {
      const { series, asOf } = get();
      if (!series) return false;
      try {
        const r = applyObservations(series, asOf, rows);
        set({ series: r.series, asOf: r.asOf, dataMessage: { tone: "good", text: `Added ${r.added} new day(s)${r.revised ? `, corrected ${r.revised}` : ""}. Models are updating…` } });
        void runForecast("update");
        return true;
      } catch (e) {
        set({ dataMessage: { tone: "critical", text: e instanceof ObservationError ? e.problems.join(" · ") : errorText(e) } });
        return false;
      }
    },

    resetData() {
      const id = get().seriesId;
      if (id) get().selectSeries(id);
    },

    retrainLstm() {
      void runForecast("retrain");
    },

    setServiceLevel(v) {
      set({ serviceLevel: v });
      scheduleRisk();
      scheduleOptimisation();
      markImpactStale();
    },
    setLeadTime(patch) {
      set((s) => ({ leadTime: { ...s.leadTime, ...patch } }));
      scheduleRisk();
      scheduleOptimisation();
      markImpactStale();
    },
    setLeadTimeKind(kind) {
      set((s) => ({ leadTime: { ...s.leadTime, kind } }));
      scheduleRisk();
      scheduleOptimisation();
      markImpactStale();
    },
    setCosts(patch) {
      set((s) => ({ costs: { ...s.costs, ...patch } }));
      scheduleRisk();
      scheduleOptimisation();
      markImpactStale();
    },
    setConstraints(patch) {
      set((s) => ({ constraints: s.constraints ? { ...s.constraints, ...patch } : null }));
      scheduleOptimisation();
      markImpactStale();
    },
    setPromoPlan(multiplier) {
      set({ promoPlan: multiplier });
      debounce("promo", 500, () => void runForecast("whatif"));
    },
    runImpact() {
      void runImpact();
    },
    setSimPath(p) {
      set({ simPath: p });
      const s = get();
      const settings = s.forecast ? optimiseSettings(s, s.forecast) : null;
      if (s.forecast && settings) void runSimulations(optToken, scenarioInputs(s.forecast, s.leadTime), settings, s.bo?.best?.details?.policy);
    },

    runOptimisation() {
      void runOptimisation();
    },

    async runGridCheck() {
      const s = get();
      if (!s.forecast) return;
      const settings = optimiseSettings(s, s.forecast);
      if (!settings) return;
      const token = optToken;
      set({ gridStatus: "running" });
      try {
        const grid = await opt().grid({ scenario: scenarioInputs(s.forecast, s.leadTime), settings }, 30);
        if (token === optToken) set({ grid, gridStatus: "idle" });
      } catch {
        set({ gridStatus: "error" });
      }
    },

    async runFrontier() {
      const s = get();
      if (!s.forecast) return;
      const settings = optimiseSettings(s, s.forecast);
      if (!settings) return;
      const token = optToken;
      set({ frontier: [], frontierStatus: "running" });
      const onPoint = Comlink.proxy((p: FrontierPoint) => {
        if (token === optToken) set((st) => ({ frontier: [...st.frontier, p] }));
      });
      try {
        await opt().frontier({ scenario: scenarioInputs(s.forecast, s.leadTime), settings, seed: 9 }, undefined, onPoint);
        if (token === optToken) set({ frontierStatus: "idle" });
      } catch {
        set({ frontierStatus: "error" });
      }
    },
  };
});

/** Last day that actually has data (held-back replay days included). */
export function replayEnd(s: SeriesData | null): number {
  return s ? lastObservedIndex(s) : 0;
}
