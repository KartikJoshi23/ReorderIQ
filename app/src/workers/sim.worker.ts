/**
 * Simulation Web Worker: Monte Carlo, inventory policy simulation, Bayesian optimisation, grid
 * validation and sensitivity analysis run here so the dashboard stays responsive. Exposed via Comlink.
 */
import * as Comlink from "comlink";
import { type MonteCarloInputs, type MonteCarloResult, runMonteCarlo, runScenarios, type ScenarioResult } from "../inventory/monteCarlo";
import { buildScenarioSet, type PolicyMetrics, type PolicySpec, type PolicyTrace, type ScenarioSetInputs, simulatePolicy } from "../inventory/policySim";
import type { BoIteration, BoResult } from "../optimize/bayesOpt";
import {
  capacityShadowPrice,
  costServiceFrontier,
  type Evaluated,
  type FrontierPoint,
  gridSearch,
  type GridResult,
  type OptimizeSettings,
  optimizeInventory,
} from "../optimize/inventoryOpt";

export interface OptimizeRequest {
  scenario: ScenarioSetInputs;
  settings: OptimizeSettings;
  iterations?: number;
  seed?: number;
}

const api = {
  monteCarlo(inp: MonteCarloInputs): MonteCarloResult {
    return runMonteCarlo(inp);
  },
  scenarios(inp: Omit<MonteCarloInputs, "scenario">): ScenarioResult[] {
    return runScenarios(inp);
  },
  /** Simulate one policy; returns metrics and the day-by-day trace of path `tracePath`. */
  simulate(scenario: ScenarioSetInputs, policy: PolicySpec, settings: OptimizeSettings, tracePath = 0): PolicyMetrics & { trace?: PolicyTrace } {
    return simulatePolicy(buildScenarioSet(scenario), policy, settings.costs, tracePath);
  },
  optimize(req: OptimizeRequest, onIteration?: (it: BoIteration) => void): BoResult<Evaluated> {
    return optimizeInventory(buildScenarioSet(req.scenario), req.settings, { iterations: req.iterations, seed: req.seed, onIteration });
  },
  grid(req: OptimizeRequest, n = 30): GridResult {
    return gridSearch(buildScenarioSet(req.scenario), req.settings, n);
  },
  frontier(req: OptimizeRequest, targets?: number[], onPoint?: (p: FrontierPoint) => void): FrontierPoint[] {
    return costServiceFrontier(buildScenarioSet(req.scenario), req.settings, targets, { seed: req.seed }, onPoint);
  },
  shadowPrice(req: OptimizeRequest, baseBest: number | null) {
    return capacityShadowPrice(buildScenarioSet(req.scenario), req.settings, baseBest, { seed: req.seed });
  },
};

export type SimWorkerApi = typeof api;

Comlink.expose(api);
