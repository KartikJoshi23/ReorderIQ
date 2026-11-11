/** Derived values computed on the main thread (cheap): baseline vs AI policies, EOQ, sensitivity. */
import { useMemo } from "react";
import { economicOrderQuantity } from "../inventory/costs";
import { type AiPolicy, aiPolicy, type BaselinePolicy, baselinePolicy, dynamicPath, type DynamicPath } from "../inventory/safetyStock";
import { ropHeatmap, tornado } from "../optimize/inventoryOpt";
import { policyParams, useApp } from "./store";

export interface Policies {
  baseline: BaselinePolicy;
  ai: AiPolicy;
  path: DynamicPath;
  eoq: number;
}

export function usePolicies(): Policies | null {
  const series = useApp((s) => s.series);
  const asOf = useApp((s) => s.asOf);
  const forecast = useApp((s) => s.forecast);
  const serviceLevel = useApp((s) => s.serviceLevel);
  const leadTime = useApp((s) => s.leadTime);
  const costs = useApp((s) => s.costs);
  return useMemo(() => {
    if (!series || !forecast) return null;
    try {
      const p = policyParams({ serviceLevel, leadTime });
      const ai = aiPolicy(forecast, p);
      return { baseline: baselinePolicy(series, asOf, p), ai, path: dynamicPath(forecast, p), eoq: economicOrderQuantity(ai.dBar, costs) };
    } catch {
      return null;
    }
  }, [series, asOf, forecast, serviceLevel, leadTime, costs]);
}

export function useSensitivity() {
  const pol = usePolicies();
  return useMemo(() => {
    if (!pol) return null;
    const a = pol.ai;
    const base = { serviceLevel: a.serviceLevel, dBar: a.dBar, sigmaD: a.sigmaD, leadTime: a.leadTime, sigmaL: a.sigmaL };
    const sls = [0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995];
    const sigmaLs = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5];
    return { tornado: tornado(base), heatmap: { sls, sigmaLs, values: ropHeatmap(base, sls, sigmaLs) } };
  }, [pol]);
}
