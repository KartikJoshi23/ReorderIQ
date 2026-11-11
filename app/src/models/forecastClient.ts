/**
 * Main-thread handle to the forecast worker. Usage:
 *   const fc = createForecastClient();
 *   const result = await fc.run(series, asOf, {}, (p) => setProgress(p));
 */
import * as Comlink from "comlink";
import type { SeriesData } from "../data/series";
import type { ForecastWorkerApi } from "../workers/forecast.worker";
import type { EngineProgress, ForecastResult, RunOptions } from "./engine";

export interface ForecastClient {
  run(series: SeriesData, asOf: number, opts?: RunOptions, onProgress?: (p: EngineProgress) => void): Promise<ForecastResult>;
  /** Abandon queued and running forecasts (e.g. a different series was selected). */
  cancel(): Promise<void>;
  reset(): Promise<void>;
  terminate(): void;
}

export function createForecastClient(): ForecastClient {
  const worker = new Worker(new URL("../workers/forecast.worker.ts", import.meta.url), { type: "module", name: "forecast" });
  const remote = Comlink.wrap<ForecastWorkerApi>(worker);
  return {
    run: (series, asOf, opts = {}, onProgress) => remote.run(series, asOf, opts, onProgress ? Comlink.proxy(onProgress) : undefined),
    cancel: () => remote.cancel(),
    reset: () => remote.reset(),
    terminate: () => {
      remote[Comlink.releaseProxy]();
      worker.terminate();
    },
  };
}
