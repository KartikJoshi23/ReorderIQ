/**
 * Forecast Web Worker: all model fitting, LSTM training and backtesting run here, off the main
 * thread, so the dashboard stays responsive. Exposed to the UI through Comlink.
 */
import "./workerTiming"; // must precede every TensorFlow.js import
import * as Comlink from "comlink";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import wasmPlain from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url";
import wasmSimd from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url";
import wasmThreadedSimd from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url";
import type { SeriesData } from "../data/series";
import { type EngineProgress, ForecastEngine, type ForecastResult, type RunOptions } from "../models/engine";

// Serve the WebAssembly binaries from Vite's hashed asset URLs (same origin → works under COEP).
setWasmPaths({
  "tfjs-backend-wasm.wasm": wasmPlain,
  "tfjs-backend-wasm-simd.wasm": wasmSimd,
  "tfjs-backend-wasm-threaded-simd.wasm": wasmThreadedSimd,
});

const engine = new ForecastEngine();

// Runs are serialised: the engine keeps LSTM state between calls, so two runs must never interleave
// while one awaits TensorFlow training.
let queue: Promise<unknown> = Promise.resolve();
function enqueue<T>(job: () => Promise<T> | T): Promise<T> {
  const next = queue.then(job, job);
  queue = next.catch(() => undefined);
  return next;
}

const api = {
  run(series: SeriesData, asOf: number, opts: RunOptions, onProgress?: (p: EngineProgress) => void): Promise<ForecastResult> {
    // The cancellation generation is fixed when the run is queued, so cancel() also drops waiting runs.
    const generation = engine.currentGeneration;
    return enqueue(() => engine.run(series, asOf, onProgress, { ...opts, generation }));
  },
  /** Not queued: takes effect immediately; a training LSTM stops at the end of its current epoch. */
  cancel(): void {
    engine.cancelAll();
  },
  reset(): Promise<void> {
    return enqueue(() => engine.dispose());
  },
};

export type ForecastWorkerApi = typeof api;

Comlink.expose(api);
