/** Main-thread handle to a simulation worker (Monte Carlo, policy simulation, Bayesian optimisation). */
import * as Comlink from "comlink";
import type { SimWorkerApi } from "../workers/sim.worker";

export type SimClient = Comlink.Remote<SimWorkerApi>;

/** Each call creates a dedicated worker; the app keeps two (risk, optimiser) for its lifetime. */
export function createSimClient(): SimClient {
  const worker = new Worker(new URL("../workers/sim.worker.ts", import.meta.url), { type: "module", name: "simulation" });
  return Comlink.wrap<SimWorkerApi>(worker);
}
