/**
 * Constrained Bayesian optimisation (minimisation) — slide step 3: "instead of manual parameter
 * sweeps, use Bayesian optimization to identify optimal reorder points under constraints".
 *
 * Loop: (1) evaluate a space-filling start design (scrambled Halton); (2) fit one GP to the
 * objective and one to the feasibility margin g(x) (feasible ⇔ g ≥ 0); (3) choose the next point
 * by maximising constrained Expected Improvement  EI(x) · P(g(x) ≥ 0)  over a dense candidate set
 * (until a feasible point exists, maximise P(feasible) instead); (4) evaluate, repeat.
 *   EI(x) = σ(x) [z Φ(z) + φ(z)],  z = (f* − μ(x)) / σ(x),  f* = best feasible value so far.
 * Inputs are scaled to the unit cube; outputs are standardised inside the GP.
 */
import { createRng, type Rng } from "../lib/random";
import { cdf, pdf } from "../inventory/normal";
import { fitGpAuto, type GpModel, predictGp } from "./gp";

export interface BoEvaluation<D = unknown> {
  objective: number;
  /** Feasibility margin: ≥ 0 means all constraints are met. */
  feasibility: number;
  details?: D;
}

export interface BoProblem<D = unknown> {
  bounds: [number, number][];
  evaluate(x: number[]): BoEvaluation<D>;
}

export interface BoOptions {
  initPoints?: number;
  iterations?: number;
  seed?: number;
  candidates?: number;
  /** Grid resolution of the returned surrogate surface (2-D problems). */
  surfaceResolution?: number;
  onIteration?: (it: BoIteration) => void;
}

export interface BoIteration {
  index: number;
  phase: "initial" | "bayesian";
  x: number[];
  objective: number;
  feasibility: number;
  feasible: boolean;
  /** Best feasible objective found so far (NaN until one is found). */
  bestObjective: number;
  acquisition?: number;
}

export interface BoSurface {
  x0: number[];
  x1: number[];
  /** [i1][i0] posterior mean of the objective, its SD, and P(feasible). */
  mean: number[][];
  sd: number[][];
  pFeasible: number[][];
}

export interface BoResult<D = unknown> {
  history: (BoIteration & { details?: D })[];
  best: { x: number[]; objective: number; details?: D } | null;
  evaluations: number;
  surface?: BoSurface;
}

/** Scrambled Halton point (bases 2, 3, 5, …) with a random Cranley–Patterson shift. */
const PRIMES = [2, 3, 5, 7, 11, 13];
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}
export function haltonPoints(n: number, dim: number, rng: Rng, offset = 1): number[][] {
  const shift = Array.from({ length: dim }, () => rng.next());
  return Array.from({ length: n }, (_, i) => Array.from({ length: dim }, (_, d) => (halton(i + offset, PRIMES[d] as number) + (shift[d] as number)) % 1));
}

export function expectedImprovement(mu: number, sd: number, best: number): number {
  if (sd <= 1e-12) return Math.max(0, best - mu);
  const z = (best - mu) / sd;
  return sd * (z * cdf(z) + pdf(z));
}

export function bayesianOptimize<D>(problem: BoProblem<D>, opts: BoOptions = {}): BoResult<D> {
  const dim = problem.bounds.length;
  const rng = createRng(opts.seed ?? 42);
  const nInit = opts.initPoints ?? 4 * dim;
  const nIter = opts.iterations ?? 30;
  const nCand = opts.candidates ?? 2000;
  const toReal = (u: number[]) => u.map((v, d) => (problem.bounds[d] as [number, number])[0] + v * ((problem.bounds[d] as [number, number])[1] - (problem.bounds[d] as [number, number])[0]));

  const U: number[][] = [];
  const fs: number[] = [];
  const gs: number[] = [];
  const history: BoResult<D>["history"] = [];
  let best: BoResult<D>["best"] = null;

  const record = (u: number[], phase: BoIteration["phase"], acquisition?: number) => {
    const x = toReal(u);
    const ev = problem.evaluate(x);
    U.push(u);
    fs.push(ev.objective);
    gs.push(ev.feasibility);
    const feasible = ev.feasibility >= 0;
    if (feasible && (!best || ev.objective < best.objective)) best = { x, objective: ev.objective, details: ev.details };
    const it = { index: history.length, phase, x, objective: ev.objective, feasibility: ev.feasibility, feasible, bestObjective: best ? best.objective : Number.NaN, acquisition, details: ev.details };
    history.push(it);
    opts.onIteration?.(it);
  };

  for (const u of haltonPoints(nInit, dim, rng)) record(u, "initial");

  let fGp: GpModel | null = null;
  let gGp: GpModel | null = null;
  for (let it = 0; it < nIter; it++) {
    fGp = fitGpAuto(U, fs);
    gGp = fitGpAuto(U, gs);
    const incumbent = best ? (best as { objective: number }).objective : Number.NaN;
    const candidates = haltonPoints(nCand, dim, rng, 1 + it * nCand);
    const anchor = best ? U[fs.indexOf((best as { objective: number }).objective)] : U[gs.indexOf(Math.max(...gs))];
    for (let k = 0; k < 200 && anchor; k++) candidates.push(anchor.map((v) => Math.min(1, Math.max(0, v + 0.05 * rng.normal()))));

    let bestU: number[] | null = null;
    let bestA = -Infinity;
    for (const u of candidates) {
      if (U.some((p) => p.every((v, d) => Math.abs(v - (u[d] as number)) < 1e-6))) continue;
      const g = predictGp(gGp, u);
      const pFeas = cdf(g.mean / Math.max(g.sd, 1e-9));
      let a: number;
      if (Number.isNaN(incumbent)) a = pFeas;
      else {
        const f = predictGp(fGp, u);
        a = expectedImprovement(f.mean, f.sd, incumbent) * pFeas;
      }
      if (a > bestA) {
        bestA = a;
        bestU = u;
      }
    }
    record(bestU ?? haltonPoints(1, dim, rng, 7919 + it)[0] as number[], "bayesian", bestA);
  }

  return { history, best, evaluations: history.length, surface: dim === 2 ? surface(problem, fitGpAuto(U, fs), fitGpAuto(U, gs), opts.surfaceResolution ?? 40) : undefined };
}

function surface(problem: BoProblem<unknown>, fGp: GpModel, gGp: GpModel, res: number): BoSurface {
  const [b0, b1] = problem.bounds as [[number, number], [number, number]];
  const axis = (b: [number, number]) => Array.from({ length: res }, (_, i) => b[0] + ((b[1] - b[0]) * i) / (res - 1));
  const out: BoSurface = { x0: axis(b0), x1: axis(b1), mean: [], sd: [], pFeasible: [] };
  for (let j = 0; j < res; j++) {
    const m: number[] = [];
    const s: number[] = [];
    const p: number[] = [];
    for (let i = 0; i < res; i++) {
      const u = [i / (res - 1), j / (res - 1)];
      const f = predictGp(fGp, u);
      const g = predictGp(gGp, u);
      m.push(f.mean);
      s.push(f.sd);
      p.push(cdf(g.mean / Math.max(g.sd, 1e-9)));
    }
    out.mean.push(m);
    out.sd.push(s);
    out.pFeasible.push(p);
  }
  return out;
}
