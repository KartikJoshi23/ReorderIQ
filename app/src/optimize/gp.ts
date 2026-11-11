/**
 * Gaussian-process regression — the surrogate model inside Bayesian optimisation.
 *
 *   kernel  k(x, x') = σ_f² (1 + √5 r + 5r²/3) e^{−√5 r},  r² = Σ_d ((x_d − x'_d) / ℓ_d)²   (Matérn ν = 5/2, ARD)
 *   noise   σ_n² on the diagonal (simulation noise)
 *   fit     L = chol(K + σ_n² I),  α = K⁻¹ y
 *   predict μ(x*) = k*ᵀ α,  σ²(x*) = σ_f² − ‖L⁻¹ k*‖²   (latent function, noise excluded)
 *
 * Identical to scikit-learn GaussianProcessRegressor(C·Matern(ν=2.5) + WhiteKernel) with fixed
 * hyperparameters (verified by tests). `fitGpAuto` standardises y and chooses ℓ, σ_f², σ_n² by
 * maximising the log marginal likelihood (Nelder–Mead on log-parameters, several starts).
 */
import { cholesky, solveLower, solveUpperT } from "../lib/linalg";
import { nelderMead } from "../lib/optimize";

export interface GpHyper {
  lengthscales: number[];
  signalVar: number;
  noiseVar: number;
}

export interface GpModel {
  X: number[][];
  hyper: GpHyper;
  L: Float64Array;
  alpha: Float64Array;
  n: number;
  /** Output standardisation (μ, σ) applied before fitting; predictions are returned in original units. */
  yMean: number;
  yScale: number;
  logMarginalLikelihood: number;
}

const SQRT5 = Math.sqrt(5);

export function matern52(a: readonly number[], b: readonly number[], h: GpHyper): number {
  let r2 = 0;
  for (let d = 0; d < a.length; d++) {
    const z = ((a[d] as number) - (b[d] as number)) / (h.lengthscales[d] as number);
    r2 += z * z;
  }
  const r = Math.sqrt(r2);
  return h.signalVar * (1 + SQRT5 * r + (5 * r2) / 3) * Math.exp(-SQRT5 * r);
}

/** Fit with fixed hyperparameters. y is used as given when `standardize` is false (zero prior mean). */
export function fitGp(X: number[][], y: readonly number[], hyper: GpHyper, standardize = false): GpModel {
  const n = X.length;
  let yMean = 0;
  let yScale = 1;
  if (standardize && n > 1) {
    yMean = y.reduce((a, b) => a + b, 0) / n;
    yScale = Math.sqrt(y.reduce((a, b) => a + (b - yMean) ** 2, 0) / n) || 1;
  }
  const ys = Float64Array.from(y, (v) => (v - yMean) / yScale);
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const k = matern52(X[i] as number[], X[j] as number[], hyper) + (i === j ? hyper.noiseVar : 0);
      K[i * n + j] = k;
      K[j * n + i] = k;
    }
  }
  const L = cholesky(K, n);
  const alpha = solveUpperT(L, solveLower(L, ys, n), n);
  let logDet = 0;
  let fit = 0;
  for (let i = 0; i < n; i++) {
    logDet += Math.log(L[i * n + i] as number);
    fit += (ys[i] as number) * (alpha[i] as number);
  }
  return { X, hyper, L, alpha, n, yMean, yScale, logMarginalLikelihood: -0.5 * fit - logDet - 0.5 * n * Math.log(2 * Math.PI) };
}

export interface GpPrediction {
  mean: number;
  /** Variance of the latent function (noise excluded), original units². */
  variance: number;
  sd: number;
}

export function predictGp(m: GpModel, x: readonly number[]): GpPrediction {
  const ks = new Float64Array(m.n);
  let mu = 0;
  for (let i = 0; i < m.n; i++) {
    ks[i] = matern52(x, m.X[i] as number[], m.hyper);
    mu += (ks[i] as number) * (m.alpha[i] as number);
  }
  const v = solveLower(m.L, ks, m.n);
  let vv = 0;
  for (let i = 0; i < m.n; i++) vv += (v[i] as number) ** 2;
  const variance = Math.max(1e-12, m.hyper.signalVar - vv) * m.yScale * m.yScale;
  return { mean: m.yMean + m.yScale * mu, variance, sd: Math.sqrt(variance) };
}

/** Bounds on log-hyperparameters for inputs scaled to [0, 1] and standardised outputs. */
const BOUNDS = { logLs: [Math.log(0.03), Math.log(5)], logSig: [Math.log(0.05), Math.log(20)], logNoise: [Math.log(1e-6), Math.log(0.5)] } as const;

const clampPenalty = (v: number, [lo, hi]: readonly [number, number]) => (v < lo ? (lo - v) ** 2 : v > hi ? (v - hi) ** 2 : 0);

/** Fit with hyperparameters chosen by maximising the log marginal likelihood (standardised y). */
export function fitGpAuto(X: number[][], y: readonly number[], starts: number[][] = [[0.3, 1, 1e-3], [0.1, 1, 1e-2], [1, 1, 1e-4]]): GpModel {
  const d = (X[0] as number[]).length;
  const decode = (p: number[]): GpHyper => ({
    lengthscales: p.slice(0, d).map(Math.exp),
    signalVar: Math.exp(p[d] as number),
    noiseVar: Math.exp(p[d + 1] as number),
  });
  const objective = (p: number[]) => {
    let pen = 0;
    for (let i = 0; i < d; i++) pen += clampPenalty(p[i] as number, BOUNDS.logLs);
    pen += clampPenalty(p[d] as number, BOUNDS.logSig) + clampPenalty(p[d + 1] as number, BOUNDS.logNoise);
    if (pen > 0) return 1e6 * (1 + pen);
    try {
      return -fitGp(X, y, decode(p), true).logMarginalLikelihood;
    } catch {
      return 1e9;
    }
  };
  let best: { p: number[]; f: number } | null = null;
  for (const [ls, sig, noise] of starts) {
    const x0 = [...new Array<number>(d).fill(Math.log(ls as number)), Math.log(sig as number), Math.log(noise as number)];
    const r = nelderMead(objective, x0, { step: 0.7, maxIterations: 150 * (d + 2), tolF: 1e-7 });
    if (!best || r.fx < best.f) best = { p: r.x, f: r.fx };
  }
  return fitGp(X, y, decode((best as { p: number[] }).p), true);
}
