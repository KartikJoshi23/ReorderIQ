/**
 * Nelder–Mead simplex minimiser (derivative-free). Used to fit Holt-Winters smoothing parameters
 * and Gaussian-process hyperparameters. Standard coefficients: reflection 1, expansion 2,
 * contraction 0.5, shrink 0.5.
 */
export interface NelderMeadOptions {
  /** Initial simplex step per coordinate (default 0.5). */
  step?: number | number[];
  maxIterations?: number;
  /** Stop when the spread of function values in the simplex falls below this. */
  tolF?: number;
  /** Stop when the simplex diameter falls below this. */
  tolX?: number;
}

export interface NelderMeadResult {
  x: number[];
  fx: number;
  iterations: number;
  evaluations: number;
  converged: boolean;
}

export function nelderMead(f: (x: number[]) => number, x0: number[], opts: NelderMeadOptions = {}): NelderMeadResult {
  const n = x0.length;
  const maxIt = opts.maxIterations ?? 200 * n;
  const tolF = opts.tolF ?? 1e-10;
  const tolX = opts.tolX ?? 1e-8;
  let evaluations = 0;
  const fx = (x: number[]) => {
    evaluations++;
    const v = f(x);
    return Number.isFinite(v) ? v : Number.POSITIVE_INFINITY;
  };

  const steps = Array.isArray(opts.step) ? opts.step : new Array<number>(n).fill(opts.step ?? 0.5);
  let simplex: { x: number[]; f: number }[] = [{ x: [...x0], f: fx(x0) }];
  for (let i = 0; i < n; i++) {
    const x = [...x0];
    x[i] = (x[i] as number) + (steps[i] ?? 0.5);
    simplex.push({ x, f: fx(x) });
  }

  let it = 0;
  let converged = false;
  for (; it < maxIt; it++) {
    simplex.sort((a, b) => a.f - b.f);
    const best = simplex[0] as { x: number[]; f: number };
    const worst = simplex[n] as { x: number[]; f: number };
    const second = simplex[n - 1] as { x: number[]; f: number };

    const fSpread = Math.abs(worst.f - best.f);
    let diameter = 0;
    for (const p of simplex) for (let j = 0; j < n; j++) diameter = Math.max(diameter, Math.abs((p.x[j] as number) - (best.x[j] as number)));
    if (fSpread <= tolF * (1 + Math.abs(best.f)) && diameter <= tolX * (1 + norm(best.x))) {
      converged = true;
      break;
    }

    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] = (centroid[j] as number) + ((simplex[i] as { x: number[] }).x[j] as number) / n;
    const along = (t: number) => centroid.map((c, j) => c + t * ((worst.x[j] as number) - c));

    const xr = along(-1);
    const fr = fx(xr);
    if (fr < best.f) {
      const xe = along(-2);
      const fe = fx(xe);
      simplex[n] = fe < fr ? { x: xe, f: fe } : { x: xr, f: fr };
    } else if (fr < second.f) {
      simplex[n] = { x: xr, f: fr };
    } else {
      const outside = fr < worst.f;
      const xc = along(outside ? -0.5 : 0.5);
      const fc = fx(xc);
      if (fc < (outside ? fr : worst.f)) {
        simplex[n] = { x: xc, f: fc };
      } else {
        simplex = simplex.map((p, i) => {
          if (i === 0) return p;
          const x = p.x.map((v, j) => (best.x[j] as number) + 0.5 * (v - (best.x[j] as number)));
          return { x, f: fx(x) };
        });
      }
    }
  }
  simplex.sort((a, b) => a.f - b.f);
  const best = simplex[0] as { x: number[]; f: number };
  return { x: best.x, fx: best.f, iterations: it, evaluations, converged };
}

function norm(x: number[]): number {
  return Math.sqrt(x.reduce((s, v) => s + v * v, 0));
}

export const logistic = (z: number) => 1 / (1 + Math.exp(-z));
export const logit = (p: number) => Math.log(p / (1 - p));
