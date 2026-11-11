import { describe, expect, it } from "vitest";
import { cholesky, normalEquations, solveSpd } from "../../src/lib/linalg";
import { logistic, logit, nelderMead } from "../../src/lib/optimize";
import { createRng } from "../../src/lib/random";

describe("linear algebra", () => {
  it("solves a random SPD system (A = BᵀB + I) to machine precision", () => {
    const n = 8;
    const rng = createRng(5);
    const B = Float64Array.from({ length: n * n }, () => rng.normal());
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = i === j ? 1 : 0;
        for (let k = 0; k < n; k++) s += (B[k * n + i] as number) * (B[k * n + j] as number);
        A[i * n + j] = s;
      }
    const xTrue = Float64Array.from({ length: n }, (_, i) => i - 3);
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) b[i] = (b[i] as number) + (A[i * n + j] as number) * (xTrue[j] as number);
    const x = solveSpd(A, b, n);
    xTrue.forEach((v, i) => expect(x[i]).toBeCloseTo(v, 9));
  });

  it("adds jitter for a singular (positive semi-definite) matrix", () => {
    const L = cholesky(Float64Array.from([1, 1, 1, 1]), 2);
    expect(Number.isFinite(L[3] as number)).toBe(true);
  });

  it("forms XᵀX and Xᵀy", () => {
    const { xtx, xty } = normalEquations(Float64Array.from([1, 2, 3, 4]), Float64Array.from([1, 1]), 2, 2);
    expect(Array.from(xtx)).toEqual([10, 14, 14, 20]);
    expect(Array.from(xty)).toEqual([4, 6]);
  });
});

describe("Nelder–Mead", () => {
  it("finds the Rosenbrock minimum (1, 1)", () => {
    const r = nelderMead((x) => 100 * ((x[1] as number) - (x[0] as number) ** 2) ** 2 + (1 - (x[0] as number)) ** 2, [-1.2, 1], { maxIterations: 2000, tolF: 1e-14, tolX: 1e-10 });
    expect(r.x[0]).toBeCloseTo(1, 4);
    expect(r.x[1]).toBeCloseTo(1, 4);
    expect(r.converged).toBe(true);
  });

  it("treats non-finite objective values as +∞", () => {
    const r = nelderMead((x) => ((x[0] as number) < 0 ? Number.NaN : ((x[0] as number) - 2) ** 2), [1]);
    expect(r.x[0]).toBeCloseTo(2, 5);
  });

  it("logit and logistic are inverses", () => {
    for (const p of [0.01, 0.3, 0.5, 0.99]) expect(logistic(logit(p))).toBeCloseTo(p, 12);
  });
});
