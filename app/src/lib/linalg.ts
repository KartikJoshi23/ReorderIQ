/**
 * Small dense linear algebra (row-major Float64Array) for Ridge regression and Gaussian processes.
 */

/** Cholesky factor L (lower-triangular, row-major) of a symmetric positive-definite n×n matrix.
 *  Adds increasing diagonal jitter if the matrix is numerically indefinite. */
export function cholesky(A: Float64Array, n: number, maxJitterTries = 6): Float64Array {
  let jitter = 0;
  for (let attempt = 0; attempt <= maxJitterTries; attempt++) {
    const L = new Float64Array(n * n);
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = (A[i * n + j] as number) + (i === j ? jitter : 0);
        for (let k = 0; k < j; k++) sum -= (L[i * n + k] as number) * (L[j * n + k] as number);
        if (i === j) {
          if (!(sum > 0)) {
            ok = false;
            break;
          }
          L[i * n + i] = Math.sqrt(sum);
        } else {
          L[i * n + j] = sum / (L[j * n + j] as number);
        }
      }
    }
    if (ok) return L;
    const scale = meanDiagonal(A, n) || 1;
    jitter = jitter === 0 ? 1e-10 * scale : jitter * 10;
  }
  throw new Error("Matrix is not positive definite (Cholesky failed even with jitter)");
}

function meanDiagonal(A: Float64Array, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.abs(A[i * n + i] as number);
  return s / n;
}

/** Solve L x = b (forward substitution). */
export function solveLower(L: Float64Array, b: Float64Array, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i] as number;
    for (let k = 0; k < i; k++) s -= (L[i * n + k] as number) * (x[k] as number);
    x[i] = s / (L[i * n + i] as number);
  }
  return x;
}

/** Solve Lᵀ x = b (back substitution with the lower factor). */
export function solveUpperT(L: Float64Array, b: Float64Array, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i] as number;
    for (let k = i + 1; k < n; k++) s -= (L[k * n + i] as number) * (x[k] as number);
    x[i] = s / (L[i * n + i] as number);
  }
  return x;
}

/** Solve A x = b for SPD A via Cholesky. */
export function solveSpd(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const L = cholesky(A, n);
  return solveUpperT(L, solveLower(L, b, n), n);
}

/** XᵀX (cols×cols) and Xᵀy for row-major X (rows×cols). */
export function normalEquations(X: Float64Array, y: Float64Array, rows: number, cols: number): { xtx: Float64Array; xty: Float64Array } {
  const xtx = new Float64Array(cols * cols);
  const xty = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    const yr = y[r] as number;
    for (let i = 0; i < cols; i++) {
      const xi = X[off + i] as number;
      xty[i] = (xty[i] as number) + xi * yr;
      for (let j = 0; j <= i; j++) xtx[i * cols + j] = (xtx[i * cols + j] as number) + xi * (X[off + j] as number);
    }
  }
  for (let i = 0; i < cols; i++) for (let j = 0; j < i; j++) xtx[j * cols + i] = xtx[i * cols + j] as number;
  return { xtx, xty };
}
