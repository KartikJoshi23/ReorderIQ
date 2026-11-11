/**
 * Standard normal distribution to full double precision, with no approximation constants to trust:
 *   erfc  — power series for |x| < 2.5, continued fraction (backward recursion) beyond;
 *   Φ⁻¹   — Abramowitz & Stegun 26.2.23 starting value refined by Halley's method on the tail
 *           probability (cubic convergence → |error| ≈ 1e-15).
 * Z = Φ⁻¹(service level) is the slide's "service level factor".
 */
const SQRT_PI = Math.sqrt(Math.PI);
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/** Complementary error function. */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  if (x < 0) return 2 - erfc(-x);
  if (x < 2.5) {
    // erf(x) = 2/√π · e^{−x²} · Σ_{n≥0} 2ⁿ x^{2n+1} / (1·3·5···(2n+1))   (all terms positive)
    const x2 = x * x;
    let term = x;
    let sum = x;
    for (let n = 1; n < 200; n++) {
      term *= (2 * x2) / (2 * n + 1);
      sum += term;
      if (term < sum * 1e-17) break;
    }
    return 1 - (2 / SQRT_PI) * Math.exp(-x2) * sum;
  }
  if (x > 27) return 0; // below the smallest double
  // erfc(x) = e^{−x²}/√π · 1 / (x + (1/2)/(x + 1/(x + (3/2)/(x + …))))
  let t = x;
  for (let n = 60; n >= 1; n--) t = x + n / 2 / t;
  return Math.exp(-x * x) / (SQRT_PI * t);
}

export function pdf(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

/** Φ(x) = P(Z ≤ x). */
export function cdf(x: number): number {
  return 0.5 * erfc(-x / Math.SQRT2);
}

/** Upper tail Q(x) = P(Z > x), accurate far into the tail. */
export function upperTail(x: number): number {
  return 0.5 * erfc(x / Math.SQRT2);
}

/** Φ⁻¹(p): the z with Φ(z) = p. */
export function inverseCdf(p: number): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return Number.NaN;
  if (p === 0) return Number.NEGATIVE_INFINITY;
  if (p === 1) return Number.POSITIVE_INFINITY;
  const upper = p > 0.5;
  const q = upper ? 1 - p : p; // tail probability ≤ 0.5; solve Q(x) = q for x ≥ 0
  const t = Math.sqrt(-2 * Math.log(q));
  let x = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
  for (let i = 0; i < 10; i++) {
    const u = (upperTail(x) - q) / pdf(x); // Newton step for f(x) = Q(x) − q
    const dx = u / (1 - (x * u) / 2); // Halley correction
    x += dx;
    if (Math.abs(dx) <= 1e-15 * Math.max(1, Math.abs(x))) break;
  }
  return upper ? x : -x;
}

/** Standard normal loss function G(z) = E[(Z − z)⁺] = φ(z) − z·Q(z) (expected shortage per σ). */
export function lossFunction(z: number): number {
  return pdf(z) - z * upperTail(z);
}
