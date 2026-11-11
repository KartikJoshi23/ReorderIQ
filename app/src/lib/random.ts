/**
 * Seeded pseudo-random numbers (reproducible demo data, Monte Carlo and optimisation runs).
 * Generator: xoshiro128** (period 2^128 − 1), seeded through splitmix32.
 */
export interface Rng {
  /** Uniform on [0, 1). */
  next(): number;
  /** Standard normal N(0, 1) (Box–Muller, caches the second variate). */
  normal(): number;
  /** Integer uniform on [0, n). */
  int(n: number): number;
}

function splitmix32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

export function createRng(seed: number): Rng {
  const init = splitmix32(seed);
  let a = init(), b = init(), c = init(), d = init();
  if ((a | b | c | d) === 0) a = 1; // all-zero state is invalid for xoshiro

  const nextU32 = (): number => {
    const result = Math.imul(rotl(Math.imul(b, 5) >>> 0, 7), 9) >>> 0;
    const t = (b << 9) >>> 0;
    c ^= a;
    d ^= b;
    b ^= c;
    a ^= d;
    c ^= t;
    d = rotl(d, 11);
    return result;
  };

  let spare: number | null = null;
  const next = (): number => nextU32() / 4294967296;

  return {
    next,
    int: (n: number) => Math.floor(next() * n),
    normal(): number {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0;
      while (u === 0) u = next(); // avoid log(0)
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    },
  };
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}
