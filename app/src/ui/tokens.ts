/**
 * Colour roles for charts (validated palette — see styles/index.css). Charts reference roles,
 * never raw hex, so meaning stays consistent: blue = AI, orange = baseline, aqua = Monte Carlo.
 */
export const C = {
  ai: "#3987e5",
  base: "#d95926",
  mc: "#199e70",
  s4: "#c98500",
  actual: "#cbd5e1",
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
  ink: "#e6eaf2",
  ink2: "#a9b4c7",
  ink3: "#6b7a93",
  grid: "rgba(148,163,184,0.12)",
  axis: "rgba(148,163,184,0.28)",
  panel: "#0f1522",
  accent: "#5598e7",
  aiChrome: "#9085e9",
} as const;

/** Sequential blue ramp (dark mode: low values recede toward the surface). */
export const BLUE_RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"];

/** Series hue at ~10 % opacity for area washes. */
export function wash(hex: string, alpha = 0.1): string {
  const n = Number.parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}
