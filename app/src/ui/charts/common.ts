/** Shared ECharts option fragments (axes, tooltips, series styles) built on the design tokens. */
import { fmt } from "../../lib/format";
import { C, wash } from "../tokens";

export const unitsAxis = (name?: string) => ({
  type: "value" as const,
  name,
  nameLocation: "end" as const,
  nameGap: 10,
  axisLabel: { formatter: (v: number) => fmt.axis(v) },
  scale: false,
});

export const dateAxis = (dates: string[]) => ({
  type: "category" as const,
  data: dates,
  boundaryGap: false,
  axisLabel: { formatter: (v: string) => fmt.shortDate(v), hideOverlap: true },
});

export const axisTooltip = (valueFmt: (v: number) => string = fmt.int) => ({
  trigger: "axis" as const,
  axisPointer: { type: "line" as const, lineStyle: { color: C.axis, width: 1 } },
  valueFormatter: (v: unknown) => (typeof v === "number" ? valueFmt(v) : Array.isArray(v) ? v.map((x) => (typeof x === "number" ? valueFmt(x) : "—")).join(" – ") : "—"),
});

export const line = (name: string, data: (number | null)[], color: string, opts: { dashed?: boolean; width?: number; area?: boolean; smooth?: boolean; z?: number } = {}) => ({
  name,
  type: "line" as const,
  data,
  showSymbol: false,
  symbolSize: 8,
  smooth: opts.smooth ?? false,
  z: opts.z ?? 3,
  lineStyle: { color, width: opts.width ?? 2, type: opts.dashed ? ("dashed" as const) : ("solid" as const) },
  itemStyle: { color, borderColor: C.panel, borderWidth: 2 },
  areaStyle: opts.area ? { color: wash(color, 0.1) } : undefined,
  emphasis: { focus: "series" as const },
  connectNulls: false,
});

/** Prediction band drawn as a stacked pair (lower invisible + width filled) so it sits behind lines. */
export const band = (name: string, lo: (number | null)[], hi: (number | null)[], color: string, alpha: number) => [
  { name: `${name}-base`, type: "line" as const, data: lo, stack: name, symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false }, silent: true, z: 1 },
  {
    name,
    type: "line" as const,
    data: hi.map((h, i) => (h === null || lo[i] === null ? null : h - (lo[i] as number))),
    stack: name,
    symbol: "none",
    lineStyle: { opacity: 0 },
    areaStyle: { color: wash(color, alpha) },
    tooltip: { show: false },
    silent: true,
    z: 1,
  },
];

export const barStyle = (color: string) => ({ color, borderRadius: [4, 4, 0, 0] });
export const hbarStyle = (color: string) => ({ color, borderRadius: [0, 4, 4, 0] });

export const vLine = (x: number | string, label: string, color: string, dashed = true) => ({
  xAxis: x,
  label: { formatter: label, color: C.ink2, fontSize: 11, position: "insideEndTop" as const },
  lineStyle: { color, width: 2, type: dashed ? ("dashed" as const) : ("solid" as const) },
});
