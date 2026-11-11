/**
 * Thin React wrapper around Apache ECharts (canvas renderer, tree-shaken).
 * Charts resize with their container and are disposed on unmount. The "rop" theme applies the
 * dark design tokens: recessive hairline grid, text tokens for all labels, glass tooltips.
 */
import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { C } from "../tokens";

echarts.use([
  LineChart, BarChart, ScatterChart, HeatmapChart, CustomChart,
  GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent, MarkAreaComponent,
  MarkPointComponent, VisualMapComponent, AxisPointerComponent, CanvasRenderer,
]);

const FONT = '"Inter Variable", Inter, system-ui, sans-serif';

echarts.registerTheme("rop", {
  backgroundColor: "transparent",
  textStyle: { fontFamily: FONT, color: C.ink2 },
  grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
  legend: { textStyle: { color: C.ink2, fontSize: 11 }, itemWidth: 14, itemHeight: 8, icon: "roundRect", top: 0, left: 0 },
  tooltip: {
    backgroundColor: "rgba(11,18,32,0.94)",
    borderColor: "rgba(148,163,184,0.28)",
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: C.ink, fontSize: 12, fontFamily: FONT },
    extraCssText: "border-radius:10px;backdrop-filter:blur(8px);box-shadow:0 10px 30px -10px rgba(0,0,0,.6);",
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: C.axis } },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11 },
    splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11 },
    splitLine: { lineStyle: { color: C.grid, width: 1, type: "solid" } },
    nameTextStyle: { color: C.ink3, fontSize: 11 },
  },
  timeAxis: {
    axisLine: { lineStyle: { color: C.axis } },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11 },
    splitLine: { show: false },
  },
  line: { symbolSize: 8, lineStyle: { width: 2, cap: "round", join: "round" } },
});

export type ChartOption = echarts.EChartsCoreOption;

export interface EChartProps {
  option: ChartOption;
  height?: number | string;
  className?: string;
  ariaLabel: string;
}

export function EChart({ option, height = 300, className, ariaLabel }: EChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const latest = useRef(option);
  latest.current = option;

  const apply = (c: echarts.ECharts, o: ChartOption) =>
    c.setOption({ animationDuration: 500, animationEasing: "cubicOut", ...o }, { notMerge: true, lazyUpdate: true });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Initialise only once the container has a size (it can be 0×0 during page transitions).
    const ro = new ResizeObserver(([entry]) => {
      const { width, height: h } = entry?.contentRect ?? { width: 0, height: 0 };
      if (width === 0 || h === 0) return;
      if (!chart.current) {
        chart.current = echarts.init(el, "rop", { renderer: "canvas" });
        apply(chart.current, latest.current);
      } else {
        chart.current.resize({ animation: { duration: 0 } });
      }
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    if (chart.current) apply(chart.current, option);
  }, [option]);

  return <div ref={ref} role="img" aria-label={ariaLabel} className={className} style={{ height, width: "100%" }} />;
}
