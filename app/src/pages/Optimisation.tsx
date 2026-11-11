/** Optimisation & Sensitivity: constraints and costs, live Bayesian-optimisation convergence, the
 *  Gaussian-process surrogate, grid validation, tornado, service-level × σ_L heatmap, cost–service
 *  frontier and the shadow price of storage (slide step 3). */
import { Grid3x3, LineChart, Play } from "lucide-react";
import { useMemo } from "react";
import { fmt } from "../lib/format";
import { useSensitivity, usePolicies } from "../state/selectors";
import { useApp } from "../state/store";
import { line } from "../ui/charts/common";
import { Badge, Button, Card, ChartCard, Empty, Explain, Legend, NumberField, Stat, Status } from "../ui/components";
import { BLUE_RAMP, C } from "../ui/tokens";

export function Optimisation() {
  const constraints = useApp((s) => s.constraints);
  const costs = useApp((s) => s.costs);
  const setConstraints = useApp((s) => s.setConstraints);
  const setCosts = useApp((s) => s.setCosts);
  const its = useApp((s) => s.boIterations);
  const bo = useApp((s) => s.bo);
  const status = useApp((s) => s.boStatus);
  const grid = useApp((s) => s.grid);
  const gridStatus = useApp((s) => s.gridStatus);
  const frontier = useApp((s) => s.frontier);
  const frontierStatus = useApp((s) => s.frontierStatus);
  const shadow = useApp((s) => s.shadow);
  const sim = useApp((s) => s.sim);
  const run = useApp((s) => s.runOptimisation);
  const runGrid = useApp((s) => s.runGridCheck);
  const runFrontier = useApp((s) => s.runFrontier);
  const pol = usePolicies();
  const sens = useSensitivity();
  const D = pol?.ai.dBar ?? 1;

  const convOption = useMemo(() => ({
    grid: { top: 16, left: 8, right: 16, bottom: 28, containLabel: true },
    tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => (Array.isArray(v) ? fmt.usd(v[1] as number) : fmt.usd(v as number)) },
    xAxis: { type: "value" as const, name: "simulation #", nameLocation: "middle" as const, nameGap: 24, min: 0, max: Math.max(38, its.length) },
    yAxis: { type: "value" as const, scale: true, axisLabel: { formatter: (v: number) => `$${fmt.axis(v)}` } },
    series: [
      { name: "Feasible policy", type: "scatter" as const, symbolSize: 9, itemStyle: { color: C.ai, borderColor: C.panel, borderWidth: 2 }, data: its.filter((i) => i.feasible).map((i) => [i.index + 1, i.objective]) },
      { name: "Infeasible policy", type: "scatter" as const, symbolSize: 8, itemStyle: { color: "transparent", borderColor: "#94a3b8", borderWidth: 1.5 }, data: its.filter((i) => !i.feasible).map((i) => [i.index + 1, i.objective]) },
      { name: "Best feasible so far", type: "line" as const, step: "end" as const, showSymbol: false, lineStyle: { color: C.s4, width: 2 }, data: its.filter((i) => Number.isFinite(i.bestObjective)).map((i) => [i.index + 1, i.bestObjective]), markLine: grid?.best ? { symbol: "none", data: [{ yAxis: grid.best.objective, label: { formatter: "900-point grid optimum", color: C.ink2, fontSize: 11 }, lineStyle: { color: C.base, type: "dashed" as const, width: 1.5 } }] } : undefined },
    ],
  }), [its, grid]);

  const surfaceOption = useMemo(() => {
    const sf = bo?.surface;
    if (!sf) return null;
    const xs = sf.x0.map((v) => (v / D).toFixed(1));
    const ys = sf.x1.map((v) => (v / D).toFixed(1));
    const feasible: [number, number, number][] = [];
    const infeasible: [number, number, number][] = [];
    sf.mean.forEach((row, j) => row.forEach((m, i) => ((sf.pFeasible[j]?.[i] ?? 0) >= 0.5 ? feasible : infeasible).push([i, j, m])));
    const near = (arr: number[], v: number) => arr.reduce((b, a, i) => (Math.abs(a - v) < Math.abs((arr[b] as number) - v) ? i : b), 0);
    const pts = its.map((i) => [near(sf.x0, i.x[0] as number), near(sf.x1, i.x[1] as number), i.objective]);
    const best = bo?.best ? [[near(sf.x0, bo.best.x[0] as number), near(sf.x1, bo.best.x[1] as number), bo.best.objective]] : [];
    const vals = feasible.map((f) => f[2]);
    return {
      grid: { top: 16, left: 8, right: 72, bottom: 36, containLabel: true },
      tooltip: { trigger: "item" as const, formatter: (p: { value: number[]; seriesName: string }) => `buffer ${xs[p.value[0] as number]} d · Q ${ys[p.value[1] as number]} d<br/><b>${fmt.usd(p.value[2])}/day</b> ${p.seriesName === "Infeasible region" ? "(likely infeasible)" : ""}` },
      xAxis: { type: "category" as const, data: xs, name: "buffer b (days of demand)", nameLocation: "middle" as const, nameGap: 24, axisLabel: { interval: 7 } },
      yAxis: { type: "category" as const, data: ys, name: "order qty Q (days)", nameLocation: "end" as const, axisLabel: { interval: 7 } },
      // Every heatmap layer needs its own visualMap: blue ramp for feasible cost, flat grey for the infeasible region.
      visualMap: [
        { seriesIndex: 0, min: Math.min(...vals), max: Math.max(...vals), calculable: false, orient: "vertical" as const, right: 0, top: 16, itemHeight: 140, text: ["high", "low"], textStyle: { color: C.ink3 }, inRange: { color: [...BLUE_RAMP].reverse() } },
        { seriesIndex: 1, show: false, min: -1e12, max: 1e12, inRange: { color: ["rgba(148,163,184,0.10)", "rgba(148,163,184,0.10)"] } },
      ],
      series: [
        { name: "Expected cost", type: "heatmap" as const, data: feasible, emphasis: { disabled: true } },
        { name: "Infeasible region", type: "heatmap" as const, data: infeasible, emphasis: { disabled: true } },
        { name: "Evaluated", type: "scatter" as const, data: pts, symbolSize: 7, itemStyle: { color: "#ffffff", borderColor: C.panel, borderWidth: 1.5 } },
        { name: "Optimum", type: "scatter" as const, data: best, symbol: "pin", symbolSize: 26, itemStyle: { color: C.s4 } },
      ],
    };
  }, [bo, its, D]);

  const tornadoOption = useMemo(() => {
    if (!sens) return null;
    const bars = [...sens.tornado.bars].reverse();
    const base = sens.tornado.baseRop;
    return {
      grid: { top: 24, left: 8, right: 24, bottom: 8, containLabel: true },
      tooltip: { trigger: "axis" as const, axisPointer: { type: "shadow" as const }, valueFormatter: (v: unknown) => `${(v as number) >= 0 ? "+" : ""}${fmt.int(v as number)} units` },
      xAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => `${v > 0 ? "+" : ""}${fmt.axis(v)}` } },
      yAxis: { type: "category" as const, data: bars.map((b) => b.label), axisLabel: { color: C.ink2, fontSize: 11.5 } },
      series: [
        { name: "Input lower (−20% / 90% service)", type: "bar" as const, stack: "t", barMaxWidth: 18, data: bars.map((b) => b.low.rop - base), itemStyle: { color: C.mc, borderRadius: 3 } },
        { name: "Input higher (+20% / 99% service)", type: "bar" as const, stack: "t", barMaxWidth: 18, data: bars.map((b) => b.high.rop - base), itemStyle: { color: C.ai, borderRadius: 3 } },
      ],
    };
  }, [sens]);

  const heatOption = useMemo(() => {
    if (!sens) return null;
    const h = sens.heatmap;
    const data: [number, number, number][] = [];
    h.values.forEach((row, j) => row.forEach((v, i) => data.push([i, j, v])));
    const vals = data.map((d) => d[2]);
    return {
      grid: { top: 16, left: 8, right: 72, bottom: 36, containLabel: true },
      tooltip: { trigger: "item" as const, formatter: (p: { value: number[] }) => `service ${fmt.pct(h.sls[p.value[0] as number], 1)} · σ_L ${h.sigmaLs[p.value[1] as number]} d<br/><b>ROP ${fmt.int(p.value[2])}</b>` },
      xAxis: { type: "category" as const, data: h.sls.map((s) => fmt.pct(s, 1)), name: "service level", nameLocation: "middle" as const, nameGap: 24 },
      yAxis: { type: "category" as const, data: h.sigmaLs.map((s) => `${s} d`), name: "σ_L", nameLocation: "end" as const },
      visualMap: { min: Math.min(...vals), max: Math.max(...vals), right: 0, top: 16, itemHeight: 140, textStyle: { color: C.ink3 }, text: ["high", "low"], inRange: { color: BLUE_RAMP } },
      series: [{ type: "heatmap" as const, data, itemStyle: { borderColor: C.panel, borderWidth: 2 } }],
    };
  }, [sens]);

  const frontierOption = useMemo(() => ({
    grid: { top: 16, left: 8, right: 16, bottom: 28, containLabel: true },
    tooltip: { trigger: "axis" as const, valueFormatter: (v: unknown) => fmt.usd(v as number) },
    xAxis: { type: "category" as const, data: frontier.map((p) => fmt.pct(p.minFillRate, 1)), name: "minimum fill rate", nameLocation: "middle" as const, nameGap: 24 },
    yAxis: { type: "value" as const, scale: true, axisLabel: { formatter: (v: number) => `$${fmt.axis(v)}` } },
    series: [{ ...line("Optimal cost per day", frontier.map((p) => p.costPerDay), C.ai), showSymbol: true }],
  }), [frontier]);

  if (!constraints || !pol) return <Empty>Waiting for the forecast…</Empty>;
  const best = bo?.best;
  const pBest = best?.details?.policy;
  const m = best?.details?.metrics;
  const margins = best?.details?.margins;
  const baseCost = sim.baseline?.costPerDay;

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Constraints" subtitle="The optimiser must satisfy all of them" actions={<Button size="sm" onClick={run} disabled={status === "running"}><Play size={13} aria-hidden /> {status === "running" ? "Optimising…" : "Re-run"}</Button>}>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Min fill rate" value={+(constraints.minFillRate * 100).toFixed(2)} step={0.5} min={80} max={99.9} suffix="%" onChange={(v) => setConstraints({ minFillRate: Math.min(0.999, Math.max(0.5, v / 100)) })} />
            <NumberField label="Storage capacity" value={Math.round(constraints.capacity)} step={1000} min={0} suffix="units" onChange={(v) => setConstraints({ capacity: Math.max(1, v) })} />
            <NumberField label="Inventory budget" value={Math.round(constraints.budget)} step={500} min={0} suffix="$" onChange={(v) => setConstraints({ budget: Math.max(1, v) })} />
            <NumberField label="Supplier MOQ" value={Math.round(constraints.moq)} step={500} min={0} suffix="units" onChange={(v) => setConstraints({ moq: Math.max(0, v) })} />
            <NumberField label="Max deliveries" value={constraints.maxOrdersPerWeek} step={0.5} min={0.5} suffix="/ week" onChange={(v) => setConstraints({ maxOrdersPerWeek: Math.max(0.5, v) })} />
          </div>
          <p className="mt-2 text-[11.5px] text-ink-3">Capacity = {fmt.one(constraints.capacity / D)} days of demand · budget = {fmt.one(constraints.budget / (D * costs.unitCost))} days at cost.</p>
        </Card>
        <Card title="Cost assumptions" subtitle="Documented defaults, adjustable">
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Unit cost" value={costs.unitCost} step={0.05} min={0.01} suffix="$" onChange={(v) => setCosts({ unitCost: Math.max(0.01, v) })} />
            <NumberField label="Holding rate" value={+(costs.holdingRatePerYear * 100).toFixed(1)} step={1} min={1} suffix="%/yr" onChange={(v) => setCosts({ holdingRatePerYear: Math.max(0.01, v / 100) })} />
            <NumberField label="Order cost" value={costs.orderCost} step={10} min={0} suffix="$/order" onChange={(v) => setCosts({ orderCost: Math.max(0, v) })} />
            <NumberField label="Lost-sale penalty" value={costs.stockoutPenalty} step={0.05} min={0} suffix="$/unit" onChange={(v) => setCosts({ stockoutPenalty: Math.max(0, v) })} />
          </div>
        </Card>
        <Card title="Optimal policy" actions={<Badge kind="ai">Bayesian optimisation</Badge>}>
          {best && pBest?.kind === "dynamic" && m && margins ? (
            <div className="grid gap-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Reorder point now" value={fmt.int(pol.ai.leadTimeDemand + pBest.buffer)} sub={`forecast + buffer ${fmt.int(pBest.buffer)} (${fmt.one(pBest.buffer / D)} d)`} />
                <Stat label="Order quantity" value={fmt.int(pBest.orderQty)} sub={`${fmt.one(pBest.orderQty / D)} days of demand`} />
                <Stat label="Cost per day" value={fmt.usd(best.objective)} sub={baseCost ? `baseline ${fmt.usd(baseCost)}` : undefined} />
                <Stat label="Fill rate" value={fmt.pct(m.fillRate, 2)} sub={`cycle service ${fmt.pct(m.cycleServiceLevel, 1)}`} />
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                <Status tone={margins.fill >= 0 ? "good" : "critical"}>fill ≥ {fmt.pct(constraints.minFillRate, 1)}</Status>
                <Status tone={margins.capacity >= 0 ? "good" : "critical"}>peak {fmt.one(m.peakOnHandP95 / D)} d ≤ capacity</Status>
                <Status tone={margins.budget >= 0 ? "good" : "critical"}>avg value {fmt.usd(m.avgInventoryValue)}</Status>
                <Status tone={margins.orders >= 0 ? "good" : "critical"}>{m.ordersPerWeek.toFixed(2)} deliveries/wk</Status>
              </div>
            </div>
          ) : status === "running" ? (
            <Empty>Searching… {its.length}/38 simulations</Empty>
          ) : (
            <Empty>No feasible policy found. Relax a constraint.</Empty>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartCard
          title="Bayesian optimisation convergence"
          subtitle="8 space-filling starts, then a Gaussian process picks each next policy (Expected Improvement × P(feasible))"
          option={convOption}
          height={300}
          ariaLabel="Scatter of evaluated policies with the best feasible cost so far"
          actions={<Button size="sm" variant="outline" onClick={runGrid} disabled={gridStatus === "running" || !bo}><Grid3x3 size={13} aria-hidden /> {gridStatus === "running" ? "Sweeping 900…" : "Verify vs grid"}</Button>}
          legend={<Legend items={[{ label: "Feasible policy", color: C.ai, kind: "dot" }, { label: "Infeasible policy", color: "#94a3b8", kind: "dot" }, { label: "Best feasible so far", color: C.s4 }, ...(grid ? [{ label: "Grid optimum", color: C.base, kind: "dash" as const }] : [])]} />}
          table={{ columns: ["#", "Phase", "Buffer (units)", "Q (units)", "Cost/day", "Feasible"], rows: its.map((i) => [i.index + 1, i.phase, fmt.int(i.x[0]), fmt.int(i.x[1]), fmt.usd(i.objective), i.feasible ? "yes" : "no"]) }}
          explain={grid?.best && best ? <>Exhaustive grid: {fmt.usd(grid.best.objective)}/day after <b className="text-ink">900</b> simulations. Bayesian optimisation: {fmt.usd(best.objective)}/day after <b className="text-ink">{bo?.evaluations}</b> ({fmt.signedPct(best.objective / grid.best.objective - 1, 2)}), the manual sweep the slide replaces at 4% of the effort.</> : <>Hollow points broke a constraint. The optimiser learns where the feasible region is and concentrates its simulations near the cheapest feasible corner.</>}
        />
        {surfaceOption ? (
          <ChartCard
            title="What the optimiser learned (GP surrogate)"
            subtitle="Predicted cost over buffer × order quantity; grey = predicted infeasible; pin = optimum"
            option={surfaceOption}
            height={300}
            ariaLabel="Heatmap of the Gaussian-process predicted cost with evaluated points and the optimum"
            explain={<>The surrogate is the AI-driven sensitivity analysis: it predicts cost everywhere from {bo?.evaluations} simulations, so trade-offs can be read without sweeping every combination.</>}
          />
        ) : (
          <Card title="GP surrogate"><Empty>Appears when the optimisation finishes.</Empty></Card>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {tornadoOption && (
          <ChartCard
            title="Which input moves the reorder point most?"
            subtitle={`Change in AI reorder point (base ${fmt.int(sens?.tornado.baseRop)}) when each input moves ±20%`}
            option={tornadoOption}
            height={260}
            ariaLabel="Tornado chart of reorder point sensitivity"
            legend={<Legend items={[{ label: "Input lower", color: C.mc, kind: "bar" }, { label: "Input higher", color: C.ai, kind: "bar" }]} />}
            table={{ columns: ["Input", "Low value", "ROP", "High value", "ROP"], rows: (sens?.tornado.bars ?? []).map((b) => [b.label, b.low.value.toFixed(2), fmt.int(b.low.rop), b.high.value.toFixed(2), fmt.int(b.high.rop)]) }}
          />
        )}
        {heatOption && (
          <ChartCard
            title="Reorder point: service level × σ_L"
            subtitle="How much stock each combination needs (slide formula, AI inputs)"
            option={heatOption}
            height={260}
            ariaLabel="Heatmap of reorder point by service level and lead-time variability"
            table={{ columns: ["σ_L \\ service", ...(sens?.heatmap.sls ?? []).map((s) => fmt.pct(s, 1))], rows: (sens?.heatmap.values ?? []).map((row, j) => [`${sens?.heatmap.sigmaLs[j]} d`, ...row.map((v) => fmt.int(v))]) }}
          />
        )}
        <ChartCard
          title="Cost–service frontier"
          subtitle="Cheapest feasible policy for each fill-rate target (one optimisation per point)"
          option={frontierOption}
          height={260}
          ariaLabel="Line chart of optimal cost against required fill rate"
          actions={<Button size="sm" variant="outline" onClick={runFrontier} disabled={frontierStatus === "running" || !bo}><LineChart size={13} aria-hidden /> {frontierStatus === "running" ? `Optimising ${frontier.length}/6…` : "Compute"}</Button>}
          table={{ columns: ["Min fill rate", "Optimal cost/day"], rows: frontier.map((p) => [fmt.pct(p.minFillRate, 1), p.costPerDay === null ? "infeasible" : fmt.usd(p.costPerDay)]) }}
          explain={shadow?.savingPerDay !== undefined && shadow?.savingPerDay !== null ? <>Shadow price of storage: <b className="text-ink">{fmt.usd(shadow.savingPerDay)}/day</b> saved if capacity grew by {fmt.int(shadow.extraCapacity)} units (+10%).</> : <>Shows what each extra point of service costs. The shadow price of storage appears after the optimisation.</>}
        />
      </div>
      <Explain>
        The tornado shows the slide formula&apos;s sensitivity instantly. The surrogate and the frontier replace manual parameter sweeps: Bayesian optimisation finds the cost-optimal reorder policy under the constraints with a few dozen simulations instead of hundreds.
      </Explain>
    </div>
  );
}
