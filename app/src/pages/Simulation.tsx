/** Inventory Simulation: baseline vs AI-formula vs optimised policies on the same 64 simulated
 *  13-week demand paths (common random numbers), with a day-by-day trace of one path. */
import { useMemo, useState } from "react";
import type { PolicyMetrics, PolicyTrace } from "../inventory/policySim";
import { fmt } from "../lib/format";
import { useApp } from "../state/store";
import { hbarStyle, line } from "../ui/charts/common";
import { Button, Card, ChartCard, DataTable, Empty, Explain, Legend, Slider } from "../ui/components";
import { C, wash } from "../ui/tokens";

type Key = "baseline" | "aiFormula" | "optimised";
const LABEL: Record<Key, string> = { baseline: "Baseline (static ROP, EOQ)", aiFormula: "AI formula (dynamic ROP, EOQ)", optimised: "Bayesian-optimised" };
const COLOR: Record<Key, string> = { baseline: C.base, aiFormula: C.ai, optimised: C.s4 };

export function Simulation() {
  const sim = useApp((s) => s.sim);
  const path = useApp((s) => s.simPath);
  const setPath = useApp((s) => s.setSimPath);
  const constraints = useApp((s) => s.constraints);
  const status = useApp((s) => s.boStatus);
  const [view, setView] = useState<Key>("aiFormula");
  const keys = (["baseline", "aiFormula", "optimised"] as Key[]).filter((k) => sim[k]);
  const active: Key = sim[view] ? view : keys[0] ?? "aiFormula";
  const meanDemand = (m?: PolicyMetrics & { trace?: PolicyTrace }) => (m?.trace ? m.trace.demand.reduce((a, b) => a + b, 0) / m.trace.demand.length : Number.NaN);

  const costOption = useMemo(() => {
    const rows = [...keys].reverse();
    return {
      grid: { top: 8, left: 8, right: 64, bottom: 8, containLabel: true },
      tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => fmt.usd(v as number) },
      xAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => `$${v}` } },
      yAxis: { type: "category" as const, data: rows.map((k) => LABEL[k]), axisLabel: { color: C.ink2, fontSize: 11.5 } },
      series: [{ type: "bar" as const, barMaxWidth: 22, data: rows.map((k) => ({ value: sim[k]?.costPerDay, itemStyle: hbarStyle(COLOR[k]) })), label: { show: true, position: "right" as const, color: C.ink2, formatter: (p: { value: number }) => `${fmt.usd(p.value)}/day` } }],
    };
  }, [sim, keys]);

  const traceOption = useMemo(() => {
    const m = sim[active];
    const tr = m?.trace;
    if (!tr) return null;
    const days = Array.from(tr.onHand, (_, k) => `Day ${k + 1}`);
    const arrivals = tr.orders.filter((o) => o.arrival < tr.onHand.length).map((o) => ({ coord: [o.arrival, tr.onHand[o.arrival] ?? 0], value: fmt.compact(o.qty) }));
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: { trigger: "axis" as const, valueFormatter: (v: unknown) => fmt.int(v as number) },
      xAxis: { type: "category" as const, data: days, boundaryGap: false, axisLabel: { interval: 13 } },
      yAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => fmt.axis(v) } },
      series: [
        { ...line("Stock on hand", Array.from(tr.onHand), COLOR[active], { area: true }), markArea: { silent: true, itemStyle: { color: wash("#94a3b8", 0.07) }, data: [[{ xAxis: 0, name: "warm-up" }, { xAxis: 13 }]], label: { color: C.ink3, fontSize: 11 } }, markPoint: { symbol: "triangle", symbolSize: 10, itemStyle: { color: C.ink }, label: { show: false }, data: arrivals } },
        line("Reorder level", Array.from(tr.reorderLevel), C.ink2, { dashed: true, width: 1.5 }),
        { name: "Lost sales", type: "bar" as const, barMaxWidth: 6, data: Array.from(tr.lost, (v) => (v > 0 ? v : null)), itemStyle: { color: C.critical, borderRadius: [2, 2, 0, 0] }, z: 4 },
      ],
    };
  }, [sim, active]);

  if (!keys.length) return <Empty>{status === "running" ? "Simulating policies…" : "Waiting for forecasts and optimisation…"}</Empty>;

  const table = keys.map((k) => {
    const m = sim[k] as PolicyMetrics;
    const D = meanDemand(sim[k]) || 1;
    const ok = constraints ? m.fillRate >= constraints.minFillRate && m.peakOnHandP95 <= constraints.capacity && m.avgInventoryValue <= constraints.budget && m.ordersPerWeek <= constraints.maxOrdersPerWeek : true;
    return [LABEL[k], fmt.usd(m.costPerDay), fmt.usd(m.holdingPerDay), fmt.usd(m.orderingPerDay), fmt.usd(m.stockoutPerDay), fmt.pct(m.fillRate, 2), fmt.pct(m.cycleServiceLevel, 1), fmt.pct(m.stockoutDayRate, 1), fmt.one(m.avgOnHand / D), fmt.one(m.peakOnHandP95 / D), m.ordersPerWeek.toFixed(2), ok ? "✓ all met" : "✗ violated"];
  });

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <ChartCard
          className="xl:col-span-2"
          title="Expected cost per day"
          subtitle="Holding + ordering + lost-sales penalty · same 64 simulated 13-week paths for every policy"
          option={costOption}
          height={200}
          ariaLabel="Bar chart of expected daily cost by policy"
          explain={<>Every policy faces exactly the same demand and delivery delays (common random numbers), so the differences come from the policy alone.</>}
        />
        <Card className="xl:col-span-3" title="Policy scorecard" subtitle="Stock levels in days of average demand; constraints from the Optimisation page">
          <DataTable spec={{ columns: ["Policy", "Cost/day", "Holding", "Ordering", "Lost sales", "Fill rate", "Cycle service", "Stockout days", "Avg stock (d)", "Peak stock (d)", "Orders/wk", "Constraints"], rows: table }} />
        </Card>
      </div>

      {traceOption ? (
        <ChartCard
          title="Inventory level vs reorder point, day by day"
          subtitle="One simulated path; triangles mark deliveries, red bars mark lost sales"
          option={traceOption}
          height={340}
          ariaLabel="Simulated stock on hand against the reorder level with deliveries and lost sales"
          actions={keys.map((k) => (
            <Button key={k} size="sm" variant={active === k ? "outline" : "ghost"} aria-pressed={active === k} onClick={() => setView(k)}>
              {LABEL[k]}
            </Button>
          ))}
          legend={<Legend items={[{ label: "Stock on hand", color: COLOR[active], kind: "area" }, { label: "Reorder level", color: C.ink2, kind: "dash" }, { label: "Lost sales", color: C.critical, kind: "bar" }, { label: "Delivery", color: C.ink, kind: "dot" }]} />}
          explain={<>A sawtooth: stock falls with sales, an order is placed when stock plus open orders reaches the reorder level, and the delivery lands after a random lead time. Red bars are sales lost because the shelf was empty.</>}
        />
      ) : null}
      <Card>
        <div className="max-w-md">
          <Slider label="Simulated path" value={path} min={0} max={63} step={1} onChange={setPath} format={(v) => `#${v + 1} of 64`} hint="Each path is a different 13-week demand and delivery scenario" />
        </div>
        <Explain className="mt-3">Comparing policies on many paths shows the average picture in the scorecard. Single paths show how individual stockouts happen: usually a late delivery coinciding with a busy weekend.</Explain>
      </Card>
    </div>
  );
}
