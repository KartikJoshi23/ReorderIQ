/** Baseline vs AI: replay of the held-back weeks of real demand through the classic and the
 *  AI-enhanced policies with identical delivery delays — the enhancement the exercise asks for. */
import { AlertTriangle, Loader2, Play } from "lucide-react";
import { useMemo } from "react";
import type { ReplayPolicyId, ReplayResult } from "../inventory/replay";
import { fmt } from "../lib/format";
import { quantile } from "../models/metrics";
import { useApp } from "../state/store";
import { axisTooltip, barStyle, dateAxis, hbarStyle, line, unitsAxis } from "../ui/charts/common";
import { Badge, Button, Card, ChartCard, DataTable, Empty, Explain, Kpi, Legend } from "../ui/components";
import { C } from "../ui/tokens";

const COLOR: Record<ReplayPolicyId, string> = { baseline: C.base, aiFormula: C.ai, optimised: C.s4 };
const ORDER: ReplayPolicyId[] = ["baseline", "aiFormula", "optimised"];

function ids(r: ReplayResult): ReplayPolicyId[] {
  return ORDER.filter((id) => r.policies[id]);
}

export function Impact() {
  const impact = useApp((s) => s.impact);
  const status = useApp((s) => s.impactStatus);
  const progress = useApp((s) => s.impactProgress);
  const stale = useApp((s) => s.impactStale);
  const run = useApp((s) => s.runImpact);
  const constraints = useApp((s) => s.constraints);
  const minFill = constraints?.minFillRate ?? 0.98;

  const levelOption = useMemo(() => {
    if (!impact) return null;
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(impact.dates),
      yAxis: unitsAxis(),
      series: ids(impact).map((id) => line((impact.policies[id] as { label: string }).label, Array.from((impact.policies[id] as { reorderLevel: Float64Array }).reorderLevel), COLOR[id], { dashed: id === "baseline" })),
    };
  }, [impact]);

  const demandOption = useMemo(() => {
    if (!impact) return null;
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: { ...dateAxis(impact.dates), boundaryGap: true },
      yAxis: unitsAxis(),
      series: [{ name: "Actual sales", type: "bar" as const, barMaxWidth: 10, data: Array.from(impact.actual), itemStyle: barStyle("#64748b") }],
    };
  }, [impact]);

  const stockOption = useMemo(() => {
    if (!impact) return null;
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(impact.dates),
      yAxis: unitsAxis(),
      series: ids(impact).map((id) => line((impact.policies[id] as { label: string }).label, Array.from((impact.policies[id] as { trace: { onHand: Float64Array } }).trace.onHand), COLOR[id], { width: 1.8 })),
    };
  }, [impact]);

  const scenarioOption = useMemo(() => {
    if (!impact) return null;
    const rows = [...ids(impact)].reverse();
    const share = (id: ReplayPolicyId) => {
      const f = (impact.policies[id] as { paths: { fillRate: Float64Array } }).paths.fillRate;
      return f.filter((v) => v >= minFill).length / f.length;
    };
    return {
      grid: { top: 8, left: 8, right: 56, bottom: 8, containLabel: true },
      tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => fmt.pct(v as number, 0) },
      xAxis: { type: "value" as const, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      yAxis: { type: "category" as const, data: rows.map((id) => (impact.policies[id] as { label: string }).label), axisLabel: { color: C.ink2, fontSize: 11.5 } },
      series: [{ type: "bar" as const, barMaxWidth: 22, data: rows.map((id) => ({ value: share(id), itemStyle: hbarStyle(COLOR[id]) })), label: { show: true, position: "right" as const, color: C.ink2, formatter: (p: { value: number }) => fmt.pct(p.value, 0) } }],
    };
  }, [impact, minFill]);

  const header = (
    <Card
      title="Replay of the held-back weeks of real demand"
      subtitle="Same real sales and the same 500 random delivery-delay sequences for every policy · unserved demand is lost"
      actions={
        <Button onClick={run} disabled={status === "running"}>
          {status === "running" ? <Loader2 size={14} className="animate-spin" aria-hidden /> : <Play size={14} aria-hidden />}
          {status === "running" ? "Replaying…" : impact ? "Re-run" : "Run replay"}
        </Button>
      }
    >
      <ul className="grid gap-1.5 text-[13px] leading-relaxed text-ink-2 md:grid-cols-3">
        <li><Badge kind="baseline">Baseline</Badge> Classic calculator: one reorder point from last year&apos;s average and SD, order size from EOQ.</li>
        <li><Badge kind="ai">AI formula</Badge> The same slide formula fed by the forecast, re-planned every week as that week&apos;s sales arrive.</li>
        <li><Badge kind="neutral">AI optimised</Badge> Forecast plus the buffer and order quantity chosen by Bayesian optimisation at the start.</li>
      </ul>
      {status === "running" && progress && <p className="mt-3 text-[12.5px] text-ink-2" aria-live="polite">{progress}…</p>}
      {status === "error" && <p className="mt-3 text-[12.5px] text-[#f19a9a]" role="alert">Replay failed: {progress}</p>}
      {stale && (
        <p className="mt-3 inline-flex items-center gap-1.5 text-[12.5px] text-ink-2">
          <AlertTriangle size={14} className="text-warning" aria-hidden /> Settings changed since this replay. Re-run to update.
        </p>
      )}
      <p className="mt-3 text-[12px] text-ink-3">Each policy first runs through the 4 weeks of real sales before the window without being scored, so the simulation&apos;s artificial starting stock does not count. The weekly re-forecasts use the Holt-Winters + Ridge ensemble (the LSTM is left out here to keep the 8 re-plans fast). Held-back data you released or added on Data &amp; Updates is ignored, so this is always the original real data.</p>
    </Card>
  );

  if (!impact || !levelOption || !demandOption || !stockOption || !scenarioOption) {
    return (
      <div className="grid gap-4">
        {header}
        <Empty>{status === "running" ? "Re-forecasting each week and replaying real demand…" : "Run the replay to compare the policies on the real held-back weeks (≈ 30 seconds)."}</Empty>
      </div>
    );
  }

  const P = impact.policies;
  const b = P.baseline;
  const a = P.aiFormula;
  const o = P.optimised;
  const days = impact.dates.length;
  const lostAvoided = b && a ? (b.metrics.lostUnitsPerDay - (o ?? a).metrics.lostUnitsPerDay) * days : Number.NaN;
  const rows = ids(impact).map((id) => {
    const r = P[id];
    if (!r) return [];
    const m = r.metrics;
    const D = impact.actual.reduce((x, y) => x + y, 0) / days;
    const f = r.paths.fillRate;
    const broken = constraints
      ? [
          m.fillRate < constraints.minFillRate && "fill",
          m.peakOnHandP95 > constraints.capacity && "storage",
          m.avgInventoryValue > constraints.budget && "budget",
          m.ordersPerWeek > constraints.maxOrdersPerWeek && "deliveries",
        ].filter(Boolean)
      : [];
    return [r.label, fmt.usd(m.costPerDay * days), fmt.usd(m.costPerDay), fmt.usd(m.holdingPerDay), fmt.usd(m.orderingPerDay), fmt.usd(m.stockoutPerDay), fmt.pct(m.fillRate, 2), `${fmt.pct(quantile(f, 0.05), 1)} – ${fmt.pct(quantile(f, 0.95), 1)}`, fmt.pct(m.cycleServiceLevel, 1), fmt.pct(m.stockoutDayRate, 1), fmt.one(m.avgOnHand / D), fmt.one(m.peakOnHandP95 / D), m.ordersPerWeek.toFixed(2), broken.length ? `✗ ${broken.join(", ")}` : "✓ all met"];
  });

  return (
    <div className="grid gap-4">
      {header}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Fill rate" value={(o ?? a)?.metrics.fillRate ?? Number.NaN} format={(v) => fmt.pct(v, 2)} badge={<Badge kind="ai">AI</Badge>} note={`Baseline ${fmt.pct(b?.metrics.fillRate, 2)} · target ${fmt.pct(minFill, 1)}`} tone={((o ?? a)?.metrics.fillRate ?? 0) >= minFill ? "good" : "warning"} />
        <Kpi label="Lost sales avoided" value={lostAvoided} format={fmt.int} unit="units" delay={0.05} note={`over the ${impact.weeks} replayed weeks vs the baseline`} />
        <Kpi label="Cost per day" value={(o ?? a)?.metrics.costPerDay ?? Number.NaN} format={fmt.usd} delay={0.1} note={`Baseline ${fmt.usd(b?.metrics.costPerDay)} · AI formula ${fmt.usd(a?.metrics.costPerDay)}`} tone={((o ?? a)?.metrics.costPerDay ?? 0) <= (b?.metrics.costPerDay ?? 0) ? "good" : "warning"} />
        <Kpi label="AI fill ≥ baseline" value={impact.paired.aiFillAtLeastBaseline} format={(v) => fmt.pct(v, 0)} delay={0.15} note="of the 500 delay scenarios (paired, same delays)" />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartCard title="Real demand in the held-back weeks" subtitle="Daily units actually sold. The models had not seen these at the replay start" option={demandOption} height={240} ariaLabel="Bar chart of actual daily sales during the replay window" table={{ columns: ["Date", "Units sold"], rows: impact.dates.map((d, i) => [d, fmt.int(impact.actual[i])]) }} />
        <ChartCard
          title="Reorder level each day"
          subtitle="The baseline stays flat; the AI re-plans every week from the newest forecast"
          option={levelOption}
          height={240}
          ariaLabel="Reorder level by policy over the replay window"
          legend={<Legend items={ids(impact).map((id) => ({ label: (P[id] as { label: string }).label, color: COLOR[id], kind: (id === "baseline" ? "dash" : "line") as "dash" | "line" }))} />}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <ChartCard
          className="xl:col-span-3"
          title="Stock on hand under the same delivery delays"
          subtitle="Delay scenario 1 of 500; dips to zero are lost sales"
          option={stockOption}
          height={260}
          ariaLabel="Stock on hand over the replay window for each policy"
          legend={<Legend items={ids(impact).map((id) => ({ label: (P[id] as { label: string }).label, color: COLOR[id] }))} />}
        />
        <ChartCard
          className="xl:col-span-2"
          title={`Delay scenarios meeting ${fmt.pct(minFill, 1)} fill rate`}
          subtitle="Share of the 500 delivery-delay sequences in which the policy hit the target"
          option={scenarioOption}
          height={260}
          ariaLabel="Share of delay scenarios meeting the fill-rate target by policy"
        />
      </div>

      <Card title="Scorecard over the replayed weeks" subtitle="Means over 500 delay scenarios; fill-rate range = 5th–95th percentile; constraints from the Optimisation page">
        <DataTable spec={{ columns: ["Policy", "Total cost", "Cost/day", "Holding", "Ordering", "Lost sales", "Fill rate", "Fill range", "Cycle service", "Stockout days", "Avg stock (days)", "Peak stock (days)", "Orders/wk", "Constraints"], rows }} />
        <Explain className="mt-3">
          {b && a ? (
            <>
              On real demand the models had not seen, the classic policy served <b className="text-ink">{fmt.pct(b.metrics.fillRate, 2)}</b> of demand and the AI formula <b className="text-ink">{fmt.pct(a.metrics.fillRate, 2)}</b>
              {o ? <> (optimised: <b className="text-ink">{fmt.pct(o.metrics.fillRate, 2)}</b> at {fmt.usd(o.metrics.costPerDay)}/day vs {fmt.usd(b.metrics.costPerDay)} for the baseline)</> : null}.
              {" "}Under identical delays, the AI formula matched or beat the baseline&apos;s fill rate in {fmt.pct(impact.paired.aiFillAtLeastBaseline, 0)} of scenarios
              {impact.paired.optimisedCheaperThanBaseline !== null ? <>, and the optimised policy was cheaper in {fmt.pct(impact.paired.optimisedCheaperThanBaseline, 0)}</> : null}. A policy that looks cheaper but breaks a constraint (for example holding more stock than the store can take) is not a real saving. These are measured results on one real 8-week window, not guarantees.
            </>
          ) : null}
        </Explain>
      </Card>
    </div>
  );
}
