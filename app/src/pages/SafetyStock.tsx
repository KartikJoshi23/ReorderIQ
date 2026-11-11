/** Safety Stock & Risk: the slide formula with live numbers (baseline vs AI), Monte Carlo of
 *  demand during the lead time, risk scenarios, lead-time distribution and the dynamic ROP path. */
import { useMemo } from "react";
import type { FormulaResult } from "../inventory/safetyStock";
import { fmt } from "../lib/format";
import { usePolicies } from "../state/selectors";
import { useApp } from "../state/store";
import { axisTooltip, dateAxis, line, unitsAxis } from "../ui/charts/common";
import { Badge, Card, ChartCard, Empty, Explain, Legend, Stat, Status } from "../ui/components";
import { C, wash } from "../ui/tokens";

function FormulaPanel({ title, badge, r, sigmaName, sigmaNote }: { title: string; badge: "baseline" | "ai"; r: FormulaResult; sigmaName: string; sigmaNote: string }) {
  const demandShare = 1 - r.leadTimeShare;
  return (
    <Card title={title} actions={<Badge kind={badge}>{badge === "ai" ? "AI-enhanced" : "Classic calculator"}</Badge>}>
      <div className="num space-y-1 overflow-x-auto text-[12.5px] leading-relaxed text-ink-2">
        <div className="text-ink">SS = Z × √( {sigmaName}² × L + D̄² × σ_L² )</div>
        <div>
          &nbsp;&nbsp;&nbsp;= {r.z.toFixed(3)} × √( {fmt.int(r.sigmaD)}² × {fmt.one(r.leadTime)} + {fmt.int(r.dBar)}² × {r.sigmaL.toFixed(2)}² )
        </div>
        <div>
          &nbsp;&nbsp;&nbsp;= {r.z.toFixed(3)} × √( {fmt.compact(r.termDemand)} + {fmt.compact(r.termLeadTime)} ) = <b className="text-ink">{fmt.int(r.safetyStock)} units</b>
        </div>
        <div className="pt-1 text-ink">ROP = D̄ × L + SS = {fmt.int(r.leadTimeDemand)} + {fmt.int(r.safetyStock)} = <b>{fmt.int(r.reorderPoint)} units</b></div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="D̄ (units/day)" value={fmt.int(r.dBar)} sub={badge === "ai" ? "forecast over lead time" : "last 365 days"} />
        <Stat label={sigmaName} value={fmt.int(r.sigmaD)} sub={sigmaNote} />
        <Stat label="Z" value={r.z.toFixed(3)} sub={`service ${fmt.pct(r.serviceLevel, 1)}`} />
        <Stat label="Cover" value={fmt.days(r.daysOfCover)} sub="safety stock" />
      </div>
      <div className="mt-3">
        <div className="mb-1 flex justify-between text-[11.5px] text-ink-3">
          <span>Demand uncertainty {fmt.pct(demandShare, 0)}</span>
          <span>Lead-time uncertainty {fmt.pct(r.leadTimeShare, 0)}</span>
        </div>
        <div className="flex h-2.5 gap-[2px] overflow-hidden rounded-full" role="img" aria-label={`Variance split: demand ${fmt.pct(demandShare, 0)}, lead time ${fmt.pct(r.leadTimeShare, 0)}`}>
          <div style={{ width: `${demandShare * 100}%`, background: badge === "ai" ? C.ai : C.base }} />
          <div style={{ width: `${r.leadTimeShare * 100}%`, background: "#94a3b8" }} />
        </div>
      </div>
    </Card>
  );
}

function lnGamma(z: number): number {
  const g = 7;
  const p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  const x = z - 1;
  let a = p[0] as number;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += (p[i] as number) / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function SafetyStock() {
  const pol = usePolicies();
  const mc = useApp((s) => s.mc);
  const mcBase = useApp((s) => s.mcBaselineStockout);
  const scenarios = useApp((s) => s.scenarios);
  const leadTime = useApp((s) => s.leadTime);
  const serviceLevel = useApp((s) => s.serviceLevel);
  const forecast = useApp((s) => s.forecast);

  const histOption = useMemo(() => {
    if (!mc || !pol) return null;
    const e = mc.histogram.edges;
    const mids = mc.histogram.counts.map((_, i) => ((e[i] as number) + (e[i + 1] as number)) / 2);
    const nearest = (x: number) => mids.reduce((best, m, i) => (Math.abs(m - x) < Math.abs((mids[best] as number) - x) ? i : best), 0);
    const total = mc.trials;
    const aiIdx = nearest(pol.ai.reorderPoint);
    return {
      grid: { top: 24, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: { trigger: "axis" as const, formatter: (ps: { dataIndex: number; value: number }[]) => { const p = ps[0]; return p ? `${fmt.int(mids[p.dataIndex])} units<br/><b>${fmt.pct(p.value / total, 2)}</b> of cycles` : ""; } },
      xAxis: { type: "category" as const, data: mids.map((m) => Math.round(m)), axisLabel: { formatter: (v: string) => fmt.axis(Number(v)), hideOverlap: true } },
      yAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => fmt.pct(v / total, 1) } },
      series: [{
        type: "bar" as const,
        barCategoryGap: "8%",
        data: mc.histogram.counts.map((c, i) => ({ value: c, itemStyle: { color: i > aiIdx ? wash(C.critical, 0.75) : "#64748b", borderRadius: [3, 3, 0, 0] } })),
        markLine: {
          symbol: "none",
          silent: true,
          data: [
            { xAxis: nearest(pol.baseline.reorderPoint), label: { formatter: "baseline ROP", color: C.ink2, fontSize: 11 }, lineStyle: { color: C.base, width: 2, type: "solid" as const } },
            { xAxis: aiIdx, label: { formatter: "AI formula ROP", color: C.ink2, fontSize: 11 }, lineStyle: { color: C.ai, width: 2, type: "solid" as const } },
            { xAxis: nearest(mc.reorderPoint), label: { formatter: `MC ${fmt.pct(serviceLevel, 0)} ROP`, color: C.ink2, fontSize: 11, position: "insideEndBottom" as const }, lineStyle: { color: C.mc, width: 2, type: "dashed" as const } },
          ],
        },
      }],
    };
  }, [mc, pol, serviceLevel]);

  const ltOption = useMemo(() => {
    const { mean, sd, kind, samples } = leadTime;
    const xs = Array.from({ length: 121 }, (_, i) => (i / 120) * Math.max(mean + 4.5 * sd, mean * 1.6));
    let ys: number[];
    if (kind === "empirical" && samples?.length) {
      ys = xs.map((x) => samples.filter((v) => Math.abs(v - x) <= 0.5).length / samples.length);
    } else if (kind === "gamma" && sd > 0) {
      const k = (mean / sd) ** 2;
      const th = (sd * sd) / mean;
      ys = xs.map((x) => (x <= 0 ? 0 : Math.exp((k - 1) * Math.log(x) - x / th - lnGamma(k) - k * Math.log(th))));
    } else {
      ys = xs.map((x) => (sd > 0 ? Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)) : 0));
    }
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 24, containLabel: true },
      tooltip: { trigger: "axis" as const, valueFormatter: (v: unknown) => (v as number).toFixed(3) },
      xAxis: { type: "category" as const, data: xs.map((x) => x.toFixed(1)), name: "days", nameLocation: "middle" as const, nameGap: 24, axisLabel: { interval: 19 } },
      yAxis: { type: "value" as const, axisLabel: { show: false } },
      series: [{ ...line("density", ys, C.s4, { area: true, smooth: true }) }],
    };
  }, [leadTime]);

  const pathOption = useMemo(() => {
    if (!pol) return null;
    const p = pol.path;
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(p.dates),
      yAxis: unitsAxis(),
      series: [
        line("AI reorder point", p.reorderPoint, C.ai),
        line("AI safety stock", p.safetyStock, C.ai, { dashed: true, width: 1.5 }),
        line("Baseline reorder point (static)", p.dates.map(() => pol.baseline.reorderPoint), C.base, { dashed: true }),
      ],
    };
  }, [pol]);

  if (!pol || !forecast) return <Empty>Computing safety stock…</Empty>;
  const target = 1 - serviceLevel;
  const tone = (p?: number | null) => (p === undefined || p === null ? "neutral" : p <= target * 1.1 ? "good" : p <= target * 2 ? "warning" : "critical") as "good" | "warning" | "critical" | "neutral";

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <FormulaPanel title="Baseline safety stock" badge="baseline" r={pol.baseline} sigmaName="σ_D" sigmaNote="raw daily SD" />
        <FormulaPanel title="AI-enhanced safety stock" badge="ai" r={pol.ai} sigmaName="σ_e" sigmaNote={`forecast RMSE (${pol.ai.errorCount} errors)`} />
      </div>
      <Explain>
        Both panels use the slide&apos;s formula exactly. The baseline counts predictable weekend and holiday swings as risk (σ_D = {fmt.int(pol.baseline.sigmaD)}). The AI version only counts what the forecast cannot predict (σ_e = {fmt.int(pol.ai.sigmaD)}) and uses the demand <i>expected</i> over the coming lead time. At σ_L = {leadTime.sd.toFixed(2)} days, lead-time uncertainty dominates both buffers.
      </Explain>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        {histOption && mc ? (
          <ChartCard
            className="xl:col-span-3"
            title="Monte Carlo: demand during the lead time"
            subtitle={`${fmt.int(mc.trials)} simulated replenishment cycles · forecast × real out-of-sample errors · ${leadTime.kind} lead times`}
            option={histOption}
            height={300}
            ariaLabel="Histogram of simulated demand during the lead time with reorder points and the stockout tail"
            legend={<Legend items={[{ label: "Simulated cycles", color: "#64748b", kind: "bar" }, { label: "Stockout tail at AI ROP", color: C.critical, kind: "bar" }, { label: "Baseline ROP", color: C.base }, { label: "AI formula ROP", color: C.ai }, { label: "Monte Carlo ROP", color: C.mc, kind: "dash" }]} />}
            table={{ columns: ["Statistic", "Value"], rows: [["Mean demand during lead time", fmt.int(mc.mean)], ["SD", fmt.int(mc.sd)], ["Skewness", mc.skewness.toFixed(2)], ["95th percentile", fmt.int(mc.quantiles.p95)], [`ROP for ${fmt.pct(serviceLevel, 1)} (MC)`, fmt.int(mc.reorderPoint)], ["Safety stock needed (MC)", fmt.int(mc.safetyStock)], ["P(stockout) at AI formula ROP", fmt.pct(mc.stockoutProbability, 2)], ["P(stockout) at baseline ROP", fmt.pct(mcBase, 2)], ["Expected units short per cycle", fmt.int(mc.expectedShortage)], ["Fill rate with EOQ", fmt.pct(mc.fillRate, 2)]] }}
            explain={<>With the AI formula&apos;s reorder point, stock runs out in <b className="text-ink">{fmt.pct(mc.stockoutProbability)}</b> of cycles (target {fmt.pct(target)}); with the baseline&apos;s, <b className="text-ink">{fmt.pct(mcBase)}</b>. For exactly {fmt.pct(serviceLevel, 1)} service the simulation needs a reorder point of <b className="text-ink">{fmt.int(mc.reorderPoint)}</b>, which allows for skewed delays and real forecast errors that the formula&apos;s normal approximation misses.</>}
          />
        ) : (
          <Card className="xl:col-span-3" title="Monte Carlo"><Empty>Simulating…</Empty></Card>
        )}
        <Card className="xl:col-span-2" title="Risk scenarios" subtitle="Stress test of the current AI reorder point (20,000 cycles each)">
          {scenarios ? (
            <ul className="grid gap-2">
              {scenarios.map(({ scenario, result }) => (
                <li key={scenario.id} className="rounded-xl border border-line bg-white/[0.02] px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[13px] font-medium text-ink">{scenario.label}</span>
                    <Status tone={tone(result.stockoutProbability)}>{fmt.pct(result.stockoutProbability)} stockout risk</Status>
                  </div>
                  <div className="num mt-0.5 text-[12px] text-ink-3">needs ROP {fmt.int(result.reorderPoint)} · short {fmt.int(result.expectedShortage)} units/cycle</div>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>Running scenarios…</Empty>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <ChartCard
          className="xl:col-span-2"
          title="Lead-time distribution"
          subtitle={`${leadTime.kind === "gamma" ? "Gamma" : leadTime.kind === "normal" ? "Truncated normal" : "Empirical"} · L = ${leadTime.mean.toFixed(1)} d, σ_L = ${leadTime.sd.toFixed(2)} d`}
          option={ltOption}
          height={220}
          ariaLabel="Probability density of the replenishment lead time"
          explain={<>Gamma delays are right-skewed: a few very late deliveries drive most of the stockout risk. {mc ? <>95% of simulated deliveries arrive within {mc.leadTime.p95.toFixed(1)} days.</> : null}</>}
        />
        {pathOption && (
          <ChartCard
            className="xl:col-span-3"
            title="Dynamic reorder point over the coming weeks"
            subtitle="Recomputed for each day's upcoming lead time as the forecast moves"
            option={pathOption}
            height={220}
            ariaLabel="AI reorder point and safety stock over the coming days compared with the static baseline"
            legend={<Legend items={[{ label: "AI reorder point", color: C.ai }, { label: "AI safety stock", color: C.ai, kind: "dash" }, { label: "Baseline ROP (static)", color: C.base, kind: "dash" }]} />}
            table={{ columns: ["Date", "AI D̄", "AI SS", "AI ROP"], rows: pol.path.dates.map((d, k) => [d, fmt.int(pol.path.dBar[k]), fmt.int(pol.path.safetyStock[k]), fmt.int(pol.path.reorderPoint[k])]) }}
            explain={<>The AI reorder point rises before busy weekends and holidays and falls after them. The baseline stays flat at {fmt.int(pol.baseline.reorderPoint)}.</>}
          />
        )}
      </div>
    </div>
  );
}
