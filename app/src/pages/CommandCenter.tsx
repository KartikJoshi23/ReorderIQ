/** Command Center: the decisions an operations manager needs today, with plain-language reading. */
import { useMemo } from "react";
import { dateAt } from "../data/series";
import { fmt } from "../lib/format";
import { usePolicies } from "../state/selectors";
import { useApp } from "../state/store";
import { axisTooltip, band, dateAxis, hbarStyle, line, unitsAxis } from "../ui/charts/common";
import { Badge, Card, ChartCard, Empty, Explain, Kpi, Legend } from "../ui/components";
import { C } from "../ui/tokens";

export function CommandCenter() {
  const series = useApp((s) => s.series);
  const asOf = useApp((s) => s.asOf);
  const forecast = useApp((s) => s.forecast);
  const mc = useApp((s) => s.mc);
  const mcBase = useApp((s) => s.mcBaselineStockout);
  const bo = useApp((s) => s.bo);
  const sim = useApp((s) => s.sim);
  const serviceLevel = useApp((s) => s.serviceLevel);
  const pol = usePolicies();

  const ladder = useMemo(() => {
    if (!pol) return null;
    const optimisedRop = bo?.best?.details?.policy.kind === "dynamic" ? pol.ai.leadTimeDemand + bo.best.details.policy.buffer : null;
    return [
      { label: "Baseline (365-day average)", value: pol.baseline.reorderPoint, color: C.base },
      { label: "AI formula (forecast-driven)", value: pol.ai.reorderPoint, color: C.ai },
      { label: "Monte Carlo (risk-calibrated)", value: mc?.reorderPoint ?? null, color: C.mc },
      { label: "Bayesian-optimised (cost & constraints)", value: optimisedRop, color: C.s4 },
    ];
  }, [pol, mc, bo]);

  const demandOption = useMemo(() => {
    if (!series || !forecast) return null;
    const back = 56;
    const histIdx = Array.from({ length: back }, (_, k) => asOf - back + 1 + k);
    const dates = [...histIdx.map((i) => dateAt(series, i)), ...forecast.dates];
    const actual = [...histIdx.map((i) => (series.closed[i] ? null : (series.sales[i] as number))), ...forecast.dates.map(() => null)];
    const pad = histIdx.map(() => null);
    const f = [...pad.slice(0, -1), actual[back - 1] ?? null, ...Array.from(forecast.ensemble.forecast)];
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(dates),
      yAxis: unitsAxis(),
      series: [
        ...band("80% band", [...pad, ...Array.from(forecast.bands.lo80)], [...pad, ...Array.from(forecast.bands.hi80)], C.ai, 0.16),
        line("Actual sales", actual, C.actual, { width: 1.5 }),
        line("AI forecast", f, C.ai),
      ],
    };
  }, [series, forecast, asOf]);

  const ladderOption = useMemo(() => {
    if (!ladder) return null;
    const rows = [...ladder].reverse();
    return {
      grid: { top: 8, left: 8, right: 64, bottom: 8, containLabel: true },
      tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => fmt.int(v as number) },
      xAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => fmt.axis(v) } },
      yAxis: { type: "category" as const, data: rows.map((r) => r.label), axisLabel: { color: C.ink2, fontSize: 11.5 } },
      series: [{
        type: "bar" as const,
        barMaxWidth: 22,
        data: rows.map((r) => ({ value: r.value, itemStyle: hbarStyle(r.color) })),
        label: { show: true, position: "right" as const, color: C.ink2, fontSize: 11.5, formatter: (p: { value: number | null }) => (p.value === null ? "computing…" : fmt.int(p.value)) },
      }],
    };
  }, [ladder]);

  if (!series || !forecast || !pol || !demandOption || !ladderOption) return <Empty>Preparing forecasts and policies…</Empty>;

  const a = pol.ai;
  const b = pol.baseline;
  const wapeGain = 1 - forecast.ensemble.metrics.wape / forecast.baselines.ma28.metrics.wape;
  const risk = mc?.stockoutProbability;
  const target = 1 - serviceLevel;
  const boCost = bo?.best?.objective ?? null;
  const baseCost = sim.baseline?.costPerDay ?? null;

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Reorder point today" value={mc?.reorderPoint ?? a.reorderPoint} format={fmt.int} unit="units" badge={<Badge kind={mc ? "mc" : "ai"}>{mc ? "Monte Carlo" : "AI formula"}</Badge>} note={`Baseline would reorder at ${fmt.int(b.reorderPoint)}`} />
        <Kpi label="Safety stock (AI)" value={a.safetyStock} format={fmt.int} unit="units" delay={0.05} note={`${fmt.days(a.daysOfCover)} of cover · ${fmt.pct(a.leadTimeShare, 0)} covers late deliveries`} />
        <Kpi
          label="Stockout risk per cycle"
          value={risk ?? Number.NaN}
          format={(v) => fmt.pct(v)}
          delay={0.1}
          tone={risk === undefined ? undefined : risk <= target * 1.1 ? "good" : risk <= target * 2 ? "warning" : "critical"}
          note={risk === undefined ? "Simulating…" : `AI formula ROP · baseline ${fmt.pct(mcBase ?? Number.NaN)} · target ${fmt.pct(target)}`}
        />
        <Kpi label="Forecast error (WAPE)" value={forecast.ensemble.metrics.wape} format={(v) => fmt.pct(v)} delay={0.15} tone={wapeGain > 0 ? "good" : "warning"} note={`${fmt.pct(wapeGain, 0)} fewer units mis-forecast than the moving average`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <ChartCard
          className="xl:col-span-3"
          title="Demand: last 8 weeks and the next 4"
          subtitle="Daily units · AI ensemble forecast with 80% prediction band"
          option={demandOption}
          height={300}
          ariaLabel="Line chart of recent daily sales and the AI forecast for the next 28 days with an 80 percent band"
          legend={<Legend items={[{ label: "Actual sales", color: C.actual }, { label: "AI forecast", color: C.ai }, { label: "80% band", color: C.ai, kind: "area" }]} />}
          table={{ columns: ["Date", "Forecast", "80% low", "80% high"], rows: forecast.dates.map((d, k) => [d, fmt.int(forecast.ensemble.forecast[k]), fmt.int(forecast.bands.lo80[k]), fmt.int(forecast.bands.hi80[k])]) }}
          explain={<>Next week the AI expects <b className="text-ink">{fmt.int(a.leadTimeDemand)}</b> units over the {fmt.one(a.leadTime)}-day lead time (≈ {fmt.int(a.dBar)} a day), compared with the baseline&apos;s flat {fmt.int(b.dBar)} a day from last year&apos;s average.</>}
        />
        <ChartCard
          className="xl:col-span-2"
          title="Reorder-point ladder"
          subtitle="Each AI layer adds information; units of stock on hand + on order"
          option={ladderOption}
          height={300}
          ariaLabel="Bar chart comparing the baseline, AI formula, Monte Carlo and optimised reorder points"
          table={{ columns: ["Method", "Reorder point (units)"], rows: (ladder ?? []).map((r) => [r.label, r.value === null ? "computing…" : fmt.int(r.value)]) }}
          explain={<>Order when stock plus open orders falls to the reorder point. The Monte Carlo step uses real forecast errors and skewed delivery times, so it shows the buffer actually needed for {fmt.pct(serviceLevel, 1)} service.</>}
        />
      </div>

      <Card title="What this means for the operations manager">
        <ul className="grid gap-2 text-[13.5px] leading-relaxed text-ink-2 md:grid-cols-2">
          <li>
            • <b className="text-ink">{fmt.pct(a.leadTimeShare, 0)} of the safety stock protects against late deliveries</b>, not demand swings. Cutting supplier lead-time variability (σ_L) is the biggest lever on buffer size.
          </li>
          <li>
            • The forecast explains much of the day-to-day swing. Forecast error σ_e is <b className="text-ink">{fmt.int(a.sigmaD)}</b> a day, against raw variability σ_D of {fmt.int(b.sigmaD)}.
          </li>
          <li>
            • With the baseline reorder point the simulated stockout risk is <b className="text-ink">{fmt.pct(mcBase ?? Number.NaN)}</b> per replenishment cycle, against a target of {fmt.pct(target)}.
          </li>
          <li>
            • {boCost !== null && baseCost !== null ? (
              <>The optimised policy costs <b className="text-ink">{fmt.usd(boCost)}/day</b>, against {fmt.usd(baseCost)}/day for the classic policy, while meeting every constraint. See Optimisation.</>
            ) : (
              <>Bayesian optimisation is searching for the cheapest policy that meets the service, storage, budget and delivery limits…</>
            )}
          </li>
        </ul>
        <Explain className="mt-3">Everything updates automatically when new sales data arrives (Data &amp; Updates) or when you move the service-level and lead-time controls above.</Explain>
      </Card>
    </div>
  );
}
