/** Demand Forecast: forecast vs actual with uncertainty bands, model comparison, backtest and drivers. */
import { useMemo, useState } from "react";
import { dateAt } from "../data/series";
import { fmt } from "../lib/format";
import { MODEL_INFO, type AnyModelId } from "../models/backtest";
import { useApp } from "../state/store";
import { axisTooltip, band, barStyle, dateAxis, hbarStyle, line, unitsAxis } from "../ui/charts/common";
import { type ChartOption, EChart } from "../ui/charts/EChart";
import { Badge, Button, Card, ChartCard, DataTable, Empty, Legend, Slider, Stat } from "../ui/components";
import { C } from "../ui/tokens";

const MODEL_COLORS: Partial<Record<AnyModelId, string>> = { ensemble: C.ai, ridge: C.mc, lstm: C.s4, ets: "#94a3b8" };

export function Forecast() {
  const series = useApp((s) => s.series);
  const asOf = useApp((s) => s.asOf);
  const f = useApp((s) => s.forecast);
  const promoPlan = useApp((s) => s.promoPlan);
  const setPromoPlan = useApp((s) => s.setPromoPlan);
  const [show, setShow] = useState<Record<string, boolean>>({ ridge: false, lstm: false, ets: false });
  const [reveal, setReveal] = useState(false);

  const mainOption = useMemo(() => {
    if (!series || !f) return null;
    const back = 84;
    const hist = Array.from({ length: back }, (_, k) => asOf - back + 1 + k);
    const dates = [...hist.map((i) => dateAt(series, i)), ...f.dates];
    const pad = hist.map(() => null);
    const actual = [...hist.map((i) => (series.closed[i] ? null : (series.sales[i] as number))), ...f.dates.map(() => null)];
    const future = [...pad, ...f.dates.map((_, k) => {
      const i = f.start + k;
      const v = i < series.length ? (series.sales[i] as number) : Number.NaN;
      return reveal && Number.isFinite(v) && !series.closed[i] ? v : null;
    })];
    const join = (arr: Float64Array) => [...pad.slice(0, -1), actual[back - 1] ?? null, ...Array.from(arr)];
    const extra = (["ridge", "lstm", "ets"] as const).filter((m) => show[m] && f.models[m]).map((m) => line(MODEL_INFO[m].label, join((f.models[m] as { forecast: Float64Array }).forecast), MODEL_COLORS[m] as string, { width: 1.5, dashed: m === "ets" }));
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(dates),
      yAxis: unitsAxis(),
      series: [
        ...band("95% band", [...pad, ...Array.from(f.bands.lo95)], [...pad, ...Array.from(f.bands.hi95)], C.ai, 0.08),
        ...band("80% band", [...pad, ...Array.from(f.bands.lo80)], [...pad, ...Array.from(f.bands.hi80)], C.ai, 0.14),
        line("Actual sales", actual, C.actual, { width: 1.5 }),
        ...(reveal ? [{ ...line("Held-back actuals (not used by models)", future, C.actual, { width: 1.5, dashed: true }) }] : []),
        ...extra,
        line("AI ensemble forecast", join(f.ensemble.forecast), C.ai, { z: 5 }),
      ],
    };
  }, [series, f, asOf, show, reveal]);

  const compare = useMemo(() => {
    if (!f) return null;
    const rows: { id: AnyModelId; m: { wape: number; mae: number; rmse: number; mase: number; bias: number; n: number } }[] = [
      { id: "ensemble", m: f.ensemble.metrics },
      ...(["ridge", "lstm", "ets"] as const).filter((id) => f.models[id]).map((id) => ({ id, m: (f.models[id] as { metrics: typeof f.ensemble.metrics }).metrics })),
      { id: "snaive", m: f.baselines.snaive.metrics },
      { id: "ma28", m: f.baselines.ma28.metrics },
    ];
    return rows;
  }, [f]);

  const compareOption = useMemo(() => {
    if (!compare) return null;
    const rows = [...compare].reverse();
    return {
      grid: { top: 8, left: 8, right: 48, bottom: 8, containLabel: true },
      tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => fmt.pct(v as number) },
      xAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      yAxis: { type: "category" as const, data: rows.map((r) => MODEL_INFO[r.id].label), axisLabel: { color: C.ink2, fontSize: 11.5 } },
      series: [{
        type: "bar" as const,
        barMaxWidth: 20,
        data: rows.map((r) => ({ value: r.m.wape, itemStyle: hbarStyle(r.id === "ensemble" ? C.ai : r.id === "ma28" ? C.base : "#64748b") })),
        label: { show: true, position: "right" as const, color: C.ink2, fontSize: 11.5, formatter: (p: { value: number }) => fmt.pct(p.value) },
      }],
    };
  }, [compare]);

  const foldOption = useMemo(() => {
    if (!series || !f) return null;
    const folds = f.ensemble.folds.map((x) => fmt.shortDate(dateAt(series, x.origin + 1)));
    const wapeOf = (fold: { actual: Float64Array; forecast: Float64Array }) => {
      let e = 0;
      let a = 0;
      fold.actual.forEach((v, k) => {
        if (Number.isFinite(v)) {
          e += Math.abs(v - (fold.forecast[k] as number));
          a += v;
        }
      });
      return a ? e / a : null;
    };
    const ids = ["ensemble", "ridge", "lstm", "ets"] as const;
    return {
      grid: { top: 28, left: 8, right: 16, bottom: 8, containLabel: true },
      legend: { show: false },
      tooltip: { trigger: "axis" as const, valueFormatter: (v: unknown) => fmt.pct(v as number) },
      xAxis: { type: "category" as const, data: folds, name: "fold starting", nameLocation: "middle" as const, nameGap: 26 },
      yAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      series: ids.filter((id) => (id === "ensemble" ? true : f.models[id])).map((id) => ({
        name: MODEL_INFO[id].label,
        type: "bar" as const,
        barMaxWidth: 12,
        barGap: "15%",
        itemStyle: barStyle(MODEL_COLORS[id] as string),
        data: (id === "ensemble" ? f.ensemble.folds : (f.models[id] as { folds: typeof f.ensemble.folds }).folds).map(wapeOf),
      })),
    };
  }, [series, f]);

  const driverOption = useMemo(() => {
    if (!f) return null;
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
      tooltip: { trigger: "item" as const, valueFormatter: (v: unknown) => `× ${(v as number).toFixed(2)}` },
      xAxis: { type: "category" as const, data: days },
      yAxis: { type: "value" as const, min: 0, axisLabel: { formatter: (v: number) => `×${v.toFixed(1)}` } },
      series: [{ type: "bar" as const, barMaxWidth: 24, data: f.drivers.weekday, itemStyle: barStyle(C.ai), label: { show: true, position: "top" as const, color: C.ink2, fontSize: 11, formatter: (p: { value: number }) => `×${p.value.toFixed(2)}` } }],
    };
  }, [f]);

  if (!series || !f || !mainOption || !compare || !compareOption || !foldOption || !driverOption) return <Empty>Training forecasting models…</Empty>;
  const ens = f.ensemble.metrics;
  const d = f.drivers;
  const toggles = (["ridge", "lstm", "ets"] as const).filter((m) => f.models[m]);

  return (
    <div className="grid gap-4">
      <ChartCard
        title="Forecast vs actual"
        subtitle={`Last 12 weeks of actual sales and the next ${f.horizon} days · bands from out-of-sample errors (split-conformal)`}
        option={mainOption}
        height={360}
        ariaLabel="Forecast versus actual sales with 80 and 95 percent prediction bands"
        actions={
          <>
            {toggles.map((m) => (
              <Button key={m} size="sm" variant={show[m] ? "outline" : "ghost"} aria-pressed={show[m]} onClick={() => setShow((s) => ({ ...s, [m]: !s[m] }))}>
                {MODEL_INFO[m].label}
              </Button>
            ))}
            <Button size="sm" variant={reveal ? "outline" : "ghost"} aria-pressed={reveal} onClick={() => setReveal((v) => !v)}>
              Reveal held-back actuals
            </Button>
          </>
        }
        legend={<Legend items={[
          { label: "Actual sales", color: C.actual },
          { label: "AI ensemble forecast", color: C.ai },
          { label: "80% / 95% bands", color: C.ai, kind: "area" },
          ...toggles.filter((m) => show[m]).map((m) => ({ label: MODEL_INFO[m].label, color: MODEL_COLORS[m] as string, kind: (m === "ets" ? "dash" : "line") as "dash" | "line" })),
          ...(reveal ? [{ label: "Held-back actuals (not used)", color: C.actual, kind: "dash" as const }] : []),
        ]} />}
        table={{ columns: ["Date", "Ensemble", "ETS", "Ridge", "LSTM", "80% low", "80% high"], rows: f.dates.map((dt, k) => [dt, fmt.int(f.ensemble.forecast[k]), fmt.int(f.models.ets?.forecast[k]), fmt.int(f.models.ridge?.forecast[k]), fmt.int(f.models.lstm?.forecast[k]), fmt.int(f.bands.lo80[k]), fmt.int(f.bands.hi80[k])]) }}
        explain={<>The ensemble combines {Object.keys(f.ensemble.weights).length} models, weighted by recent accuracy ({Object.entries(f.ensemble.weights).map(([k, w]) => `${MODEL_INFO[k as AnyModelId].label} ${fmt.pct(w, 0)}`).join(", ")}). In the rolling backtest, 80% of actual days should fall inside the darker band.</>}
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <ChartCard
          className="xl:col-span-3"
          title="Model comparison"
          subtitle="Rolling backtest: 6 folds × 28 days, scored on store-open days · WAPE (lower is better)"
          option={compareOption}
          height={260}
          ariaLabel="Bar chart of WAPE by forecasting model"
          explain={<>The AI ensemble mis-forecasts <b className="text-ink">{fmt.pct(1 - ens.wape / f.baselines.ma28.metrics.wape, 0)} fewer units</b> than the 28-day moving average the classic calculator relies on. A MASE below 1 beats the seasonal-naive benchmark.</>}
        />
        <Card className="xl:col-span-2" title="Accuracy metrics" subtitle="Pooled over all backtest folds">
          <DataTable spec={{ columns: ["Model", "WAPE", "MAE", "RMSE", "MASE", "Bias"], rows: compare.map((r) => [MODEL_INFO[r.id].label, fmt.pct(r.m.wape), fmt.int(r.m.mae), fmt.int(r.m.rmse), r.m.mase.toFixed(2), fmt.signedPct(r.m.bias)]) }} />
          {f.lstm && (
            <p className="mt-3 text-[12px] text-ink-3">
              LSTM: {f.lstm.action} on the {f.lstm.backend.toUpperCase()} backend, {f.lstm.epochsRun} epochs in total; backtest from model v{f.lstm.backtestVersion}.
            </p>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartCard
          title="Accuracy by backtest fold"
          subtitle="WAPE per 28-day fold; the ensemble uses only earlier folds to set its weights"
          option={foldOption}
          height={250}
          ariaLabel="Grouped bar chart of WAPE by fold and model"
          legend={<Legend items={(["ensemble", "ridge", "lstm", "ets"] as const).filter((id) => id === "ensemble" || f.models[id]).map((id) => ({ label: MODEL_INFO[id].label, color: MODEL_COLORS[id] as string, kind: "bar" as const }))} />}
        />
        <Card title="What drives demand" subtitle="Effects measured by the regression model (holding recent sales level fixed)">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-ink-3">Weekday multiplier vs Monday</div>
              <ChartLite option={driverOption} />
            </div>
            <div className="grid grid-cols-2 content-start gap-2">
              <Stat label="Holiday" value={fmt.signedPct(d.holiday, 0)} />
              <Stat label="Promotions ×2" value={fmt.signedPct(d.promoPerDoubling, 1)} sub="per doubling of items" />
              <Stat label="Payday" value={fmt.signedPct(d.payday, 1)} />
              <Stat label="Christmas Eve" value={fmt.signedPct(d.christmasEve, 0)} sub="beyond the December trend" />
              <Stat label="Oil +$10" value={fmt.signedPct(d.oilPer10, 1)} />
              <Stat label="Events" value={fmt.signedPct(d.event, 0)} />
            </div>
          </div>
          <p className="mt-3 text-[12.5px] text-ink-2">
            <Badge kind="ai">Measured, not assumed</Badge> Weekly pattern and holidays dominate. Promotions at product-family level have only a small effect here, because they run on almost every day.
          </p>
          <div className="mt-4 rounded-xl border border-line bg-white/[0.02] p-3">
            <Slider
              label="Promotion planner: promoted items in the coming weeks"
              value={promoPlan}
              min={0}
              max={3}
              step={0.25}
              onChange={setPromoPlan}
              format={(v) => (v === 1 ? "as planned" : `${v.toFixed(2)}× plan`)}
              hint="What-if: scales planned promotions after the as-of date; the forecast, safety stock and optimisation update."
            />
            <p className="mt-1 text-[12px] text-ink-2">
              {promoPlan === 1
                ? "Using the promotions planned in the data (Kaggle test.csv for the first 16 days, then the recent typical level)."
                : promoPlan === 0
                  ? "No promotions planned. The regression and LSTM forecasts drop by the measured promotion effect."
                  : `Measured effect: about ${fmt.signedPct(Math.pow(1 + d.promoPerDoubling, Math.log2(promoPlan)) - 1, 1)} on daily demand at ${promoPlan.toFixed(2)}× the planned promotions (regression estimate).`}
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}

function ChartLite({ option }: { option: ChartOption }) {
  return <EChart option={option} height={200} ariaLabel="Weekday sales multipliers" />;
}
