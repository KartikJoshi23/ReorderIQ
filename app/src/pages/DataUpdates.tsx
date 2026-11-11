/** Data & Updates: data health, live replay of held-back sales, manual / CSV entry, lead-time
 *  records, model version log and live accuracy of issued forecasts (slide step 1: new data). */
import { FastForward, FileUp, Play, RotateCcw, SkipForward, Truck } from "lucide-react";
import { type ChangeEvent, type FormEvent, useMemo, useState } from "react";
import { dateAt, lastObservedIndex } from "../data/series";
import { parseObservationsCsv } from "../data/newData";
import { parseLeadTimeCsv } from "../inventory/leadTime";
import { fmt } from "../lib/format";
import { scoreIssued } from "../models/tracking";
import { useApp } from "../state/store";
import { axisTooltip, barStyle, dateAxis, line, unitsAxis } from "../ui/charts/common";
import { Button, Card, ChartCard, DataTable, Empty, Explain, Legend, NumberField, Stat, Status } from "../ui/components";
import { C, wash } from "../ui/tokens";

export function DataUpdates() {
  const series = useApp((s) => s.series);
  const asOf = useApp((s) => s.asOf);
  const source = useApp((s) => s.source);
  const msg = useApp((s) => s.dataMessage);
  const releaseDays = useApp((s) => s.releaseDays);
  const resetData = useApp((s) => s.resetData);
  const addObservations = useApp((s) => s.addObservations);
  const setLeadTime = useApp((s) => s.setLeadTime);
  const versionLog = useApp((s) => s.versionLog);
  const issued = useApp((s) => s.issued);
  const busy = useApp((s) => s.forecastStatus === "running");
  const retrain = useApp((s) => s.retrainLstm);
  const [lt, setLt] = useState<string | null>(null);

  const historyOption = useMemo(() => {
    if (!series) return null;
    const end = Math.max(asOf, lastObservedIndex(series));
    const idx = Array.from({ length: end + 1 }, (_, i) => i);
    const dates = idx.map((i) => dateAt(series, i));
    const known = idx.map((i) => (i <= asOf && !series.closed[i] ? (series.sales[i] as number) : null));
    const held = idx.map((i) => (i > asOf && !series.closed[i] && Number.isFinite(series.sales[i] as number) ? (series.sales[i] as number) : null));
    const added = idx.map((i) => (series.origin[i] === 1 && i <= asOf ? (series.sales[i] as number) : null));
    return {
      grid: { top: 16, left: 8, right: 16, bottom: 56, containLabel: true },
      tooltip: axisTooltip(),
      xAxis: dateAxis(dates),
      yAxis: unitsAxis(),
      dataZoom: [{ type: "inside" as const, start: 70, end: 100 }, { type: "slider" as const, start: 70, end: 100, height: 22, bottom: 8, borderColor: C.grid, textStyle: { color: C.ink3 }, fillerColor: wash(C.ai, 0.15), dataBackground: { lineStyle: { color: C.ink3 }, areaStyle: { color: wash(C.ink3, 0.1) } } }],
      series: [
        { ...line("Known to the models", known, C.actual, { width: 1.2 }), markLine: { symbol: "none", silent: true, data: [{ xAxis: dateAt(series, asOf), label: { formatter: "as of", color: C.ink2 }, lineStyle: { color: C.ai, type: "dashed" as const } }, { xAxis: dateAt(series, series.modelStart), label: { formatter: "model start", color: C.ink3 }, lineStyle: { color: C.ink3, type: "dotted" as const } }] } },
        line("Held back (not yet released)", held, C.ink3, { width: 1, dashed: true }),
        { ...line("Added by you", added, C.s4), showSymbol: true, symbolSize: 6 },
      ],
    };
  }, [series, asOf]);

  const live = useMemo(() => (series ? scoreIssued(issued, series, asOf, 7) : []), [issued, series, asOf]);
  const liveOption = useMemo(() => ({
    grid: { top: 16, left: 8, right: 16, bottom: 8, containLabel: true },
    tooltip: { trigger: "axis" as const, valueFormatter: (v: unknown) => fmt.pct(v as number) },
    xAxis: { type: "category" as const, data: live.map((l) => `v${l.version} · ${fmt.shortDate(l.asOfDate)}`) },
    yAxis: { type: "value" as const, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    series: [{ name: "WAPE on newly arrived days", type: "bar" as const, barMaxWidth: 24, data: live.map((l) => l.wape), itemStyle: barStyle(C.ai) }],
  }), [live]);

  if (!series || !historyOption) return <Empty>Loading data…</Empty>;
  const end = lastObservedIndex(series);
  const remaining = Math.max(0, end - asOf);
  let closedDays = 0;
  for (let i = 0; i <= asOf; i++) closedDays += series.closed[i] ?? 0;

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Data health" subtitle={`${source === "demo" ? "Synthetic demo series" : "Kaggle Store Sales (Favorita)"} · ${series.storeMeta.city || "—"}`}>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="History" value={`${dateAt(series, 0).slice(0, 4)} → ${fmt.date(dateAt(series, asOf))}`} sub={`${fmt.int(asOf + 1)} days from ${fmt.date(dateAt(series, 0))}`} />
            <Stat label="Model window from" value={fmt.date(dateAt(series, series.modelStart))} sub="promotions recorded from here" />
            <Stat label="Closed days" value={fmt.int(closedDays)} sub="excluded from training & stats" />
            <Stat label="Held back" value={`${fmt.int(remaining)} days`} sub={remaining ? `until ${fmt.shortDate(dateAt(series, end))}` : "all released"} />
          </div>
        </Card>

        <Card title="Live replay: new data arriving" subtitle="Real sales from the held-back final 8 weeks are released to the models">
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => releaseDays(1)} disabled={!remaining}>
              <Play size={14} aria-hidden /> Next day
            </Button>
            <Button onClick={() => releaseDays(7)} disabled={!remaining}>
              <SkipForward size={14} aria-hidden /> Next week
            </Button>
            <Button variant="outline" onClick={() => releaseDays(remaining)} disabled={!remaining}>
              <FastForward size={14} aria-hidden /> All
            </Button>
            <Button variant="ghost" onClick={resetData}>
              <RotateCcw size={14} aria-hidden /> Reset
            </Button>
          </div>
          {msg && <p className="mt-3" role="status"><Status tone={msg.tone === "good" ? "good" : "critical"}>{msg.text}</Status></p>}
          <Explain className="mt-3">Each release refits Holt-Winters and Ridge from scratch, <b className="text-ink">fine-tunes the LSTM</b> on the newest data, re-weights the ensemble and recalculates safety stock, Monte Carlo risk and the optimised policy.{busy ? " Updating now…" : ""}</Explain>
          <Button variant="ghost" size="sm" className="mt-2" onClick={retrain} disabled={busy}>Retrain LSTM from scratch</Button>
        </Card>

        <AddData seriesFamily={series.family} store={series.storeMeta.store} nextDate={dateAt(series, asOf + 1)} onAdd={addObservations} onLeadTimes={(text) => {
          const r = parseLeadTimeCsv(text);
          if (r.samples.length >= 2) {
            setLeadTime({ kind: "empirical", samples: r.samples, mean: r.mean, sd: r.sd });
            setLt(`Loaded ${r.samples.length} lead times: mean ${r.mean.toFixed(1)} d, σ_L ${r.sd.toFixed(2)} d. The empirical distribution is now used.${r.errors.length ? ` ${r.errors.length} rows skipped.` : ""}`);
          } else setLt(r.errors[0]?.message ?? "Need at least two valid lead times.");
        }} ltMessage={lt} />
      </div>

      <ChartCard
        title="Sales history and what the models know"
        subtitle="Drag the slider to zoom; the dashed line marks the current 'as of' date"
        option={historyOption}
        height={320}
        ariaLabel="Daily sales history with the as-of date and held-back data"
        legend={<Legend items={[{ label: "Known to the models", color: C.actual }, { label: "Held back (not yet released)", color: C.ink3, kind: "dash" }, { label: "Added by you", color: C.s4, kind: "dot" }]} />}
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Model version log" subtitle="Every update is recorded; WAPE is the rolling-backtest accuracy at that version">
          {versionLog.length ? (
            <DataTable maxHeight={260} spec={{ columns: ["Version", "As of", "Action", "WAPE", "Time"], rows: [...versionLog].reverse().map((v) => [`v${v.version}`, fmt.date(v.asOfDate), v.action, fmt.pct(v.wape), `${(v.ms / 1000).toFixed(1)} s`]) }} />
          ) : (
            <Empty>No versions yet.</Empty>
          )}
        </Card>
        <ChartCard
          title="Live accuracy on data the models had not seen"
          subtitle="WAPE of each issued forecast over its first 7 days, scored as actuals arrive"
          option={liveOption}
          height={240}
          ariaLabel="Bar chart of live forecast accuracy per model version"
          table={{ columns: ["Version", "Issued as of", "Days scored", "WAPE", "Bias", "80% band coverage"], rows: live.map((l) => [`v${l.version}`, l.asOfDate, l.n, fmt.pct(l.wape), fmt.signedPct(l.bias), fmt.pct(l.coverage80, 0)]) }}
          explain={live.length ? <>Latest scored forecast: WAPE {fmt.pct(live.at(-1)?.wape)}, with {fmt.pct(live.at(-1)?.coverage80, 0)} of actual days inside the 80% band.</> : <>Release a day or a week of data to score the forecasts already issued.</>}
        />
      </div>
    </div>
  );
}

function AddData({ seriesFamily, store, nextDate, onAdd, onLeadTimes, ltMessage }: { seriesFamily: string; store: number; nextDate: string; onAdd: (rows: { date: string; sales: number; promo?: number }[]) => boolean; onLeadTimes: (text: string) => void; ltMessage: string | null }) {
  const [sales, setSales] = useState(0);
  const [promo, setPromo] = useState(0);
  const [csvMsg, setCsvMsg] = useState<string | null>(null);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    onAdd([{ date: nextDate, sales, promo }]);
  };
  const readFile = (e: ChangeEvent<HTMLInputElement>, then: (text: string) => void) => {
    const file = e.target.files?.[0];
    if (!file) return;
    void file.text().then(then);
    e.target.value = "";
  };

  return (
    <Card title="Add new data" subtitle="Manual entry or CSV upload, plus supplier lead-time records">
      <form onSubmit={submit} className="grid grid-cols-2 gap-2">
        <NumberField label={`Units sold on ${fmt.date(nextDate)}`} value={sales} min={0} onChange={setSales} />
        <NumberField label="Items on promotion" value={promo} min={0} onChange={(v) => setPromo(Math.round(v))} />
        <Button type="submit" className="col-span-2" variant="outline">
          Add day &amp; update models
        </Button>
      </form>
      <div className="mt-3 grid gap-2">
        <label className="focus-ring flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-line-strong px-3 py-2 text-[12.5px] text-ink-2 hover:bg-white/[0.03]">
          <FileUp size={15} aria-hidden /> Upload sales CSV (date,sales[,onpromotion] or Kaggle train.csv format)
          <input
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            onChange={(e) =>
              readFile(e, (text) => {
                const r = parseObservationsCsv(text, { store, family: seriesFamily });
                if (r.errors.length) setCsvMsg(r.errors.slice(0, 3).map((x) => `line ${x.line}: ${x.message}`).join(" · "));
                else setCsvMsg(onAdd(r.rows) ? `Read ${r.rows.length} row(s)${r.skipped ? ` (${r.skipped} for other items skipped)` : ""}.` : null);
              })
            }
          />
        </label>
        {csvMsg && <p className="text-[12px] text-ink-2">{csvMsg}</p>}
        <label className="focus-ring flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-line-strong px-3 py-2 text-[12.5px] text-ink-2 hover:bg-white/[0.03]">
          <Truck size={15} aria-hidden /> Upload lead-time records (order_date,receipt_date)
          <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => readFile(e, onLeadTimes)} />
        </label>
        {ltMessage && <p className="text-[12px] text-ink-2">{ltMessage}</p>}
      </div>
    </Card>
  );
}
