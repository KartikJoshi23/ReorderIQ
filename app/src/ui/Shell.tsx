/** App shell: sidebar navigation, header (series, data mode, model status) and the policy control row. */
import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, Boxes, BrainCircuit, Database, LayoutDashboard, Loader2, Menu, Scale, ShieldCheck, Sparkles, TrendingUp, Warehouse, X } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { dateAt } from "../data/series";
import { fmt } from "../lib/format";
import { useApp } from "../state/store";
import { Badge, cx, Progress, Select, Slider } from "./components";

export const PAGES = [
  { id: "command", label: "Command Center", icon: LayoutDashboard },
  { id: "data", label: "Data & Updates", icon: Database },
  { id: "forecast", label: "Demand Forecast", icon: TrendingUp },
  { id: "safety", label: "Safety Stock & Risk", icon: ShieldCheck },
  { id: "simulation", label: "Inventory Simulation", icon: Boxes },
  { id: "optimise", label: "Optimisation", icon: Sparkles },
  { id: "impact", label: "Baseline vs AI", icon: Scale },
  { id: "method", label: "Method & Assumptions", icon: BookOpen },
] as const;

export type PageId = (typeof PAGES)[number]["id"];

export function useHashPage(): [PageId, (p: PageId) => void] {
  const read = (): PageId => {
    const h = window.location.hash.replace(/^#\/?/, "");
    return (PAGES.find((p) => p.id === h)?.id ?? "command") as PageId;
  };
  const [page, setPage] = useState<PageId>(read);
  useEffect(() => {
    const on = () => setPage(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [page, (p) => (window.location.hash = `/${p}`)];
}

function Nav({ page, onNavigate }: { page: PageId; onNavigate: (p: PageId) => void }) {
  return (
    <nav aria-label="Dashboard pages" className="flex flex-col gap-1">
      {PAGES.map(({ id, label, icon: Icon }) => {
        const active = id === page;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onNavigate(id)}
            aria-current={active ? "page" : undefined}
            className={cx(
              "focus-ring relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] transition-colors",
              active ? "text-ink" : "text-ink-3 hover:bg-white/[0.04] hover:text-ink-2",
            )}
          >
            {active && <motion.span layoutId="nav-active" className="absolute inset-0 rounded-lg border border-accent/30 bg-accent/12" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
            <Icon size={16} className="relative" aria-hidden />
            <span className="relative">{label}</span>
          </button>
        );
      })}
    </nav>
  );
}

function Brand() {
  const source = useApp((s) => s.source);
  return (
    <div className="flex items-center gap-2.5 px-1">
      <div className="grid h-9 w-9 place-items-center rounded-xl border border-line-strong bg-accent/15">
        <Warehouse size={18} className="text-accent" aria-hidden />
      </div>
      <div className="min-w-0">
        <div className="text-[14px] font-semibold tracking-tight">ReorderIQ</div>
        <div className="mt-0.5">{source === "demo" ? <Badge kind="demo">Synthetic demo data</Badge> : <Badge kind="real">Real data · Favorita</Badge>}</div>
      </div>
    </div>
  );
}

/** Switch between the real Favorita data and the synthetic demo series. */
function DataSourceSwitch() {
  const source = useApp((s) => s.source);
  const loadReal = useApp((s) => s.loadReal);
  const useDemo = useApp((s) => s.useDemo);
  const options = [
    { id: "real", label: "Real data", onSelect: () => void loadReal() },
    { id: "demo", label: "Synthetic demo", onSelect: useDemo },
  ] as const;
  return (
    <div className="px-1">
      <div className="mb-1 text-[11px] text-ink-3">Data source</div>
      <div role="radiogroup" aria-label="Data source" className="grid grid-cols-2 gap-1 rounded-lg border border-line p-1">
        {options.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={source === o.id}
            onClick={() => source !== o.id && o.onSelect()}
            className={cx("focus-ring rounded-md px-2 py-1 text-[12px] transition-colors", source === o.id ? "bg-accent/15 text-ink" : "text-ink-3 hover:text-ink-2")}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function ModelStatus() {
  const status = useApp((s) => s.forecastStatus);
  const progress = useApp((s) => s.forecastProgress);
  const forecast = useApp((s) => s.forecast);
  const error = useApp((s) => s.forecastError);
  if (status === "error") return <span className="text-[12px] text-[#f19a9a]">Model error: {error}</span>;
  if (status === "running") {
    const pct = progress ? Math.round(progress.fraction * 100) : 0;
    return (
      <div className="flex min-w-[220px] items-center gap-2" aria-live="polite">
        <Loader2 size={14} className="shrink-0 animate-spin text-accent" aria-hidden />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[11.5px] text-ink-2">
            {progress?.stage ?? "Starting models"}
            {progress?.lstm ? ` · epoch ${progress.lstm.epoch}/${progress.lstm.epochs}` : ""} · {pct}%
          </div>
          <Progress value={progress?.fraction ?? 0.02} label="Model training progress" />
        </div>
      </div>
    );
  }
  if (!forecast) return null;
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
      <BrainCircuit size={14} className="text-ai" aria-hidden />
      Model v{forecast.version} · {forecast.lstm ? `LSTM ${forecast.lstm.action}` : "ETS + Ridge"} · WAPE {fmt.pct(forecast.ensemble.metrics.wape)}
    </span>
  );
}

function ControlRow() {
  const serviceLevel = useApp((s) => s.serviceLevel);
  const leadTime = useApp((s) => s.leadTime);
  const setServiceLevel = useApp((s) => s.setServiceLevel);
  const setLeadTime = useApp((s) => s.setLeadTime);
  const setLeadTimeKind = useApp((s) => s.setLeadTimeKind);
  return (
    <section aria-label="Policy settings" className="glass grid grid-cols-1 gap-x-6 gap-y-3 px-4 py-3 sm:grid-cols-2 xl:grid-cols-4">
      <Slider label="Service level (Z)" value={serviceLevel} min={0.8} max={0.995} step={0.005} onChange={setServiceLevel} format={(v) => `${(v * 100).toFixed(1)}%`} />
      <Slider label="Lead time L" value={leadTime.mean} min={1} max={21} step={0.5} onChange={(v) => setLeadTime({ mean: v })} format={(v) => `${v.toFixed(1)} days`} />
      <Slider label="Lead-time variability σ_L" value={leadTime.sd} min={0} max={7} step={0.25} onChange={(v) => setLeadTime({ sd: v })} format={(v) => `${v.toFixed(2)} days`} />
      <Select
        label="Lead-time distribution"
        value={leadTime.kind}
        onChange={setLeadTimeKind}
        options={[
          { value: "gamma", label: "Gamma (right-skewed delays)" },
          { value: "normal", label: "Normal (formula's assumption)" },
          ...(leadTime.samples?.length ? [{ value: "empirical" as const, label: `Empirical (${leadTime.samples.length} records)` }] : []),
        ]}
      />
    </section>
  );
}

export function Shell({ page, onNavigate, children }: { page: PageId; onNavigate: (p: PageId) => void; children: ReactNode }) {
  const [drawer, setDrawer] = useState(false);
  const seriesOptions = useApp((s) => s.seriesOptions);
  const seriesId = useApp((s) => s.seriesId);
  const selectSeries = useApp((s) => s.selectSeries);
  const series = useApp((s) => s.series);
  const asOf = useApp((s) => s.asOf);
  const title = PAGES.find((p) => p.id === page)?.label ?? "";
  const go = (p: PageId) => {
    onNavigate(p);
    setDrawer(false);
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-[1680px]">
      <aside className="sticky top-0 hidden h-screen w-[248px] shrink-0 flex-col gap-6 border-r border-line px-4 py-5 lg:flex">
        <Brand />
        <Nav page={page} onNavigate={go} />
        <div className="mt-auto grid gap-3">
          <DataSourceSwitch />
          <p className="px-1 text-[11px] leading-relaxed text-ink-3">AI in Operations · Session II exercise. All models run in your browser.</p>
        </div>
      </aside>

      <AnimatePresence>
        {drawer && (
          <motion.div className="fixed inset-0 z-40 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <button type="button" aria-label="Close menu" className="absolute inset-0 bg-black/60" onClick={() => setDrawer(false)} />
            <motion.aside initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }} transition={{ type: "spring", stiffness: 380, damping: 36 }} className="glass absolute inset-y-0 left-0 flex w-[270px] flex-col gap-6 rounded-none px-4 py-5">
              <div className="flex items-start justify-between">
                <Brand />
                <button type="button" className="focus-ring rounded-md p-1 text-ink-3" onClick={() => setDrawer(false)} aria-label="Close menu">
                  <X size={18} />
                </button>
              </div>
              <Nav page={page} onNavigate={go} />
              <div className="mt-auto">
                <DataSourceSwitch />
              </div>
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col gap-4 px-4 pt-4 pb-10 sm:px-6">
        <header className="flex flex-wrap items-center gap-x-4 gap-y-3">
          <button type="button" className="focus-ring rounded-lg border border-line p-2 text-ink-2 lg:hidden" onClick={() => setDrawer(true)} aria-label="Open menu">
            <Menu size={18} />
          </button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-[20px] font-semibold tracking-tight sm:text-[22px]">{title}</h1>
            {series && <p className="text-[12.5px] text-ink-3">Actual sales known through {fmt.date(dateAt(series, asOf))} · orders are planned from the next day</p>}
          </div>
          <div className="w-full sm:w-[260px]">
            <Select
              label="Store · product family"
              value={seriesId ?? ""}
              onChange={(v) => selectSeries(v)}
              options={seriesOptions.map((o) => ({ value: o.id, label: `${o.label}${o.city ? ` (${o.city})` : ""}` }))}
            />
          </div>
          <div className="w-full sm:w-auto">
            <ModelStatus />
          </div>
        </header>
        <ControlRow />
        <AnimatePresence mode="wait">
          <motion.main key={page} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }} className="min-w-0">
            {children}
          </motion.main>
        </AnimatePresence>
      </div>
    </div>
  );
}
