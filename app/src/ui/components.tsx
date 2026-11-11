/**
 * Design-system primitives: glass cards, KPI tiles, chart cards with table view, plain-language
 * callouts, sliders, buttons, badges, tables. Text always uses text tokens; colour carries identity
 * only on marks and small key swatches.
 */
import { animate, motion, useMotionValue, useReducedMotion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, Lightbulb, Table2, XCircle } from "lucide-react";
import { lazy, type ReactNode, Suspense, useEffect, useId, useState } from "react";
import type { ChartOption } from "./charts/EChart";

/** ECharts is loaded on demand, so the start screen and shell do not wait for the chart library. */
const EChart = lazy(() => import("./charts/EChart").then((m) => ({ default: m.EChart })));

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------------------------------
export function Card({ title, subtitle, actions, children, className, delay = 0 }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: "easeOut" }}
      className={cx("glass min-w-0 p-4 sm:p-5", className)}
    >
      {(title || actions) && (
        <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold tracking-tight text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[12.5px] leading-snug text-ink-3">{subtitle}</p>}
          </div>
          {actions && <div className="flex min-w-0 max-w-full flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </motion.section>
  );
}

// ---------------------------------------------------------------------------------------------
export interface TableSpec {
  columns: string[];
  rows: (string | number)[][];
}

export function ChartCard({ title, subtitle, option, height = 300, table, explain, actions, legend, className, delay, ariaLabel }: {
  title: ReactNode;
  subtitle?: ReactNode;
  option: ChartOption;
  height?: number;
  table?: TableSpec;
  explain?: ReactNode;
  actions?: ReactNode;
  legend?: ReactNode;
  className?: string;
  delay?: number;
  ariaLabel: string;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <Card
      title={title}
      subtitle={subtitle}
      className={className}
      delay={delay}
      actions={
        <>
          {actions}
          {table && (
            <Button variant="ghost" size="sm" onClick={() => setShowTable((v) => !v)} aria-pressed={showTable}>
              <Table2 size={14} aria-hidden /> {showTable ? "Chart" : "Table"}
            </Button>
          )}
        </>
      }
    >
      {legend && <div className="mb-2">{legend}</div>}
      {showTable && table ? (
        <DataTable spec={table} maxHeight={height} />
      ) : (
        <Suspense fallback={<div style={{ height }} aria-hidden />}>
          <EChart option={option} height={height} ariaLabel={ariaLabel} />
        </Suspense>
      )}
      {explain && <Explain className="mt-3">{explain}</Explain>}
    </Card>
  );
}

export function Legend({ items }: { items: { label: string; color: string; kind?: "line" | "dash" | "area" | "bar" | "dot" }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-ink-2">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <LegendKey color={it.color} kind={it.kind ?? "line"} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function LegendKey({ color, kind }: { color: string; kind: "line" | "dash" | "area" | "bar" | "dot" }) {
  if (kind === "bar" || kind === "area")
    return <span aria-hidden className="inline-block h-2.5 w-3.5 rounded-[3px]" style={{ background: color, opacity: kind === "area" ? 0.35 : 1 }} />;
  if (kind === "dot") return <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />;
  return (
    <svg aria-hidden width="18" height="8" viewBox="0 0 18 8">
      <line x1="1" y1="4" x2="17" y2="4" stroke={color} strokeWidth="2" strokeLinecap="round" strokeDasharray={kind === "dash" ? "4 3" : undefined} />
    </svg>
  );
}

export function DataTable({ spec, maxHeight = 320 }: { spec: TableSpec; maxHeight?: number }) {
  return (
    <div className="overflow-auto rounded-lg border border-line" style={{ maxHeight }}>
      <table className="w-full text-left text-[12px]">
        <thead className="sticky top-0 bg-panel text-ink-3">
          <tr>
            {spec.columns.map((c) => (
              <th key={c} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="num text-ink-2">
          {spec.rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((v, j) => (
                <td key={j} className={cx("px-3 py-1.5 whitespace-nowrap", j === 0 && "text-ink")}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export function Explain({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cx("flex gap-2.5 rounded-xl border border-line bg-white/[0.025] px-3 py-2.5 text-[13px] leading-relaxed text-ink-2", className)}>
      <Lightbulb size={16} className="mt-0.5 shrink-0 text-warning" aria-hidden />
      <div>
        <span className="sr-only">What this means: </span>
        {children}
      </div>
    </div>
  );
}

export type Tone = "good" | "warning" | "critical" | "neutral";

export function Status({ tone, children }: { tone: Tone; children: ReactNode }) {
  const Icon = tone === "good" ? CheckCircle2 : tone === "warning" ? AlertTriangle : tone === "critical" ? XCircle : Info;
  const color = tone === "good" ? "text-good" : tone === "warning" ? "text-warning" : tone === "critical" ? "text-critical" : "text-ink-3";
  return (
    <span className="inline-flex items-center gap-1 text-[12px] text-ink-2">
      <Icon size={14} className={color} aria-hidden />
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------------------------
export function AnimatedNumber({ value, format }: { value: number; format: (v: number) => string }) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(value);
  const [text, setText] = useState(format(value));
  useEffect(() => {
    // Jump (no tween) when there is nothing finite to animate from or to, or motion is reduced.
    if (!Number.isFinite(value) || !Number.isFinite(mv.get()) || reduce) {
      mv.set(value);
      setText(format(value));
      return;
    }
    const controls = animate(mv, value, { duration: 0.8, ease: "easeOut", onUpdate: (v) => setText(format(v)) });
    return () => controls.stop();
  }, [value, format, mv, reduce]);
  return <>{text}</>;
}

export function Kpi({ label, value, format, unit, note, tone, badge, delay = 0 }: {
  label: string;
  value: number;
  format: (v: number) => string;
  unit?: string;
  note?: ReactNode;
  tone?: Tone;
  badge?: ReactNode;
  delay?: number;
}) {
  return (
    <Card delay={delay} className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[12px] font-medium tracking-wide text-ink-3 uppercase">{label}</span>
        {badge}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="num text-[26px] leading-tight font-semibold text-ink sm:text-[28px]">
          <AnimatedNumber value={value} format={format} />
        </span>
        {unit && <span className="text-[12px] text-ink-3">{unit}</span>}
      </div>
      {note && <div className="text-[12px] leading-snug">{tone ? <Status tone={tone}>{note}</Status> : <span className="text-ink-2">{note}</span>}</div>}
    </Card>
  );
}

export function Badge({ kind, children }: { kind: "ai" | "baseline" | "mc" | "neutral" | "real" | "demo"; children: ReactNode }) {
  const styles: Record<string, string> = {
    ai: "border-ai/40 text-[#c4bdf5] bg-ai/10",
    baseline: "border-s-base/40 text-[#f0a887] bg-s-base/10",
    mc: "border-s-mc/40 text-[#7fd6b6] bg-s-mc/10",
    neutral: "border-line-strong text-ink-2 bg-white/5",
    real: "border-good/40 text-[#8fe08f] bg-good/10",
    demo: "border-warning/40 text-[#fcd27a] bg-warning/10",
  };
  return <span className={cx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap", styles[kind])}>{children}</span>;
}

// ---------------------------------------------------------------------------------------------
export function Button({ variant = "primary", size = "md", className, children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline"; size?: "sm" | "md" }) {
  const v = {
    primary: "bg-accent text-[#06101f] hover:bg-[#6da7ec] disabled:bg-white/10 disabled:text-ink-3",
    ghost: "text-ink-2 hover:bg-white/[0.06] hover:text-ink disabled:text-ink-3",
    outline: "border border-line-strong text-ink hover:bg-white/[0.06] disabled:text-ink-3",
  }[variant];
  const s = size === "sm" ? "h-7 px-2.5 text-[12px]" : "h-9 px-3.5 text-[13px]";
  return (
    <button type="button" className={cx("focus-ring inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:cursor-not-allowed", v, s, className)} {...rest}>
      {children}
    </button>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  hint?: string;
}) {
  const id = useId();
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[12px] font-medium text-ink-2">
          {label}
        </label>
        <span className="num text-[13px] text-ink">{format(value)}</span>
      </div>
      <input id={id} type="range" className="range focus-ring" min={min} max={max} step={step} value={value} style={{ ["--fill" as string]: fill }} onChange={(e) => onChange(Number(e.target.value))} aria-describedby={hint ? `${id}-h` : undefined} />
      {hint && (
        <p id={`${id}-h`} className="text-[11px] text-ink-3">
          {hint}
        </p>
      )}
    </div>
  );
}

export function Select<T extends string>({ label, value, options, onChange, className }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; className?: string }) {
  const id = useId();
  return (
    <div className={cx("min-w-0", className)}>
      <label htmlFor={id} className="mb-1 block text-[12px] font-medium text-ink-2">
        {label}
      </label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value as T)} className="focus-ring h-9 w-full rounded-lg border border-line-strong bg-panel px-2.5 text-[13px] text-ink">
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function NumberField({ label, value, onChange, step = 1, min, max, suffix }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; suffix?: string }) {
  const id = useId();
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1 block text-[12px] font-medium text-ink-2">
        {label}
      </label>
      <div className="flex items-center gap-1.5">
        <input
          id={id}
          type="number"
          className="focus-ring num h-9 w-full rounded-lg border border-line-strong bg-panel px-2.5 text-[13px] text-ink"
          value={Number.isFinite(value) ? value : ""}
          step={step}
          min={min}
          max={max}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (e.target.value !== "" && Number.isFinite(v)) onChange(v);
          }}
        />
        {suffix && <span className="shrink-0 text-[12px] text-ink-3">{suffix}</span>}
      </div>
    </div>
  );
}

export function Progress({ value, label }: { value: number; label?: string }) {
  return (
    <div className="w-full" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)} aria-label={label}>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
        <motion.div className="h-full rounded-full bg-accent" animate={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} transition={{ duration: 0.3 }} />
      </div>
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-white/[0.02] px-3 py-2">
      <div className="text-[11.5px] text-ink-3">{label}</div>
      <div className="num mt-0.5 text-[16px] font-semibold text-ink">{value}</div>
      {sub && <div className="text-[11.5px] text-ink-3">{sub}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="grid min-h-40 place-items-center rounded-xl border border-dashed border-line p-6 text-center text-[13px] text-ink-3">{children}</div>;
}
