import "@fontsource-variable/inter";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "./styles/index.css";
import { MotionConfig } from "framer-motion";
import { Component, type ComponentType, type ReactNode, lazy, StrictMode, Suspense, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { useApp } from "./state/store";
import { Empty } from "./ui/components";
import { type PageId, Shell, useHashPage } from "./ui/Shell";
import { Start } from "./ui/Start";

/**
 * If a page chunk fails to load, the open tab is most likely from an older deploy whose files are
 * gone: reload once to fetch the current version (the flag prevents a reload loop).
 */
const RELOAD_FLAG = "reorderiq:chunk-reload";
function recoverFromStaleDeploy(error: unknown): never {
  let reloaded = false;
  try {
    reloaded = sessionStorage.getItem(RELOAD_FLAG) === "1";
    if (!reloaded) sessionStorage.setItem(RELOAD_FLAG, "1");
  } catch {
    reloaded = true; // storage unavailable: do not risk a loop
  }
  if (!reloaded) window.location.reload();
  throw error;
}

/** Pages are code-split: each loads (with the chart library) the first time it is opened. */
const page = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) =>
  lazy(() =>
    load().then(
      (m) => {
        try {
          sessionStorage.removeItem(RELOAD_FLAG);
        } catch {
          /* storage unavailable */
        }
        return { default: m[name] };
      },
      recoverFromStaleDeploy,
    ),
  );

const PAGE_COMPONENTS: Record<PageId, ComponentType> = {
  command: page(() => import("./pages/CommandCenter"), "CommandCenter"),
  data: page(() => import("./pages/DataUpdates"), "DataUpdates"),
  forecast: page(() => import("./pages/Forecast"), "Forecast"),
  safety: page(() => import("./pages/SafetyStock"), "SafetyStock"),
  simulation: page(() => import("./pages/Simulation"), "Simulation"),
  optimise: page(() => import("./pages/Optimisation"), "Optimisation"),
  impact: page(() => import("./pages/Impact"), "Impact"),
  method: page(() => import("./pages/Method"), "Method"),
};

/** Keeps a failure on one page from blanking the whole dashboard; navigation keeps working. */
class PageBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="glass grid gap-3 p-6 text-[13.5px] text-ink-2">
        <p className="text-ink">This page could not be displayed.</p>
        <p className="text-[12.5px] text-ink-3">{this.state.error.message}</p>
        <div>
          <button type="button" className="rounded-lg border border-line-strong px-3 py-1.5 text-ink hover:bg-white/5" onClick={() => window.location.reload()}>
            Reload the dashboard
          </button>
        </div>
      </div>
    );
  }
}

let started = false;

function App() {
  const phase = useApp((s) => s.phase);
  const loadReal = useApp((s) => s.loadReal);
  const [current, setPage] = useHashPage();
  // Load the real Favorita data once on start-up (the store keeps it for the session).
  useEffect(() => {
    if (started) return;
    started = true;
    void loadReal();
  }, [loadReal]);
  if (phase !== "ready") return <Start />;
  const Page = PAGE_COMPONENTS[current];
  return (
    <Shell page={current} onNavigate={setPage}>
      <PageBoundary key={current}>
        <Suspense fallback={<Empty>Loading page…</Empty>}>
          <Page />
        </Suspense>
      </PageBoundary>
    </Shell>
  );
}

// Reuse the React root if this module is re-executed (development hot reload).
const host = document.getElementById("root") as (HTMLElement & { __reactRoot?: Root }) | null;
if (!host) throw new Error("Root element #root not found");
host.__reactRoot ??= createRoot(host);
host.__reactRoot.render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </StrictMode>,
);
