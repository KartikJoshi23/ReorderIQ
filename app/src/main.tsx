import "@fontsource-variable/inter";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "./styles/index.css";
import { MotionConfig } from "framer-motion";
import { type ComponentType, lazy, StrictMode, Suspense, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { useApp } from "./state/store";
import { Empty } from "./ui/components";
import { type PageId, Shell, useHashPage } from "./ui/Shell";
import { Start } from "./ui/Start";

/** Pages are code-split: each loads (with the chart library) the first time it is opened. */
const page = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) => lazy(() => load().then((m) => ({ default: m[name] })));

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
      <Suspense fallback={<Empty>Loading page…</Empty>}>
        <Page />
      </Suspense>
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
