/** Start screen: shown while the Favorita data loads, or if it cannot be loaded (with a demo fallback). */
import { motion } from "framer-motion";
import { FlaskConical, Loader2, RotateCcw, Warehouse } from "lucide-react";
import { useApp } from "../state/store";
import { Button } from "./components";

export function Start() {
  const phase = useApp((s) => s.phase);
  const error = useApp((s) => s.loadError);
  const loadReal = useApp((s) => s.loadReal);
  const useDemo = useApp((s) => s.useDemo);

  return (
    <main className="grid min-h-screen place-items-center px-4 py-10">
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="glass w-full max-w-[460px] p-6 sm:p-8">
        <div className="mb-5 flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-xl border border-line-strong bg-accent/15">
            <Warehouse className="text-accent" size={22} aria-hidden />
          </div>
          <div>
            <h1 className="text-[19px] font-semibold tracking-tight">ReorderIQ</h1>
            <p className="text-[12.5px] text-ink-3">AI-powered reorder points for grocery retail</p>
          </div>
        </div>

        {phase === "error" ? (
          <div className="space-y-3">
            <p role="alert" className="text-[13px] text-[#f19a9a]">
              {error}
            </p>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void loadReal()}>
                <RotateCcw size={14} aria-hidden /> Try again
              </Button>
              <Button variant="outline" onClick={useDemo}>
                <FlaskConical size={15} aria-hidden /> Explore with synthetic demo data
              </Button>
            </div>
          </div>
        ) : (
          <p className="flex items-center gap-2 text-[13px] text-ink-2" aria-live="polite">
            <Loader2 size={16} className="animate-spin text-accent" aria-hidden /> Loading Favorita grocery sales data…
          </p>
        )}
      </motion.div>
    </main>
  );
}
