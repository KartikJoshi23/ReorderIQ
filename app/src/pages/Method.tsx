/** Method & Assumptions: formulas, units, data handling, models, assumptions and limitations. */
import { useApp } from "../state/store";
import { Card, DataTable } from "../ui/components";

export function Method() {
  const source = useApp((s) => s.source);
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="The exercise, step by step">
        <ol className="list-decimal space-y-2 pl-5 text-[13.5px] leading-relaxed text-ink-2">
          <li>
            <b className="text-ink">Demand forecasting.</b> Three model families run in your browser:
            <ul className="mt-1 list-disc space-y-1 pl-5">
              <li><b className="text-ink">Time series:</b> Holt-Winters ETS.</li>
              <li><b className="text-ink">Regression:</b> Ridge, using weekday, holidays, promotions, paydays, Christmas, oil and lags.</li>
              <li><b className="text-ink">Deep learning:</b> an LSTM in TensorFlow.js.</li>
            </ul>
            An adaptive ensemble combines them, weighted by recent out-of-sample accuracy. New data refits ETS and Ridge, fine-tunes the LSTM and re-weights the ensemble.
          </li>
          <li>
            <b className="text-ink">Dynamic safety stock.</b> The slide&apos;s formula SS = Z·√(σ_D²·L + D̄²·σ_L²) is fed in two ways:
            <ul className="mt-1 list-disc space-y-1 pl-5">
              <li><b className="text-ink">Baseline:</b> 365-day average and SD.</li>
              <li><b className="text-ink">AI:</b> forecast demand over the lead time, and forecast-error RMSE.</li>
            </ul>
            A Monte Carlo simulation (forecast × real out-of-sample errors × random lead times) measures stockout risk and the buffer actually needed, including under stress scenarios.
          </li>
          <li>
            <b className="text-ink">AI-driven sensitivity analysis.</b> Bayesian optimisation (Gaussian process with constrained Expected Improvement) searches the buffer and order quantity that minimise holding, ordering and lost-sales cost. It does this subject to fill-rate, storage, budget and delivery limits, using a few dozen simulations instead of a manual sweep. A 900-point grid is available as a check.
          </li>
        </ol>
      </Card>
      <Card title="Units and definitions">
        <DataTable
          spec={{
            columns: ["Symbol", "Meaning", "Unit"],
            rows: [
              ["D̄", "average daily demand (baseline: last 365 open days; AI: forecast over the lead time ÷ L)", "units/day"],
              ["σ_D / σ_e", "demand SD (baseline) / out-of-sample forecast RMSE, horizons 1…L (AI)", "units/day"],
              ["L", "mean replenishment lead time", "days"],
              ["σ_L", "lead-time standard deviation", "days"],
              ["Z", "Φ⁻¹(service level), cycle service level", "—"],
              ["SS", "safety stock", "units"],
              ["ROP", "reorder point = expected lead-time demand + SS", "units"],
              ["WAPE", "Σ|actual − forecast| ÷ Σ actual", "%"],
              ["MASE", "MAE ÷ in-sample seasonal-naive MAE", "ratio"],
              ["Fill rate", "units served from stock ÷ units demanded", "%"],
            ],
          }}
        />
      </Card>
      <Card title="Data">
        <ul className="list-disc space-y-1.5 pl-5 text-[13.5px] leading-relaxed text-ink-2">
          {source === "demo" ? (
            <li><b className="text-ink">Demo mode:</b> the series is simulated in your browser from aggregate statistics (weekday pattern, trend, seasonality, promotion uplift, autocorrelated noise). It contains no real sales.</li>
          ) : (
            <li>Kaggle <i>Store Sales – Time Series Forecasting</i> (Corporación Favorita, Ecuador): 9 curated non-perishable store × family series, daily 2013–2017, with planned promotions to 31 Aug 2017.</li>
          )}
          <li>
            <b className="text-ink">Cleaning:</b>
            <ul className="mt-1 list-disc space-y-1 pl-5">
              <li>25 Dec and 1 Jan are treated as closed and excluded from statistics.</li>
              <li>Models train from 1 Apr 2014, when promotion records start.</li>
              <li>Oil prices are interpolated and lagged 7 days.</li>
              <li>Holidays handle transfers and bridges, and are matched to each store&apos;s city and region.</li>
            </ul>
          </li>
          <li>The final 8 weeks are held back and released through <b className="text-ink">Data &amp; Updates</b>, so every update is scored on data the models never saw.</li>
          <li>One planning item = one product family at one store. Favorita does not publish SKU-level data.</li>
          <li>All processing happens in the browser; nothing you enter is uploaded. The Kaggle data is used for academic education under the competition rules and is shown here for course review only.</li>
        </ul>
      </Card>
      <Card title="Assumptions and limitations">
        <DataTable
          spec={{
            columns: ["Assumption", "Default", "Why / how to change"],
            rows: [
              ["Lead time", "Gamma, L = 7 d, σ_L = 2 d", "No supplier lead times in the data; adjustable, or upload order/receipt records"],
              ["Service level", "95% cycle service", "Control row above every page"],
              ["Unit cost / holding", "$0.70, 25% per year", "Optimisation page"],
              ["Order cost", "$120 per order", "Optimisation page"],
              ["Lost-sale penalty", "$0.50 per unit", "Lost margin plus goodwill; Optimisation page"],
              ["Storage / budget", "18 / 12 days of demand", "Scaled to the selected item; Optimisation page"],
              ["Policy", "Continuous review (R, Q), lost sales", "Standard for retail shelves"],
              ["Unknown future promotions", "Median of the last 28 days", "Planned promotions are used where known"],
            ],
          }}
        />
        <p className="mt-3 text-[12.5px] leading-relaxed text-ink-3">
          Promotions at product-family level are almost always running, so their measured effect is small. Weekday, holiday, December and trend effects dominate. The formula assumes normal lead-time demand, and the Monte Carlo shows where that under-protects.
        </p>
      </Card>
    </div>
  );
}
