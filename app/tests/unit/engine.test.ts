/**
 * End-to-end engine tests on the synthetic demo series, including the LSTM (TF.js WebAssembly
 * backend) with a reduced training budget, and the continuous-update path (fine-tune, not retrain).
 */
import { describe, expect, it } from "vitest";
import { DEMO_SERIES_ID, generateDemoPack } from "../../src/data/demo";
import { initialAsOf, releaseDays, seriesFromPack } from "../../src/data/series";
import { CancelledError, DEFAULT_ENGINE_SETTINGS, type EngineProgress, ForecastEngine } from "../../src/models/engine";
import { DEFAULT_LSTM_CONFIG } from "../../src/models/lstm";

const s = seriesFromPack(generateDemoPack(2026), DEMO_SERIES_ID);
const asOf = initialAsOf(s);

describe("ForecastEngine without LSTM", () => {
  it("returns backtests, forecasts, ensemble, bands, errors and drivers", async () => {
    const engine = new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, useLstm: false });
    const r = await engine.run(s, asOf);
    expect(r.version).toBe(1);
    expect(r.dates).toHaveLength(28);
    expect(r.models.ets?.folds).toHaveLength(6);
    expect(r.models.lstm).toBeUndefined();
    expect((r.ensemble.weights.ets ?? 0) + (r.ensemble.weights.ridge ?? 0)).toBeCloseTo(1, 12);
    // AI models should beat the moving-average benchmark out of sample on this series
    expect(r.ensemble.metrics.wape).toBeLessThan(r.baselines.ma28.metrics.wape);
    expect(r.ensemble.metrics.mase).toBeLessThan(1.2);
    for (let k = 0; k < 28; k++) {
      expect(r.bands.lo95[k]).toBeLessThanOrEqual(r.ensemble.forecast[k] as number);
      expect(r.bands.hi95[k]).toBeGreaterThanOrEqual(r.ensemble.forecast[k] as number);
    }
    expect(r.foldErrors).toHaveLength(6);
    expect(r.drivers.weekday[6]).toBeGreaterThan(1.2); // Sunday uplift vs Monday
  });
});

describe("ForecastEngine cancellation", () => {
  it(
    "abandons a run mid-LSTM-training and stays usable afterwards",
    async () => {
      const engine = new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, lstm: { ...DEFAULT_LSTM_CONFIG, epochs: 30, patience: 30 } });
      let epochs = 0;
      const running = engine.run(s, asOf, (p) => {
        if (p.lstm) epochs = p.lstm.epoch;
        if (p.lstm && p.lstm.epoch === 2) engine.cancelAll();
      });
      await expect(running).rejects.toBeInstanceOf(CancelledError);
      expect(epochs).toBeLessThan(5); // stopped within an epoch of the cancel, not after 30
      const ok = await engine.run(s, asOf, undefined, { useLstm: false });
      expect(ok.models.ets).toBeDefined();
      await expect(engine.run(s, asOf, undefined, { useLstm: false, generation: -1 })).rejects.toBeInstanceOf(CancelledError);
      engine.dispose();
    },
    120_000,
  );
});

describe("ForecastEngine with LSTM and continuous updating", () => {
  const cfg = { ...DEFAULT_LSTM_CONFIG, epochs: 4, patience: 2, fineTuneEpochs: 1 };
  const engine = new ForecastEngine({ ...DEFAULT_ENGINE_SETTINGS, lstm: cfg });

  it(
    "trains the LSTM through the rolling folds, then fine-tunes when a new week arrives",
    async () => {
      const stages: string[] = [];
      const first = await engine.run(s, asOf, (p: EngineProgress) => stages.push(p.stage));
      expect(first.lstm?.action).toBe("trained");
      expect(first.lstm?.backend).toBe("wasm");
      expect(first.models.lstm?.folds).toHaveLength(6);
      expect(Number.isFinite(first.models.lstm?.metrics.wape)).toBe(true);
      expect(Object.keys(first.ensemble.weights).sort()).toEqual(["ets", "lstm", "ridge"]);
      expect(stages).toContain("Training LSTM");

      const nextAsOf = releaseDays(s, asOf, 7);
      const second = await engine.run(s, nextAsOf);
      expect(second.version).toBe(2);
      expect(second.lstm?.action).toBe("fine-tuned");
      expect(second.lstm?.backtestVersion).toBe(1);
      expect(second.dates[0]).not.toBe(first.dates[0]);
      expect(second.ensemble.forecast).not.toEqual(first.ensemble.forecast);

      const third = await engine.run(s, nextAsOf, undefined, { retrainLstm: true });
      expect(third.lstm?.action).toBe("trained");
      expect(third.lstm?.backtestVersion).toBe(3);
      engine.dispose();
    },
    240_000,
  );
});
