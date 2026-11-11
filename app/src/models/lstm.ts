/**
 * Deep-learning model: LSTM network (TensorFlow.js) trained in the browser.
 *
 * Architecture (chosen in the notebook):
 *   past 28 days × [log-sales deviation from 28-day level, weekday sin/cos, open flag] → LSTM(32)
 *   ⊕ target-day drivers [weekday sin/cos, yearly sin/cos, promotions, holiday, event, earthquake,
 *     payday, Christmas ramp]                                                            → Dense(16, ReLU) → Dense(1)
 * Target = log(1 + sales_t) − mean of the previous 28 days (level-free, so trend does not break it),
 * scaled by its training SD. Forecasts are recursive. Training is seeded and deterministic on the
 * CPU backend: chronological validation split (early stopping) + seeded shuffling of the training part.
 *
 * Continuous updating: `fineTuneLstm` continues training the existing network on the most recent
 * year of data after new observations arrive (warm start), instead of retraining from scratch.
 */
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm"; // registers the WebAssembly backend
import { dayParts } from "../data/dates";
import type { SeriesData } from "../data/series";
import { createRng } from "../lib/random";
import { type History, isClosed, level28, smearingFactor, toUnits } from "./history";

export interface LstmConfig {
  window: number;
  units: number;
  dense: number;
  epochs: number;
  patience: number;
  batchSize: number;
  learningRate: number;
  validationFraction: number;
  fineTuneEpochs: number;
  fineTuneDays: number;
  fineTuneLearningRate: number;
  seed: number;
}

export const DEFAULT_LSTM_CONFIG: LstmConfig = {
  window: 28,
  units: 32,
  dense: 16,
  epochs: 40,
  patience: 5,
  batchSize: 32,
  learningRate: 0.005,
  validationFraction: 0.15,
  fineTuneEpochs: 4,
  fineTuneDays: 365,
  fineTuneLearningRate: 0.002,
  seed: 7,
};

const SEQ_FEATURES = 4;
const EXO_FEATURES = 10;

export interface LstmProgress {
  stage: "train" | "fine-tune";
  epoch: number;
  epochs: number;
  loss: number;
  valLoss?: number;
}

export interface LstmModel {
  net: tf.LayersModel;
  config: LstmConfig;
  /** SD of the training target (log-deviation scale). */
  scale: number;
  /** Max of log(1 + promotions) in training, used to scale the promotion input. */
  promoScale: number;
  origin: number;
  smear: number;
  epochsRun: number;
  samples: number;
}

let backendReady: Promise<string> | null = null;

/**
 * Initialise TensorFlow.js once. WebAssembly (SIMD, multi-threaded when the page is cross-origin
 * isolated) trains this small recurrent net ~3–4× faster than the plain-JS CPU backend; the CPU
 * backend is the fallback. Resolves to the backend actually in use.
 */
export function initTensorflow(preferred: "wasm" | "cpu" = "wasm"): Promise<string> {
  backendReady ??= (async () => {
    for (const name of preferred === "wasm" ? ["wasm", "cpu"] : ["cpu"]) {
      try {
        if (await tf.setBackend(name)) {
          await tf.ready();
          return name;
        }
      } catch {
        // try the next backend
      }
    }
    throw new Error("No TensorFlow.js backend could be initialised");
  })();
  return backendReady;
}

const TWO_PI = 2 * Math.PI;

function exoRow(s: SeriesData, t: number, promoScale: number, out: Float32Array, o: number): void {
  const p = dayParts(s.epoch0 + t);
  out[o] = Math.sin((TWO_PI * p.dow) / 7);
  out[o + 1] = Math.cos((TWO_PI * p.dow) / 7);
  out[o + 2] = Math.sin((TWO_PI * p.dayOfYear) / 365.25);
  out[o + 3] = Math.cos((TWO_PI * p.dayOfYear) / 365.25);
  out[o + 4] = Math.log1p(s.promo[t] ?? 0) / promoScale;
  out[o + 5] = s.holiday[t] ?? 0;
  out[o + 6] = s.event[t] ?? 0;
  out[o + 7] = s.quake[t] ?? 0;
  out[o + 8] = p.day === 15 || p.isMonthEnd ? 1 : 0;
  out[o + 9] = p.month === 12 && p.day <= 24 ? p.day / 24 : 0;
}

function seqRows(s: SeriesData, y: ArrayLike<number>, openFlag: (j: number) => number, t: number, scale: number, W: number, out: Float32Array, o: number): void {
  const lvl = level28(y, t);
  for (let k = 0; k < W; k++) {
    const j = t - W + k;
    const p = dayParts(s.epoch0 + j);
    const base = o + k * SEQ_FEATURES;
    out[base] = ((y[j] as number) - lvl) / scale;
    out[base + 1] = Math.sin((TWO_PI * p.dow) / 7);
    out[base + 2] = Math.cos((TWO_PI * p.dow) / 7);
    out[base + 3] = openFlag(j);
  }
}

interface Samples {
  seq: Float32Array;
  exo: Float32Array;
  target: Float32Array;
  n: number;
}

function buildSamples(s: SeriesData, h: History, targets: number[], scale: number, promoScale: number, W: number): Samples {
  const n = targets.length;
  const seq = new Float32Array(n * W * SEQ_FEATURES);
  const exo = new Float32Array(n * EXO_FEATURES);
  const target = new Float32Array(n);
  const openFlag = (j: number) => (h.open[j] as number);
  targets.forEach((t, r) => {
    seqRows(s, h.y, openFlag, t, scale, W, seq, r * W * SEQ_FEATURES);
    exoRow(s, t, promoScale, exo, r * EXO_FEATURES);
    target[r] = ((h.y[t] as number) - level28(h.y, t)) / scale;
  });
  return { seq, exo, target, n };
}

function toTensors(x: Samples, W: number) {
  return {
    xs: [tf.tensor3d(x.seq, [x.n, W, SEQ_FEATURES]), tf.tensor2d(x.exo, [x.n, EXO_FEATURES])],
    ys: tf.tensor2d(x.target, [x.n, 1]),
  };
}

function subset(x: Samples, idx: number[], W: number): Samples {
  const seqLen = W * SEQ_FEATURES;
  const out: Samples = { seq: new Float32Array(idx.length * seqLen), exo: new Float32Array(idx.length * EXO_FEATURES), target: new Float32Array(idx.length), n: idx.length };
  idx.forEach((src, r) => {
    out.seq.set(x.seq.subarray(src * seqLen, (src + 1) * seqLen), r * seqLen);
    out.exo.set(x.exo.subarray(src * EXO_FEATURES, (src + 1) * EXO_FEATURES), r * EXO_FEATURES);
    out.target[r] = x.target[src] as number;
  });
  return out;
}

function buildNetwork(cfg: LstmConfig): tf.LayersModel {
  const seqIn = tf.input({ shape: [cfg.window, SEQ_FEATURES] });
  const exoIn = tf.input({ shape: [EXO_FEATURES] });
  const encoded = tf.layers
    .lstm({
      units: cfg.units,
      kernelInitializer: tf.initializers.glorotUniform({ seed: cfg.seed }),
      // Glorot (not orthogonal) recurrent init: TF.js warns about slow orthogonal init on 32×128 kernels.
      recurrentInitializer: tf.initializers.glorotUniform({ seed: cfg.seed + 1 }),
    })
    .apply(seqIn) as tf.SymbolicTensor;
  const merged = tf.layers.concatenate().apply([encoded, exoIn]) as tf.SymbolicTensor;
  const hidden = tf.layers
    .dense({ units: cfg.dense, activation: "relu", kernelInitializer: tf.initializers.glorotUniform({ seed: cfg.seed + 2 }) })
    .apply(merged) as tf.SymbolicTensor;
  const out = tf.layers.dense({ units: 1, kernelInitializer: tf.initializers.glorotUniform({ seed: cfg.seed + 3 }) }).apply(hidden) as tf.SymbolicTensor;
  return tf.model({ inputs: [seqIn, exoIn], outputs: out });
}

/** Seeded Fisher–Yates permutation of 0..n−1. */
function permutation(n: number, seed: number): number[] {
  const rng = createRng(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [a[i], a[j]] = [a[j] as number, a[i] as number];
  }
  return a;
}

/** Returns true when training should stop early (e.g. the run was cancelled); checked after every epoch. */
export type StopSignal = () => boolean;

async function fitOn(model: LstmModel, samples: Samples, epochs: number, lr: number, stage: LstmProgress["stage"], withValidation: boolean, onProgress?: (p: LstmProgress) => void, shouldStop?: StopSignal): Promise<number> {
  const cfg = model.config;
  const nVal = withValidation ? Math.max(1, Math.floor(samples.n * cfg.validationFraction)) : 0;
  const trainIdx = permutation(samples.n - nVal, cfg.seed + model.epochsRun);
  const train = toTensors(subset(samples, trainIdx, cfg.window), cfg.window);
  const val = nVal ? toTensors(subset(samples, Array.from({ length: nVal }, (_, i) => samples.n - nVal + i), cfg.window), cfg.window) : null;
  model.net.compile({ optimizer: tf.train.adam(lr), loss: "meanSquaredError" });
  let ran = 0;
  const progress = new tf.CustomCallback({
    onEpochEnd: async (epoch, logs) => {
      ran = epoch + 1;
      onProgress?.({ stage, epoch: epoch + 1, epochs, loss: logs?.loss ?? Number.NaN, valLoss: logs?.val_loss });
      if (shouldStop?.()) model.net.stopTraining = true;
    },
  });
  try {
    await model.net.fit(train.xs, train.ys, {
      epochs,
      batchSize: cfg.batchSize,
      shuffle: false,
      validationData: val ? [val.xs, val.ys] : undefined,
      callbacks: val ? [tf.callbacks.earlyStopping({ monitor: "val_loss", patience: cfg.patience }), progress] : [progress],
      verbose: 0,
    });
  } finally {
    tf.dispose([...train.xs, train.ys, ...(val ? [...val.xs, val.ys] : [])]);
  }
  model.epochsRun += ran;
  return ran;
}

function trainingTargets(s: SeriesData, h: History, from: number, to: number, W: number): number[] {
  const out: number[] = [];
  for (let t = Math.max(from, s.modelStart, W + 28); t <= to; t++) if (h.open[t]) out.push(t);
  return out;
}

function updateSmearing(model: LstmModel, s: SeriesData, h: History, targets: number[]): void {
  const samples = buildSamples(s, h, targets, model.scale, model.promoScale, model.config.window);
  const { xs, ys } = toTensors(samples, model.config.window);
  const pred = model.net.predict(xs) as tf.Tensor;
  const p = pred.dataSync();
  const resid = Array.from(samples.target, (v, i) => (v - (p[i] as number)) * model.scale);
  tf.dispose([...xs, ys, pred]);
  model.smear = smearingFactor(resid);
}

export async function trainLstm(s: SeriesData, h: History, origin: number, cfg: LstmConfig = DEFAULT_LSTM_CONFIG, onProgress?: (p: LstmProgress) => void, shouldStop?: StopSignal): Promise<LstmModel> {
  await initTensorflow();
  const targets = trainingTargets(s, h, s.modelStart, origin, cfg.window);
  if (targets.length < 100) throw new Error("Not enough history to train the LSTM (need at least 100 open days).");
  let m = 0;
  let m2 = 0;
  let promoMax = 0;
  for (const t of targets) {
    const d = (h.y[t] as number) - level28(h.y, t);
    m += d / targets.length;
    m2 += d * d;
    promoMax = Math.max(promoMax, Math.log1p(s.promo[t] ?? 0));
  }
  const scale = Math.max(1e-3, Math.sqrt(m2 / targets.length - m * m));
  const model: LstmModel = { net: buildNetwork(cfg), config: cfg, scale, promoScale: Math.max(1, promoMax), origin, smear: 1, epochsRun: 0, samples: targets.length };
  const samples = buildSamples(s, h, targets, scale, model.promoScale, cfg.window);
  await fitOn(model, samples, cfg.epochs, cfg.learningRate, "train", true, onProgress, shouldStop);
  updateSmearing(model, s, h, targets);
  return model;
}

/** Warm-start update: continue training on the most recent `fineTuneDays` up to the new origin. */
export async function fineTuneLstm(model: LstmModel, s: SeriesData, h: History, newOrigin: number, onProgress?: (p: LstmProgress) => void, shouldStop?: StopSignal): Promise<LstmModel> {
  await initTensorflow();
  const cfg = model.config;
  const targets = trainingTargets(s, h, newOrigin - cfg.fineTuneDays + 1, newOrigin, cfg.window);
  if (targets.length) {
    const samples = buildSamples(s, h, targets, model.scale, model.promoScale, cfg.window);
    await fitOn(model, samples, cfg.fineTuneEpochs, cfg.fineTuneLearningRate, "fine-tune", false, onProgress, shouldStop);
    updateSmearing(model, s, h, targets);
  }
  model.origin = newOrigin;
  return model;
}

/** Recursive forecast in units for days origin+1 … origin+steps (closed days → 0). */
export function forecastLstm(model: LstmModel, s: SeriesData, h: History, steps: number): Float64Array {
  const cfg = model.config;
  const W = cfg.window;
  const yExt = new Float64Array(model.origin + 1 + steps);
  yExt.set(h.y.subarray(0, model.origin + 1));
  const openFlag = (j: number) => (j <= model.origin ? (h.open[j] as number) : isClosed(s, j) ? 0 : 1);
  const seq = new Float32Array(W * SEQ_FEATURES);
  const exo = new Float32Array(EXO_FEATURES);
  const out = new Float64Array(steps);
  for (let k = 1; k <= steps; k++) {
    const t = model.origin + k;
    if (isClosed(s, t)) {
      out[k - 1] = 0;
      yExt[t] = yExt[t - 7] as number;
      continue;
    }
    seqRows(s, yExt, openFlag, t, model.scale, W, seq, 0);
    exoRow(s, t, model.promoScale, exo, 0);
    const dev = tf.tidy(() => {
      const pred = model.net.predict([tf.tensor3d(seq, [1, W, SEQ_FEATURES]), tf.tensor2d(exo, [1, EXO_FEATURES])]) as tf.Tensor;
      return pred.dataSync()[0] as number;
    });
    const logPred = level28(yExt, t) + dev * model.scale;
    yExt[t] = logPred;
    out[k - 1] = toUnits(logPred, model.smear);
  }
  return out;
}

export function disposeLstm(model: LstmModel | null | undefined): void {
  model?.net.dispose();
}
