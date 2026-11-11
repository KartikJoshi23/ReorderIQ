/**
 * Regression model: Ridge (L2-penalised) dynamic regression on log(1 + sales).
 *
 * Inputs = exogenous drivers known in advance (calendar, holidays, promotions, oil — see
 * data/features.ts) + autoregressive terms (lag 7, lag 14, 7- and 28-day means of past log sales).
 * Columns are standardised (population SD, as sklearn StandardScaler); the intercept is not
 * penalised. Coefficients solve (XᵀX + λI) β = Xᵀ(y − ȳ), which equals
 * sklearn.linear_model.Ridge(alpha=λ) fitted on the standardised matrix.
 * Multi-day forecasts are recursive: each predicted day feeds the next day's lag inputs.
 */
import { DEFAULT_FEATURES, FEATURES, type FeatureFn } from "../data/features";
import type { SeriesData } from "../data/series";
import { normalEquations, solveSpd } from "../lib/linalg";
import { AR_FEATURES, arFeatures, type History, isClosed, smearingFactor, toUnits } from "./history";

/** Penalty selected by rolling-origin validation across the 9-series shortlist
 *  (grid 0.1…300, lowest mean WAPE at λ = 30; see notebook §8 and app/scripts/backtestReport.ts). */
export const DEFAULT_RIDGE_LAMBDA = 30;
/** First usable target needs 28 days of lag history. */
const MIN_LAG = 28;

export interface RidgeFit {
  beta: Float64Array;
  intercept: number;
  mean: Float64Array;
  sd: Float64Array;
}

export interface RidgeModel extends RidgeFit {
  names: string[];
  exogNames: readonly string[];
  lambda: number;
  origin: number;
  smear: number;
  residualSd: number;
  n: number;
}

/** Standardise columns (population SD; constant columns keep SD 1) and solve the ridge system. */
export function ridgeSolve(X: Float64Array, y: Float64Array, rows: number, cols: number, lambda: number): RidgeFit {
  const mean = new Float64Array(cols);
  const sd = new Float64Array(cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] = (mean[c] as number) + (X[r * cols + c] as number) / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const d = (X[r * cols + c] as number) - (mean[c] as number);
      sd[c] = (sd[c] as number) + (d * d) / rows;
    }
  }
  for (let c = 0; c < cols; c++) sd[c] = (sd[c] as number) > 1e-12 ? Math.sqrt(sd[c] as number) : 1;
  let yMean = 0;
  for (let r = 0; r < rows; r++) yMean += (y[r] as number) / rows;

  const Z = new Float64Array(rows * cols);
  const yc = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    yc[r] = (y[r] as number) - yMean;
    for (let c = 0; c < cols; c++) Z[r * cols + c] = ((X[r * cols + c] as number) - (mean[c] as number)) / (sd[c] as number);
  }
  const { xtx, xty } = normalEquations(Z, yc, rows, cols);
  for (let c = 0; c < cols; c++) xtx[c * cols + c] = (xtx[c * cols + c] as number) + lambda;
  const beta = solveSpd(xtx, xty, cols);
  return { beta, intercept: yMean, mean, sd };
}

export function ridgePredictRow(fit: RidgeFit, x: ArrayLike<number>): number {
  let v = fit.intercept;
  for (let c = 0; c < fit.beta.length; c++) v += (fit.beta[c] as number) * (((x[c] as number) - (fit.mean[c] as number)) / (fit.sd[c] as number));
  return v;
}

function rowBuilder(s: SeriesData, exogNames: readonly string[]) {
  const fns = exogNames.map((n) => FEATURES[n] as FeatureFn);
  const cols = fns.length + AR_FEATURES.length;
  return {
    cols,
    fill(y: ArrayLike<number>, t: number, out: Float64Array, offset = 0) {
      for (let c = 0; c < fns.length; c++) out[offset + c] = (fns[c] as FeatureFn)(s, t);
      arFeatures(y, t, out, offset + fns.length);
    },
  };
}

export function fitRidge(s: SeriesData, h: History, origin: number, lambda = DEFAULT_RIDGE_LAMBDA, exogNames: readonly string[] = DEFAULT_FEATURES): RidgeModel {
  const builder = rowBuilder(s, exogNames);
  const targets: number[] = [];
  for (let t = Math.max(s.modelStart, MIN_LAG); t <= origin; t++) if (h.open[t]) targets.push(t);
  const rows = targets.length;
  const X = new Float64Array(rows * builder.cols);
  const y = new Float64Array(rows);
  targets.forEach((t, r) => {
    builder.fill(h.y, t, X, r * builder.cols);
    y[r] = h.y[t] as number;
  });
  const fit = ridgeSolve(X, y, rows, builder.cols, lambda);

  const residuals = new Float64Array(rows);
  let ss = 0;
  for (let r = 0; r < rows; r++) {
    residuals[r] = (y[r] as number) - ridgePredictRow(fit, X.subarray(r * builder.cols, (r + 1) * builder.cols));
    ss += (residuals[r] as number) ** 2;
  }
  return {
    ...fit,
    names: [...exogNames, ...AR_FEATURES],
    exogNames,
    lambda,
    origin,
    smear: smearingFactor(residuals),
    residualSd: Math.sqrt(ss / Math.max(1, rows - builder.cols - 1)),
    n: rows,
  };
}

/** Recursive forecast in units for days origin+1 … origin+steps (closed days → 0). */
export function forecastRidge(model: RidgeModel, s: SeriesData, h: History, steps: number): Float64Array {
  const builder = rowBuilder(s, model.exogNames);
  const yExt = new Float64Array(model.origin + 1 + steps);
  yExt.set(h.y.subarray(0, model.origin + 1));
  const row = new Float64Array(builder.cols);
  const out = new Float64Array(steps);
  for (let k = 1; k <= steps; k++) {
    const t = model.origin + k;
    builder.fill(yExt, t, row);
    const logPred = ridgePredictRow(model, row);
    if (isClosed(s, t)) {
      out[k - 1] = 0;
      yExt[t] = yExt[t - 7] as number; // same imputation as History for closed days
    } else {
      out[k - 1] = toUnits(logPred, model.smear);
      yExt[t] = logPred;
    }
  }
  return out;
}

/** Original-scale coefficient of one input (effect on log sales per unit of that input). */
export function rawCoefficient(model: RidgeModel, name: string): number {
  const c = model.names.indexOf(name);
  return c < 0 ? 0 : (model.beta[c] as number) / (model.sd[c] as number);
}
