# Verification log

Evidence that every calculation is correct, recorded as each build part completes.
Reference implementation: `notebooks/01_eda_model_selection_verification.ipynb` (Python, SciPy/pandas).
Production implementation: `app/src` (TypeScript), checked by `npm test` in `app/`.

---

## Part A — Data layer, data pack, formula hand-check

> **Update (Part E):** at the author's request the passphrase and encryption were removed. The site is live only for the faculty review. The data pack is now a plain JSON (`app/public/data/favorita-pack.json`) loaded on open. The decryption rows in A4, A5 and A7 below are kept as history. They are replaced by a test that the deployed JSON is schema-valid and identical to the build output.

### A1. Data audit (Kaggle *Store Sales*, reproduced in notebook §1)

| Check | Result |
|---|---|
| Rows / series | 3,000,888 rows · 54 stores × 33 families = 1,782 daily series |
| Coverage | 2013-01-01 → 2017-08-15: 1,684 dates out of 1,688 calendar days (25 Dec missing every year) |
| Nulls / duplicates / negative sales | 0 / 0 / 0 |
| Promotions recorded from | 2014-04-01 (earlier zeros are missing data) → model window starts 2014-04-01 |
| Oil gaps | 43 null prices + 486 non-trading days → interpolated, lagged 7 days |
| Store 44 calendar (2013-01-01 → 2017-08-31) | 90 holiday days, 25 event days, 31 earthquake days, 5 work days, 9 closed days (every 1 Jan and 25 Dec) |

### A2. Slide formula — hand calculation (Store 44 · BEVERAGES)

Inputs: trailing 365 calendar days to 2017-08-15, store-open days only (n = 363).
D̄ = 10,865.3554 units/day · σ_D = 3,587.1772 units/day (sample SD, ddof = 1).

**SS = Z · √(σ_D²·L + D̄²·σ_L²) · ROP = D̄·L + SS**

| L (days) | σ_L (days) | Service level | Z | σ_D²·L | D̄²·σ_L² | Lead-time share | SS (units) | ROP (units) | Days of cover |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 2.0 | 90% | 1.2816 | 90,074,884 | 472,223,789 | 84% | 30,389.2 | 106,446.7 | 2.80 |
| 7 | 2.0 | **95%** | **1.6449** | 90,074,884 | 472,223,789 | 84% | **39,004.1** | **115,061.6** | 3.59 |
| 7 | 2.0 | 99% | 2.3263 | 90,074,884 | 472,223,789 | 84% | 55,164.3 | 131,221.8 | 5.08 |
| 5 | 1.5 | 90% | 1.2816 | 64,339,203 | 265,625,882 | 81% | 23,279.3 | 77,606.1 | 2.14 |
| 5 | 1.5 | 95% | 1.6449 | 64,339,203 | 265,625,882 | 81% | 29,878.7 | 84,205.4 | 2.75 |
| 5 | 1.5 | 99% | 2.3263 | 64,339,203 | 265,625,882 | 81% | 42,258.0 | 96,584.7 | 3.89 |

Step by step (default L = 7, σ_L = 2, 95%):

1. Z = Φ⁻¹(0.95) = 1.644854
2. σ_D²·L = 3,587.18² × 7 = 90,074,883.8
3. D̄²·σ_L² = 10,865.36² × 2² = 472,223,789.4
4. SS = 1.6449 × √562,298,673.2 = **39,004.15 units**
5. ROP = 10,865.36 × 7 + 39,004.15 = **115,061.63 units**

These values are stored in `app/tests/golden/favorita-golden.json`. Part C's TypeScript safety-stock module must reproduce them.

### A3. Monte Carlo cross-check (400,000 draws, seed 20261111)

| Lead-time model | SS from Monte Carlo (95%) | SS from formula | Difference | P(stockout) at formula ROP |
|---|---|---|---|---|
| Normal (the formula's own assumption) | 39,810 | 39,004 | +2.1% | 5.34% |
| Gamma (right-skewed delays) | 42,242 | 39,004 | +8.3% | 6.19% |

Under its own assumptions the formula agrees with simulation to within about 2%. The residual comes from the compound lead-time-demand distribution not being exactly normal. With realistic skewed delays, the formula under-protects: the stockout probability is 6.2%, not the intended 5%.

### A4. Python ↔ TypeScript parity (the `realPack.test.ts` suite, run locally on the git-ignored data)

| Quantity | Python (notebook) | TypeScript (`openWindowStats`, `seriesFromPack`) |
|---|---|---|
| n open days | 363 | 363 ✓ |
| D̄ | 10,865.355371900827 | equal to 1e-8 ✓ |
| σ_D | 3,587.177238318146 | equal to 1e-6 ✓ |
| Holiday / event / earthquake / work / closed days | 90 / 25 / 31 / 5 / 9 | identical ✓ |
| Promotion days since 2014-04-01 | 1,068 | 1,068 ✓ |
| Encrypted pack → decrypt (WebCrypto) | — | byte-for-byte equal to the plaintext pack ✓ |

### A5. Unit tests: `npm test` → 7 files, **32 tests passed**

- **dates:** epoch-day round trips, leap years, pandas weekday, day-of-year and month-end conventions.
- **crypto:**
  - Decrypts the Python-encrypted fixture.
  - Rejects a wrong passphrase, a tampered ciphertext and non-pack files.
- **series:**
  - Pack → typed arrays.
  - Replay window release.
  - Adding new data: contiguous append, extension past the calendar, corrections, re-opening closed days.
  - Validation errors: gaps, duplicates, negative values, fractional units, bad promotion counts, invalid dates.
- **features:** payday, Christmas ramp, weekday dummies, Fourier terms, trend, log-promotion, and a 7-day oil lag with no same-day information.
- **newData:** CSV parsing in the simple format, header synonyms, semicolons, BOM, Kaggle format filtered by store and family, line-level errors.
- **demo:**
  - Schema-valid, labelled synthetic, deterministic for a given seed.
  - Calendar offsets mirror the real pack; the weekly pattern matches the fitted coefficients.
  - The RNG has standard-normal moments.
- **realPack:** the parity table above.

### A6. Demo generator validation (notebook §6)

| Statistic | Real (Store 44 BEVERAGES) | Synthetic demo |
|---|---|---|
| Mean units/day | 9,109 | 9,084 |
| Coefficient of variation | 0.406 | 0.410 |
| Weekday max/min | 2.04 | 2.02 |
| Lag-7 autocorrelation (log) | 0.835 | 0.643 |

The demo reproduces the level, volatility and weekly pattern. Its weaker lag-7 autocorrelation reflects real level shifts that a stationary AR(1) process does not model. That is acceptable for a clearly labelled demo.

### A7. Data protection checks

- `.gitignore` verified with `git check-ignore`:
  - **Ignored:** the raw Kaggle folder, `data/processed/` (plaintext pack) and `.env.local` (dev passphrase).
  - **Tracked:** app source, the encrypted pack, the synthetic fixture and the notebook.
- The production build (`npm run build`) contains no plaintext pack content: no `favorita-pack/1`, holiday names or family names in the JavaScript bundle. Only `dist/data/favorita-pack.enc` (ciphertext) ships.

---

## Part B — Forecasting models, backtesting, continuous updating

### B1. Python ↔ TypeScript parity (`parity.test.ts`; reference values from `scripts/make_parity_golden.py`, synthetic inputs)

| Component | Reference | Result |
|---|---|---|
| Holt-Winters filter: 140 one-step fitted values, SSE, final level and trend | statsmodels `ExponentialSmoothing` (additive damped trend, additive weekly season, fixed parameters, known initial state) | equal to 1e-10 ✓ |
| Holt-Winters 14-day forecast | statsmodels | equal to 1e-10 on 12 of 14 steps ✓ (see note) |
| Ridge coefficients, intercept, prediction | scikit-learn `Ridge(alpha=3)` on `StandardScaler` inputs | equal to 1e-9 ✓ |
| MAE, RMSE, WAPE, bias | scikit-learn / NumPy | equal to 1e-10 ✓ |
| Quantiles (2.5%, 10%, 50%, 90%, 97.5%) | `numpy.quantile` (linear) | equal to 1e-10 ✓ |

**Note on Holt-Winters steps 7 and 14.** The textbook forecast (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, §8.3) is ŷ_{T+h} = ℓ_T + φ_h b_T + s_{T+h−m(k+1)} with k = ⌊(h−1)/m⌋, so for h = m it uses the **latest** seasonal estimate s_T.

statsmodels uses the previous week's estimate s_{T−7} at exact multiples of the period. I confirmed this numerically: statsmodels' value at h = 7 equals the textbook value with s_{T−7} substituted (9.668150 against 9.655417).

The dashboard follows the textbook formula. The test asserts that at h = 7 and 14 the two differ by exactly s_T − s_{T−7}, and agree to 1e-10 everywhere else.

### B2. Model behaviour tests (`models.test.ts`, `engine.test.ts`, `mathlib.test.ts`)

- **No look-ahead:** inflating every sales value after a fold origin tenfold leaves the ETS and Ridge forecasts from that origin bit-for-bit unchanged.
- **Fold placement:** 6 × 28-day origins sit at as-of − 168, −140, …, −28 days.
- **Benchmark beaten:** ETS and Ridge each beat the 28-day moving average out of sample on the demo series. Ridge recovers the demo generator's weekend uplift.
- **Ensemble:**
  - Weights are ∝ 1/WAPE and sum to 1.
  - Fold k uses weights learned from folds before k only.
  - A perfect model gets about 100% of the weight. This test caught and fixed a bug: a WAPE of 0 had been excluding the model.
- **Prediction intervals:** split-conformal 80% bands cover 75–85% of the calibration errors, and the 95% band is never inside the 80% band.
- **Live tracking:** issued forecasts are scored only on days observed after they were issued.
- **Engine, full path (LSTM on the TF.js WebAssembly backend):**
  1. Trained through the 6 rolling folds.
  2. A new week arrives → version 2, LSTM **fine-tuned** (not retrained), backtest reused from version 1, forecasts change.
  3. `retrainLstm` → full retrain, new backtest version.
- **Math library:** Cholesky solve on a random symmetric positive-definite system (1e-9), jitter for a singular matrix, Nelder–Mead finds the Rosenbrock minimum (1, 1).

**Test suite:** `npm test` → 11 files, **55 tests passed**. `tsc` is clean.

### B3. Performance (in-browser training budget)

| Item | Measured |
|---|---|
| ETS + Ridge, 7 fits each (6 folds + final) | ≈ 1.5 s in total |
| LSTM epoch, ~1,000 samples, TF.js **CPU** backend (Node) | 4.5 s/epoch → rejected |
| LSTM epoch, TF.js **WebAssembly** backend (Node, SIMD) | **1.3 s/epoch** → adopted (threads add more speed in a cross-origin-isolated browser) |
| LSTM fine-tune on new data (365 days, 4 epochs) | ≈ 2–4 s |

All training runs in the forecast Web Worker, so the UI thread is never blocked. The production bundle check confirms the worker and all three WebAssembly variants (plain, SIMD, threaded-SIMD) are packaged.

### B4. Real-data model comparison (notebook §8; identical folds: 6 × 28 days before 2017-06-20)

**Ridge penalty.** λ was selected by rolling-origin validation across the 9 shortlist series. Mean WAPE by λ:

| λ | 0.1 | 1 | 3 | 10 | **30** | 100 | 300 |
|---|---|---|---|---|---|---|---|
| Mean WAPE | 14.11% | 14.10% | 14.06% | 13.99% | **13.93%** | 14.15% | 14.94% |

**All 9 series.** The browser models run in production TypeScript via `app/scripts/backtestReport.ts`; LightGBM runs in Python.

| Model | Mean WAPE | Mean MASE | Mean bias |
|---|---|---|---|
| Ridge regression (browser) | **13.9%** | 0.89 | −6.6% |
| **Adaptive ensemble (browser), the dashboard default** | 14.1% | **0.88** | **−1.4%** |
| LSTM, TF.js (browser) | 14.8% | 0.93 | −1.0% |
| LightGBM (Python reference) | 15.2% | 0.97 | −4.5% |
| Holt-Winters ETS (browser) | 20.4% | 1.24 | +5.2% |
| Seasonal naive (benchmark) | 22.6% | 1.42 | 0.0% |
| 28-day moving average (classic calculator) | 28.0% | 1.78 | +1.6% |

**3 representative series** (44-BEVERAGES, 44-CLEANING, 47-BEVERAGES), all models:
- Ensemble **12.6%**
- LSTM (TF.js) 13.3%
- SARIMAX 13.4%
- Ridge 13.6%
- LightGBM 15.0%
- ETS 16.7%
- LSTM (Keras) 17.3%
- Seasonal naive 21.5%
- Moving average 26.5%

**Store 44 · BEVERAGES (default series):**
- Ensemble 12.0%, against 28.2% for the moving average and 21.3% for seasonal naive.
- Weights: ETS 0.29, Ridge 0.36, LSTM 0.35.

**Conclusions (reported as found):**
1. Every AI model beats both benchmarks. The ensemble mis-forecasts **50% fewer units** than the classic moving-average calculator (14.1% vs 28.0% WAPE).
2. Ridge has the lowest average WAPE by 0.2 pp, but **under-forecasts by 6.6%** on average. Under-forecasting directly causes stockouts. The ensemble is nearly unbiased (−1.4%) and has the best MASE, so it stays the default forecast. Every model's accuracy is shown side by side on the dashboard.
3. Holt-Winters is the weakest model. It cannot use holidays or other drivers, and it struggles most on Personal Care. The inverse-WAPE weights automatically reduce its influence.
4. The in-browser LSTM (13.3%) beat the Keras reference (17.3%) on the same architecture. The main difference is the training scheme: the browser LSTM is trained once and then warm-start fine-tuned at each fold, while the reference retrains from scratch per fold. The browser models give up no accuracy against heavier Python tools; only SARIMAX comes close.
5. Driver effects measured by Ridge for 44-BEVERAGES, conditional on recent sales level:
   - Saturday ×1.44, Sunday ×1.53 vs Monday
   - holidays +21%
   - promotions +0.8% per doubling
   - paydays +0.5%
   - oil +0.2% per $10

   These agree with the EDA: weekday and holiday effects dominate, and promotions are weak at product-family level.

**Engine cost per series on the real data:** LSTM rolling training takes 43–87 s (46–61 epochs, WebAssembly backend in Node). Re-running Holt-Winters, Ridge and the ensemble, or fine-tuning the LSTM after new data, takes seconds.

---

## Part C — Safety stock (baseline vs AI) and Monte Carlo

### C1. Formula, standard normal and lead-time models (`normal.test.ts`, `safetyStock.test.ts`, `monteCarlo.test.ts`)

| Check | Reference | Result |
|---|---|---|
| Φ, upper tail, density, loss function at x ∈ [−8, 12] | `scipy.stats.norm` | relative error < 1e-12 ✓ |
| Φ⁻¹ (service-level factor Z), p from 1e-12 to 1 − 1e-6 | `scipy.stats.norm.ppf` | \|error\| < 1e-12 ✓ |
| SS = Z·√(σ_D²L + D̄²σ_L²), all 6 hand-calculated cases (A2) | notebook §5 | equal to 1e-6 ✓; worked example SS 39,004.15, ROP 115,061.63 ✓ |
| Special cases σ_L = 0 → Z·σ_D·√L; σ_D = 0 → Z·D̄·σ_L; 50% service → SS = 0 | algebra | ✓ |
| Gamma sampler (Marsaglia–Tsang), k = 12.25, θ = 4/7: mean, SD, 95th percentile | `scipy.stats.gamma` | within 0.3–0.5% at N = 400k ✓ |
| Monte Carlo under the formula's own assumptions (normal lead time, i.i.d. normal demand) | notebook +2.1% | TypeScript **+2.06%**, P(stockout) 5.33% ✓ |
| Monte Carlo with Gamma lead time | notebook +8.3%, 6.19% | TypeScript **+8.66%**, P(stockout) **6.25%** ✓ |
| Zero forecast errors and fixed L → DDLT = forecast lead-time demand exactly; weekday alignment of the error blocks | construction | ✓ |
| Riskier scenarios (delay, erratic supplier, surge, perfect storm) raise stockout risk at a fixed ROP | monotonicity | ✓ |

**Bug found and fixed during verification.** The first error bootstrap took error runs with any start offset. That applied errors from quiet weekdays to busy weekdays, which inflated the tail for strongly weekly series. It now resamples **weekday-aligned weekly blocks**: fold origins are whole weeks before the live origin, so fold day p and live day k share a weekday when p ≡ k (mod 7). A dedicated test pins this behaviour, and the NumPy twin in the notebook implements the same rule.

### C2. Real data — orders placed on 2017-06-20, 95%, L = 7 d, σ_L = 2 d, Gamma (`safetyReport.ts` → notebook §9)

- The formula is recomputed in Python from the same inputs: **max relative difference 0.0**.
- Independent NumPy Monte Carlo vs TypeScript (44-BEVERAGES, 400k vs 200k trials): buffer needed 41,476 vs 41,169 (**+0.75%**, within Monte Carlo error).

| Finding (mean over 9 series) | Value |
|---|---|
| σ_e (out-of-sample forecast RMSE, horizons 1–7) ÷ raw σ_D | **70%**: the forecast explains about a third of daily variability |
| AI safety stock vs baseline safety stock | −0.8% (lead-time variance dominates the buffer at σ_L = 2 d) |
| AI reorder point vs baseline | moves with the forecast, e.g. 47-BEVERAGES 115,386 vs 96,837 (+19%): sales are growing, and a 365-day average lags |
| Stockout risk per cycle at the **baseline** ROP (Monte Carlo) | **9.3%** on average (3.9–21.3%) |
| Stockout risk per cycle at the **AI formula** ROP | **7.5%** on average |
| Monte Carlo ROP needed for 95% vs AI formula ROP | **+5.7%** on average: the formula's normal approximation under-protects (skewed lead times; forecast bias, notably 1-CLEANING +19%) |

**Implication for the dashboard.** It shows a reorder-point ladder: baseline → AI formula (the slide's equation) → Monte Carlo (risk-calibrated) → Bayesian-optimised (Part D). Each step's added protection is visible.

---

## Part D — Bayesian optimisation of the reorder policy under constraints

### D1. Algorithms (`optimize.test.ts`; reference values from scikit-learn)

| Check | Reference | Result |
|---|---|---|
| GP posterior mean, posterior SD, log marginal likelihood (Matérn 5/2 ARD + white noise) | scikit-learn `GaussianProcessRegressor` (alpha = 0) | equal to 1e-9 ✓ |
| Branin benchmark, global minimum 0.3979 | analytic | found below 0.448 in **38** evaluations ✓ |
| Constrained toy problem: min (x−0.8)² + (y−0.8)² subject to x + y ≤ 1 (optimum 0.18) | analytic | feasible, below 0.20 ✓ |
| Stock conservation every day: start + receipts − sales = end, and sales + lost = demand | invariant | ✓ |
| Common random numbers: a larger buffer never loses more sales | monotonicity | ✓ |
| BO vs 900-point grid on a synthetic inventory problem | exhaustive grid | within 2% using 38 evaluations ✓ |
| Tornado and heatmap behaviour (ROP rises with service level and σ_L) | formula | ✓ |

The GP parity test exposed a subtle difference: scikit-learn adds `alpha = 1e-10` to the kernel diagonal by default. The reference now uses `alpha = 0` so both implementations solve the same system.

### D2. Real data (`optimizeReport.ts` → notebook §10)

**Setup:**
- 64 paths × 13 weeks (after a 2-week warm-up).
- Gamma lead times, common random numbers.
- Costs: unit $0.70, holding 25%/yr, $120 per order, $0.50 per lost unit.
- Constraints: fill rate ≥ 98%, peak stock ≤ 18 days of demand, average inventory value ≤ 12 days of demand at cost, ≤ 3 deliveries per week.
- Forecasts in this offline sweep: ETS + Ridge ensemble (the LSTM is omitted for speed).

**Independent NumPy policy simulator** on the identical paths, for the baseline, AI-formula and BO policies: every metric (cost/day, fill rate, cycle service level, orders/week, average and peak stock) matches with **max relative difference 0.0**.

| Series group | Baseline (static formula ROP + EOQ) | AI formula (dynamic, EOQ) | **Bayesian-optimised** | 900-point grid |
|---|---|---|---|---|
| Beverages (4 series) | $231/day avg, fill 96.9%, **meets 98% in 0 of 4** | $134/day, fill 98.4–98.9%, feasible | **$70/day, fill ≈ 99.99%, all constraints met** | $71/day |
| Grocery I | $87/day, feasible | $97/day | **$66/day** | $67/day |
| Cleaning / Personal Care (4 series) | infeasible: EOQ = 14–28 days of demand, so peak stock is 22–37 days against an 18-day capacity | infeasible, same cause | cheapest **feasible** policy: Q ≈ 9 days, peak = capacity (binding) | within 0.3–3.5% |

- **BO vs grid:** median gap **−1.71%** (range −3.46% to +0.59%). BO matches or beats the grid using **38 instead of 900** simulations (4%), because it can choose points between grid nodes.
- **BO run time:** 5–8 s per series in Node, inside the optimiser Web Worker in the browser.
- **Honest reading:**
  - Under these cost assumptions lost sales are expensive relative to holding ($0.50 vs $0.0005 per unit per day). The optimum therefore keeps about 5–6 days of buffer, with a fill rate near 100%.
  - With such large buffers, dynamic and static reorder points cost about the same ($72 vs $73 for 44-BEVERAGES). The optimiser's gain comes mainly from choosing the buffer and order size jointly under the constraints.
  - The dashboard lets these cost assumptions be changed, and the optimum moves with them.

---

## Part E — Dashboard UI (browser checks)

| Check | Result |
|---|---|
| Real data loads automatically (no passphrase); data-source switch Real ⇄ Synthetic demo | ✓ |
| Full browser pipeline on real data (44-BEVERAGES), production build | Fast ETS + Ridge result in ≈ 2–3 s. Full LSTM rolling training ≈ 50 s (59 epochs, WebAssembly backend). Ensemble WAPE **12.0%**, identical to the offline Node run |
| New data: "Next week" on the live replay | v3 in **4.6 s**: LSTM fine-tuned, ETS and Ridge refitted, version log and live accuracy updated |
| Bayesian optimisation in the browser (44-BEVERAGES) | $73/day vs baseline $194/day, fill rate 99.96%, all constraints met. Matches the offline report |
| Switching series or data source while the LSTM trains | stale run cancelled within one epoch; new results in ≈ 10 s (`engine.test.ts` cancellation test) |
| Console | 0 errors, 0 warnings in the production build (all pages, both data sources, switching, replay) |
| Phone layout (375 × 812), all pages | no horizontal overflow; navigation in a drawer; controls stack |
| Production bundle | main JS 352 KB gzip; forecast worker (TensorFlow.js) loaded only inside the worker; three WebAssembly variants emitted |

**Defects found and fixed during browser testing:**
1. **TF.js WebAssembly package excluded from Vite pre-bundling.** Its ES-module entry imports a CommonJS helper, so the worker failed to load. It is now pre-bundled.
2. **`simClient` mutated a Comlink proxy.** That triggered a DataCloneError; the client now returns the proxy unchanged.
3. **Monte Carlo KPI tile showed "—".** The animated counter tweened from NaN; it now jumps to the first finite value.
4. **GP surrogate heatmap.** Each heatmap layer needs its own colour mapping; the infeasible-region layer now has one.
5. **LSTM training stalled whenever the page was not painting.** TF.js yields with `requestAnimationFrame` inside workers. `workerTiming.ts` now forces a `setTimeout` yield, which always runs and lets cancel messages through.
6. **Stale forecast runs blocked new ones after a series or source switch.** A cancellation generation now drops queued runs and stops LSTM training between epochs.
7. **ECharts warned about zero-size containers.** Charts now initialise only once their container has a size.
8. **Cosmetic issues:**
   - the uppercase style turned "σ_D" into "Σ_D";
   - one axis mixed "5,000" and "10K";
   - the history range was missing its year;
   - dev hot reloads re-created the React root.

---

## Part F — Integration: Baseline vs AI replay, promotion planner, code-splitting, deployment files

### F1. Replay protocol (`app/src/inventory/replay.ts`, page *Baseline vs AI*)

- **Window:** the last 56 days of the pack (2017-06-21 → 2017-08-15). The models never saw these days at the replay start.
- **Delays:** 500 Gamma delivery-delay paths (L = 7 d, σ_L = 2 d), the same for every policy (common random numbers). Unserved demand is lost.
- **Costs and constraints:** the Part D defaults.
- **Warm-up:** 28 days of real sales before the window are simulated but not scored (see F3).
- **Policies:**
  - **Baseline:** static slide-formula ROP from the last 365 days, EOQ order size.
  - **AI formula:** the slide formula fed by the forecast, re-planned weekly. Eight forecasts, each issued with data only up to its own origin.
  - **AI optimised:** forecast lead-time demand plus the Bayesian-optimised buffer and order quantity, chosen once at the replay start.

**Tests (`replay.test.ts`, 3 tests):**
- every path shares the real demand and only the delays differ;
- the week-2 AI level equals `aiPolicy(forecast issued at week-2 origin, offset 1)`, so there is no look-ahead;
- the warm-up keeps the starting stock out of the scored days, with a hand-worked case: R = 1,000, Q = 700, 100 units/day, L = 5. Peak without warm-up = R + Q = 1,700; with warm-up = R − 5·100 + Q + 100 = 1,300 ✓;
- trace length 56, warm-up 28, paired shares in [0, 1].

### F2. Results on real held-back demand (all 9 series; `replay_all` offline run = browser run)

Fill rate · cost per day · peak stock in days of actual demand. "✗" marks a broken constraint (storage 18 d, budget 12 d of demand at cost, ≥ 98% fill, ≤ 3 deliveries a week).

| Series | Baseline | AI formula (weekly) | AI optimised | Optimised cheaper (paired) |
|---|---|---|---|---|
| 44-BEVERAGES | 99.65% · $73 · 15.9 d | 99.66% · $73 · 16.9 d | **99.99% · $70 · 18.7 d** | 15% of paths |
| 45-BEVERAGES | 99.58% · $70 · 15.8 d | 99.45% · $81 · 16.2 d | **100.00% · $67 · 18.0 d** | 14% |
| 47-BEVERAGES | 99.33% · $77 · 16.2 d | 99.46% · $75 · 17.8 d | **100.00% · $72 · 21.2 d** | 21% |
| 3-BEVERAGES | 99.53% · $69 · 16.5 d | 99.45% · $72 · 16.8 d | **100.00% · $69 · 19.8 d** | 18% |
| 44-GROCERY I | 99.68% · $71 · 16.2 d | 99.55% · $77 · 15.5 d | **99.96% · $67 · 17.0 d** | 10% |
| 44-CLEANING | 99.36% · $30 · ✗ storage | 99.18% · $32 · ✗ storage | 99.45% · $35 · ✗ storage | 11% |
| 44-PERSONAL CARE | 99.94% · $10 · ✗ storage + budget | 99.73% · $10 · ✗ storage + budget | 99.50% · $19 · ✗ storage (18.2 d) | 0% |
| 45-PERSONAL CARE | 99.85% · $11 · ✗ storage + budget | 99.94% · $11 · ✗ storage + budget | **99.73% · $21 · all met** | 1% |
| 1-CLEANING | 99.74% · $10 · ✗ storage + budget | 99.81% · $10 · ✗ storage + budget | **99.85% · $18 · all met** | 2% |

The storage check uses capacity = 18 × forecast mean, as on the Optimisation page. A peak of 21.2 days of *actual* demand can therefore still fit when the forecast mean is higher.

**Honest reading:**
- **Fast movers (beverages, grocery; 5 series):** the optimised policy has the highest fill rate (99.96–100%) and the lowest or equal cost ($67–72 vs $69–77 per day). It meets every constraint.
  - The paired figure shows *how* it wins: it is cheaper on only 10–21% of delay paths but much cheaper on the bad ones. It carries slightly more stock every day (higher holding cost) and avoids the costly stockouts when deliveries run late. It works like insurance.
- **AI formula alone:** at the same 95% target it matches the baseline's fill rate (within ±0.2 pp) and is not cheaper. Better forecasts shrink σ, but with EOQ ordering and a 99%+ fill rate there is little left to gain. The gain comes from **optimising buffer and order size jointly**, which the AI pipeline makes possible.
- **Slow movers (cleaning, personal care; 4 series):** the classic EOQ orders 14–28 days of demand at once, so the baseline and the AI formula **break storage and budget**. Their low $10/day is not achievable in the store. The optimiser finds policies that fit:
  - all constraints met on 2 of 4 series;
  - 44-PERSONAL CARE just over capacity (18.2 d);
  - 44-CLEANING over by forecast-vs-actual mismatch.

  This costs $18–21 per day: the real price of respecting the constraint.
- **Scope:** these are one 8-week window of real data, not guarantees. The page states this.

**Browser check (dev server, 44-BEVERAGES):**
- the store's `runImpact()` completed: eight weekly re-forecasts, BO at the start, then the replay;
- results equal to the offline run to the cent: 99.65 / 99.66 / 99.99% and $73 / $73 / $70;
- trace length 56, warm-up 28, **0 console errors**;
- 18 s alone, about 50 s while the LSTM trains in parallel. Each runs in its own worker, and the UI stayed responsive.

### F3. Defect found and fixed: replay starting stock counted as the "peak"

- **Symptom:** in the first replay version the optimised policy broke storage on 9 of 9 series, although it was feasible when optimised.
- **Cause:** every simulation starts at R + Q ("just after a delivery"). That is a full lead time of demand above where a running policy sits after a delivery. The Part D scenarios have a 14-day warm-up, but the replay scored from day 0, so the artificial opening stock became the peak and inflated holding.
- **Fix:** the replay now runs each policy through the 28 days of real sales before the window, unscored. The hand-worked unit test above pins the fix.
- **Effect:**
  - 44-BEVERAGES baseline $81 → $73/day;
  - optimised peak 21.4 → 18.7 days (feasible);
  - the conclusions in F2 replace the earlier ones.

### F4. Promotion planner (*Demand Forecast* page)

- **Control:** a slider scales the future promotion intensity (0–3×) and re-runs the forecast without retraining the LSTM.
- **Browser check (44-BEVERAGES):**
  - a "Promotion plan 2.00× → forecasts recomputed" entry was added to the version log in **0.6 s**;
  - the 28-day ensemble total rose by **+0.82%** (329,089 → 331,781 units).
- **Plausibility:** the Ridge model's measured effect is +1–6% per doubling. The ensemble also blends ETS, which has no promotion input, so the blended effect is smaller.
- The reset back to 1.00× restores the forecast. 0 console errors.

### F5. Build and deployment files

- **Code-splitting:**
  - every page is lazy-loaded;
  - long-lived vendor chunks: react, motion, echarts.
- **Production bundle (gzip):**
  - first load ≈ 124 KB: index 38.6 + react 43.2 + motion 42.6, plus 19.7 KB CSS;
  - ECharts 208.6 KB, loaded with the first chart;
  - pages 2.4–5.1 KB each;
  - TensorFlow.js only inside the forecast worker (1.87 MB raw), with three WebAssembly variants.
- **`netlify.toml`:**
  - base `app`, `npm run build`, publish `dist`, Node 22;
  - COOP/COEP/CORP for cross-origin isolation (multi-threaded WASM);
  - security headers;
  - immutable cache for hashed assets, 5 min for `/data/*`;
  - SPA fallback.
- **README:** overview, slide mapping, architecture, run, deploy and take-down steps.
- **Project name:** **ReorderIQ** in the page title and meta description, the header, the start screen, `package.json`, the README and `netlify.toml`.
- **Checks:** `npm test` passes 15 files and **92 tests**; `npm run build` (type-check + Vite) passes with 0 errors.
