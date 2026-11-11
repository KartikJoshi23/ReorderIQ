# Phase 2 — Dataset Inspection & Implementation Plan

**Exercise:** AI Applications in Reorder Point & Inventory Management (Session II)
**Industry:** FMCG / grocery retail (non-perishable)
**Dataset:** Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita, Ecuador)
**Status:** Plan for approval. No application code has been written.

---

## 1. Dataset inspection

### 1.1 Files (in `store-sales-time-series-forecasting/`)

| File | Rows | Columns | Coverage | Role in this project |
|---|---|---|---|---|
| `train.csv` | 3,000,888 | id, date, store_nbr, family, sales, onpromotion | 2013-01-01 → 2017-08-15 (1,684 dates) | Demand history and promotions |
| `test.csv` | 28,512 | id, date, store_nbr, family, onpromotion | 2017-08-16 → 2017-08-31 | **Promotions planned for the next 16 days**, used as known future inputs |
| `oil.csv` | 1,218 | date, dcoilwtico | 2013-01-01 → 2017-08-31 | External factor (Ecuador is oil-dependent) |
| `holidays_events.csv` | 350 | date, type, locale, locale_name, description, transferred | 2012-03-02 → 2017-12-26 | External factor: national, regional and local holidays and events |
| `stores.csv` | 54 | store_nbr, city, state, type, cluster | — | Matches local and regional holidays to stores |
| `transactions.csv` | 83,488 | date, store_nbr, transactions | 2013-01-01 → 2017-08-15 | EDA only. It is known only after the day ends, so using it as a forecasting input would leak future information |
| `sample_submission.csv` | 28,512 | — | — | Not used |

Structure: 54 stores × 33 product families = **1,782 daily series**. Every date has exactly 1,782 rows, with no duplicates and no nulls in `train.csv`.

### 1.2 Quality findings and the cleaning needed

| # | Finding | Evidence | Treatment |
|---|---|---|---|
| 1 | **25 December is missing** every year | 2013–2016 Dec-25 dates absent | Insert the rows and flag them `closed=1`. Demand is 0 that day; the day is excluded from model training, D̄/σ_D and metrics |
| 2 | **1 January: stores closed** | 52–53 of 54 stores record zero sales on every 1 Jan | Flag as `closed` and treat as above |
| 3 | **Promotions are recorded only from 2014-04-01** | The first non-zero `onpromotion` is 2014-04-01 | Zeros before that date mean *missing data*, not *no promotions*. Training window starts **2014-04-01** (1,233 days, about 3.4 yearly cycles) |
| 4 | **Structural break in 2013** for BEVERAGES | Store 44 yearly means: 2013 = 4,275, 2014 = 7,689 units/day | Starting at 2014-04-01 also avoids this break |
| 5 | Late-opening stores | Stores 20, 21, 22, 29, 36, 42, 52, 53 open between 2013-05 and 2017-04 | Excluded from the curated shortlist |
| 6 | Long temporary closures | Stores 12, 14, 18, 24, 25, 30 have runs of 23–110 consecutive zero days | Excluded from the shortlist |
| 7 | Oil price gaps | 43 null prices plus 486 non-trading calendar days | Reindex to the daily calendar and interpolate linearly. Use a 7-day lagged average so the model never sees same-day information |
| 8 | Holiday table quirks | 12 `transferred=True` rows, 38 duplicate dates, and types Transfer, Bridge, Work Day, Additional and Event | Drop transferred originals and keep the `Transfer` dates. Bridge and Additional count as holidays; Work Day counts as a normal day; Events (e.g. Black Friday, World Cup) get their own flag. Local holidays are matched by city and regional ones by state |
| 9 | Earthquake shock | The data notes describe a magnitude 7.8 earthquake on 2016-04-16 | Earthquake-window flag from 2016-04-16 to 2016-05-16 |
| 10 | Fractional sales | 15.4% of rows overall. Weighed families (PRODUCE, MEATS) are fractional; **BEVERAGES, CLEANING and PERSONAL CARE are whole units (0% fractional)** | Prefer whole-unit families, so "units" are real countable units |
| 11 | Paydays | The data notes say public wages are paid on the 15th and the last day of the month | Payday flags as features |
| 12 | Sparse families | BOOKS 97% zeros, BABY CARE 94% | Out of scope; intermittent demand doesn't suit this formula |

### 1.3 Screening for "good enough to predict"

I screened **245 store × family series**: non-perishable families in stores that opened on time and had no long closures, covering 2014-04-01 → 2017-08-15. Each was tested on a 12-week holdout:

- **Moving average** (28-day): what a classic calculator uses for D̄.
- **Seasonal naive**: repeats last week.
- **Simple Ridge regression**: weekday, yearly seasonality, promotions, holidays, paydays, oil, earthquake and trend, with no lag features yet.

Median results by family:

| Family | Mean units/day | Promo–sales corr. | Weekday max/min | Dec lift | WAPE: moving avg | WAPE: seasonal naive | WAPE: simple Ridge |
|---|---|---|---|---|---|---|---|
| **BEVERAGES** | 2,681 | **0.56** | 1.79× | 1.21× | 23.6% | 14.8% | 20.5% |
| CLEANING | 1,097 | 0.10 | 1.66× | 1.12× | 23.3% | 19.5% | 17.5% |
| GROCERY I | 3,864 | 0.28 | 1.69× | 1.25× | 20.0% | 13.7% | 13.0% |
| PERSONAL CARE | 287 | 0.14 | 1.96× | 1.11× | 26.2% | 19.7% | 17.3% |
| HOME CARE | 201 | 0.44 | 1.89× | 1.28× | — | — | unstable (structural break) |
| LIQUOR, WINE, BEER | 83 | 0.23 | 6.23× | 1.98× | — | — | unstable (dry laws, holidays) |

**Conclusion:** the data clearly has learnable patterns:
- A strong weekly cycle (weekend peaks up to 2–3× weekdays).
- December peaks.
- A clear upward trend from 2014 to 2017.
- Promotion uplift, which is strongest in BEVERAGES.

Even the simple Ridge without lags beats the moving average by 12–34%. Adding lag and level terms (Holt-Winters, Ridge with lags, LSTM) is expected to also beat the seasonal-naive benchmark, which is the bar the real models must clear.

### 1.4 Default series: **Store 44 · BEVERAGES**

- Store 44 is the chain's largest store (Quito, type A). Its BEVERAGES sales average **10,865 units/day** over the last 365 open days, in whole units.
- Promotions run on 87% of days, and the promotion–sales correlation is 0.50.
- The only zero days since 2014-04 are the three 1 January closures.
- Of all the candidates, simple-model errors explain the largest share of its demand variability (ratio of forecast-error SD to raw-demand SD = 0.64).
- The dashboard also offers a **curated shortlist** of about 6 clean non-perishable series (e.g. 44 CLEANING, 44 PERSONAL CARE, 45 GROCERY I, 47 BEVERAGES, 1 CLEANING). An advanced selector allows any store × family, with data-quality warnings.

> **Modelling note:** one "item" here is a *product family at one store*, which works as an aggregate planning unit replenished from the distributor. Favorita does not publish data at true SKU level. This limitation will be stated in the dashboard's Method page.

### 1.5 Hand calculation of the slide formula (cross-check preview)

Store 44 BEVERAGES, last 365 open days (n = 363). Assumed L = 7 days, σ_L = 2 days.

- D̄ = **10,865.36 units/day**, σ_D = **3,587.18 units/day** (CV 0.33)
- σ_D²·L = 3,587.18² × 7 = **90,074,884**
- D̄²·σ_L² = 10,865.36² × 2² = **472,223,789**
- Service level 95% → Z = Φ⁻¹(0.95) = 1.6449
- **SS = 1.6449 × √(90,074,884 + 472,223,789) = 39,004 units** (≈ 3.6 days of sales)
- **ROP = D̄·L + SS = 76,058 + 39,004 = 115,062 units**

Monte Carlo cross-check (200,000 draws):

| Assumption | Monte Carlo SS at 95% | Formula SS | Difference |
|---|---|---|---|
| Normal lead time, normal daily demand (the formula's own assumptions) | 40,031 | 39,004 | +2.6% |
| **Gamma lead time** (right-skewed, more realistic) | 42,377 | 39,004 | **+8.7%** |

Under Gamma lead times, a stock policy set with the formula's ROP actually runs out of stock **6.2%** of the time, not the intended 5%.

**What this tells an operations manager:**
1. **84% of the buffer protects against late deliveries, not demand swings.** Supplier reliability (σ_L) is the biggest lever.
2. The formula slightly **under-protects** when delays are skewed. That is exactly the gap the slide's Monte Carlo step is meant to expose.
3. Better forecasts shrink σ_D. They also move the *timing* of the reorder point ahead of promotions and peaks, through a forecast-driven D̄. Both effects will be measured, not assumed.

---

## 2. System architecture (resolving the Netlify constraint)

### 2.1 Principle
The site is a **fully static single-page app**, and **all computation runs in the user's browser, inside Web Workers.** It needs no server, has no cold starts, isn't bound by Netlify Functions' 10-second limit, and never sends data off the device. That last point is required anyway by the competition rules and the slide's no-sharing note.

### 2.2 Where each computation runs

| Computation | Runs in | Why |
|---|---|---|
| EDA, model selection, hyperparameter tuning; heavier reference models (SARIMAX, LightGBM, Keras LSTM); BO-vs-grid validation; golden values for unit tests | **Python notebook, offline on your machine** | Heavy, one-off work that documents the method. Never deployed |
| Extracting and cleaning the curated series from the Kaggle CSVs; building the data pack | **Python script, offline** (`scripts/build_data_pack.py`) | Runs once; produces a compact JSON (≈ 230 KiB, compressed in transit) |
| Loading and validating the pack, feature engineering | **Browser** (fetch + schema validation) | Under a second; the data stays in memory for the session |
| Forecasting: Holt-Winters (ETS), Ridge regression, LSTM (TensorFlow.js), adaptive ensemble | **Browser: forecast worker** | One series is about 1,233 points. ETS and Ridge fit in milliseconds; the LSTM trains in about 10–30 s on the WASM backend |
| **Retraining when new data arrives** | **Browser: forecast worker** | ETS and Ridge refit fully (ms). The LSTM fine-tunes from its current weights for a few epochs (about 2–5 s), and a full retrain is available on demand |
| Monte Carlo (10k–200k draws) | **Browser: simulation worker** | Under 100 ms using typed arrays and a seeded random-number generator |
| Bayesian optimisation (Gaussian process plus constrained Expected Improvement, about 40 evaluations, each a stochastic inventory simulation) | **Browser: simulation worker** | A few seconds in total, with progress streamed live |
| Netlify Functions | **Not used** | Nothing needs a server, and using one would mean uploading the data |

**What is precomputed:** only the *model configuration* chosen in the notebook (feature set, LSTM architecture, hyperparameter ranges) and the parameters of the synthetic demo generator. These are compiled into the code. **No forecasts, safety stocks, simulations or optimisation results are precomputed.** Everything is computed live from the loaded data.

The main thread only renders. Workers report progress through `postMessage`, so the UI stays responsive during heavy work.

### 2.3 Data access in the public deployment (licence + no-sharing) — **decided: real data loaded directly, short-lived site**

Kaggle's rules forbid publishing or redistributing the competition data, and the slide forbids sharing data with peers. A Netlify site is public.

**Decision history:**
1. Bring-your-own data (proposed).
2. A passphrase-locked encrypted pack (built in Part A).
3. **Final, at your request:** the passphrase is removed, because the site is live only for the faculty review and is taken down the following night.

How it works now:
- **Real data (default mode):**
  - `scripts/build_data_pack.py` extracts the curated series (sales, promotions, and the per-store calendar of holidays, events, closures and oil price) into `app/public/data/favorita-pack.json`.
  - The dashboard loads and validates it automatically on open.
- **Demo mode (synthetic, clearly labelled):** a data-source switch in the sidebar.
  - It runs on a synthetic series whose parameters (weekday profile, trend, seasonality, promotion uplift, noise) are aggregate statistics estimated in the notebook.
- **Never in Git or the bundle:** the raw Kaggle CSV folder, enforced by `.gitignore`.
- **Accepted risk (yours):** while the site is live, anyone with the link can download the curated series. Mitigation: a short live window, then **take down both the Netlify site and the GitHub repository** (or keep the repository private).

---

## 3. Forecasting design (Slide Step 1)

### 3.1 Inputs (all known in advance, so no leakage)

| Driver (slide wording) | Features |
|---|---|
| Historical data | Lags 7, 14 and 28; 7- and 28-day rolling means (built recursively during forecasting); trend |
| Seasonality | Day of week; yearly Fourier terms (k = 1–3); days to Christmas; paydays (15th and month-end) |
| Promotions | `log1p(onpromotion)`. Future values come from `test.csv` for the next 16 days, and after that from a **promotion planner** in the UI (default: the typical promo level) |
| External factors | National, regional and local holidays, events, bridge and work days; 7-day lagged oil price; earthquake window; store closures |

### 3.2 Models compared

| Slide category | Model | Implementation |
|---|---|---|
| Benchmarks | 28-day moving average (what the baseline calculator uses); seasonal naive | TypeScript |
| **Time series** | Holt-Winters ETS on `log1p` sales: weekly seasonality, damped trend, parameters fitted by Nelder–Mead | TypeScript |
| **Regression** | Ridge regression on `log1p` sales with every feature above, forecast recursively | TypeScript (closed form) |
| **LSTM** | 28-day input window × [scaled sales, weekday sin/cos, promo, holiday, payday] → LSTM(32) → Dense(1), one step ahead with recursive multi-step forecasting; early stopping; seeded | TensorFlow.js (WASM backend, lazy-loaded) |
| Adaptive ensemble | Weighted average of the three models, with weights ∝ 1/WAPE on the most recent backtest. Re-weighted on every update, which is the "adaptive" part | TypeScript |
| Reference only (notebook) | SARIMAX, LightGBM, Keras LSTM | Python, to show how the browser models compare with heavier tools |

### 3.3 Evaluation
- **Rolling-origin backtest:** 6 folds × 28-day horizon over the 24 weeks before the replay window. No random splits.
- **Metrics:** MAE, RMSE, **WAPE** (MAPE breaks on zero-sales days), **MASE** (relative to seasonal naive) and **bias** (% over or under).
- **Prediction intervals:** 80% and 95% bands from empirical out-of-sample residual quantiles at each horizon (split-conformal). Actual coverage is reported, so the bands are checked rather than assumed.
- Results are reported honestly even if the LSTM does not win. Single-series LSTMs often lose to good regression.

### 3.4 Adding new data and continuous updating
1. **Live replay of real data:** the last 8 weeks of real sales (2017-06-21 → 2017-08-15) are held back as "arriving data". A button releases the next day or week of genuine actuals. Models update, forecasts and intervals shift, accuracy is re-scored on truly unseen data, and a **model version log** records each update.
2. **Manual entry:** date, units sold and promotion count, with validation.
3. **CSV upload:** new rows in Kaggle's format, or a simple `date,sales,onpromotion` file.
4. **Lead-time records (optional):** order and receipt dates. When given, L and σ_L are *estimated from data* rather than assumed.
5. **Reset to original data.** All changes persist in IndexedDB.

Each update triggers: rebuild features → refit ETS and Ridge → fine-tune the LSTM → recompute backtest errors, ensemble weights and σ_e → recompute safety stock, ROP and Monte Carlo → flag the Bayesian optimisation result as stale, with a one-click re-run.

---

## 4. Safety stock and Monte Carlo design (Slide Step 2)

### 4.1 The slide's formula, implemented exactly
**SS = Z · √( σ_D² · L + D̄² · σ_L² )**, with Z = Φ⁻¹(service level).

**Units:** D̄ and σ_D in units/day; L and σ_L in days, so SS is in units. A unit test fixes the hand calculation in §1.5.

| | Baseline calculator | AI-enhanced (dynamic) |
|---|---|---|
| D̄ | Mean of the last 365 open days (static) | **Mean forecast over the next L days**, recomputed daily (captures promotions, weekdays, December, trend) |
| σ_D | SD of raw daily demand (counts *predictable* weekday swings as risk) | **σ_e: SD of out-of-sample one-step forecast errors** over the last 90 days (only what is genuinely unpredictable) |
| L, σ_L | Assumed (default 7 d, 2 d) or estimated from uploaded lead-time records | Same |
| ROP | D̄·L + SS | Σ forecast over L + SS<sub>t</sub> (moves ahead of peaks) |

The formula is unchanged; only its inputs become forecast-driven. The dashboard shows the formula with live numbers substituted, so it can be hand-checked on screen.

### 4.2 Monte Carlo (risk scenarios and buffer requirements)
- **Lead-time model:** Gamma, moment-matched with shape k = (L/σ_L)² and scale θ = σ_L²/L. Selectable alternatives: truncated normal, or empirical from uploaded records.
- **Demand during lead time:** the forecast path over the sampled lead time (with a fractional final day), plus residuals drawn by a 7-day *block* bootstrap of real forecast errors, which keeps their autocorrelation and skew.
- **Outputs:**
  - Distribution of demand during lead time.
  - Monte Carlo reorder point at the target service level, and **SS<sub>MC</sub>** compared with SS from the formula.
  - **Stockout probability** at the formula's ROP.
  - Expected units short.
  - Fill rate for a chosen order quantity Q.
- **Scenarios:**
  - Normal
  - Promotion ahead
  - Supplier delay (+3 days)
  - Lead-time volatility ×2
  - Demand surge (+25%)
  - December peak
  - Custom
- **Verification built in:** with normal demand and normal lead time, the Monte Carlo result must reproduce the formula within the known compound-distribution gap (about +2.6%, §1.5). A convergence chart shows the estimate stabilising as N grows.

---

## 5. Bayesian optimisation design (Slide Step 3)

- **Policy simulated:** daily continuous review (R, Q) with lost sales (standard retail):
  - Arrivals are received each morning.
  - Demand comes from forecast-driven scenario paths: mean path plus block-bootstrapped residuals.
  - When inventory position ≤ R, an order of Q is placed, with lead time sampled from the Gamma distribution.
- **Decision variables:**
  - **AI mode (default):** a safety buffer *b* on top of the forecast lead-time demand, so R<sub>t</sub> = D̂<sub>L,t</sub> + b, plus the order quantity Q.
  - **Static mode** (for comparison): a fixed R and Q.
- **Objective:** minimise expected cost per day = holding + ordering + stockout penalty. It is estimated over a 13-week forward horizon × 64 scenario paths, using common random numbers to reduce noise.
- **Constraints, all adjustable in the UI:**
  - Fill rate ≥ target (default 95%)
  - Peak on-hand ≤ storage capacity
  - Average inventory value ≤ working-capital budget
  - Q ≥ supplier minimum order quantity
  - Deliveries per week ≤ dock limit
- **Defaults scale with the series** (e.g. capacity = 18 days of average demand) and are documented as assumptions:
  - Unit cost $0.70
  - Holding 25% per year
  - Ordering cost $120 per order
  - Stockout penalty $0.50 per unit lost
- **Algorithm:**
  - Gaussian process with a Matérn-5/2 ARD kernel on normalised inputs, a noise term, and hyperparameters chosen by maximising the marginal likelihood (multi-start Nelder–Mead).
  - **Constrained Expected Improvement:** EI × P(feasible), with a second GP modelling the fill-rate slack.
  - 8 space-filling starting points (Sobol), then about 30 iterations.
  - The acquisition function is maximised over a dense candidate set.
- **Validation:** results are compared with an exhaustive 30 × 30 grid (900 simulations), run in the notebook and through a "Verify" button. The pass criterion is BO reaching within about 1–2% of the grid optimum using roughly 4% of the evaluations.
- **AI-driven sensitivity analysis, from the GP surrogate instead of manual sweeps:**
  - Tornado chart: how the ROP changes for ±20% in L, σ_L, σ_e, D̄ and service level.
  - Heatmap of ROP across service level × σ_L.
  - **Cost–service frontier:** optimal cost against the fill-rate target.
  - **Shadow price of capacity:** cost saved per extra 10% of storage.

---

## 6. "Enhancement" comparison: baseline vs AI

Both policies replay the **held-back real demand** (8–12 weeks), using identical lead-time draws over 500 repetitions:

| Policy | D̄ | σ input | ROP | Q |
|---|---|---|---|---|
| Baseline | Historical mean | Raw σ_D | Static formula ROP | Economic order quantity (EOQ) |
| AI-enhanced | Forecast | σ_e | Monte Carlo-calibrated, dynamic R<sub>t</sub> | Q* from Bayesian optimisation |

Reported metrics:
- Achieved fill rate and cycle service level
- Stockout days
- Average inventory (units and days of cover)
- Orders placed
- Total cost
- Inventory and working capital released

The numbers are reported whichever way they come out.

> **As built (Part F):**
> - **Window:** the replay uses the last **8 held-back weeks**, with **three** policies:
>   - baseline;
>   - **AI formula**, re-planned weekly from a fresh forecast;
>   - **AI optimised**, with the BO buffer and Q\*.
> - **Delays and warm-up:** 500 shared delay paths. Each policy first runs through 28 days of real sales before the window, unscored, so the simulation's starting stock does not distort the results.
> - **Scorecard:**
>   - cost split;
>   - fill rate and its 5–95% range;
>   - cycle service level;
>   - stockout days;
>   - average and peak stock;
>   - orders per week;
>   - which constraints each policy breaks;
>   - paired "how often is AI better under the same delays" shares.
> - **Results:** see `docs/VERIFICATION.md` §F2.

---

## 7. Dashboard pages and visuals

A persistent **control rail** holds:
- Series selector
- Service level (80–99.9%)
- Lead time L (1–21 days)
- Lead-time variability σ_L (0–7 days)
- Lead-time distribution
- Data mode (Real / Demo)

Optimisation constraints sit on the Optimisation page.

| # | Page | Key interactive visuals | Plain-language callout (generated from the live numbers) |
|---|---|---|---|
| 1 | **Command Center** | KPI tiles (ROP baseline vs AI, SS, stockout risk, forecast WAPE, days of cover); 30-day sparkline of SS and ROP; alert ribbon | "Reorder when stock falls to X units; at 95% service you carry Y days of buffer…" |
| 2 | **Data & Updates** | Data health panel; series explorer (full history with zoom); live replay controls; manual and CSV entry; model version log; accuracy-over-time chart | "The latest week lowered forecast error from A% to B%…" |
| 3 | **Demand Forecast** | **Forecast vs actual with 80/95% bands**; model comparison table and bar chart; backtest-by-fold chart; demand drivers (weekday profile, promotion uplift, holiday effects, oil); promotion planner | "Promotions lift sales by ~N%; the ensemble is most accurate this week…" |
| 4 | **Safety Stock & Risk** | Formula panel with live substitution; split of the variance (demand vs lead time); **Monte Carlo lead-time-demand histogram with ROP line and shaded stockout tail**; lead-time distribution; formula vs Monte Carlo comparison; scenario cards; dynamic SS and ROP over the next 8 weeks | "84% of the buffer covers late deliveries; a 3-day delay raises stockout risk to Z%…" |
| 5 | **Inventory Simulation** | **Inventory level vs reorder point** (sawtooth) for baseline and AI policies, with order and arrival markers and stockout shading; replay controls | "The AI policy avoided K stockout days while holding M% less stock…" |
| 6 | **Optimisation & Sensitivity** | Constraint panel; **BO convergence chart**; GP surrogate heatmap with feasibility boundary and sampled points; 1-D slices with uncertainty bands; optional 3-D surface (WebGL, lazy-loaded); **tornado chart; service level × σ_L heatmap; cost–service frontier** | "Optimum: reorder at X, order Y; storage is the binding constraint; +10% space saves $Z/month…" |
| 7 | **Impact: Baseline vs AI** | Side-by-side scorecard with constraint check; real demand in the held-back weeks; reorder levels (static vs weekly AI); stock on hand under identical delays; share of scenarios meeting the fill target (built in place of the waterfall) | "Versus the classic calculator, the AI policy…" |
| 8 | **Method & Assumptions** | Formulas, units, data cleaning, lead-time assumption, limitations, licence note | — |

---

## 8. Visual design language: "Warehouse control room"

- **Canvas:**
  - Deep navy-graphite background (#070B14 → #0B1324) with a faint blueprint grid, like racking floor plans.
  - **Glass panels:** `rgba(255,255,255,0.04)`, `backdrop-blur(16px)`, a 1 px border at `rgba(148,163,184,0.12)`, and soft inner glow.
- **Semantic accents:**

  | Colour | Hex | Meaning |
  |---|---|---|
  | Signal cyan | #22D3EE | Forecasts |
  | **Safety amber** | #F59E0B | Reorder point and warnings (hazard-tape cue) |
  | Stock green | #34D399 | Healthy stock |
  | Stockout red | #F43F5E | Stockouts |
  | Optimiser violet | #A78BFA | AI and Bayesian optimisation |

  Palettes are colour-blind-safe, and status is never conveyed by colour alone.
- **Typography:** Inter for the UI and JetBrains Mono with tabular figures for KPIs. Both are self-hosted.
- **Motion (Framer Motion, purposeful):**
  - Page transitions of 200 ms
  - KPI count-ups when values change
  - Optimiser points animating onto the surface as each iteration lands
  - Monte Carlo histogram filling as draws stream in
  - Respects `prefers-reduced-motion`
- **Icons:** Lucide (warehouse, truck, package, gauge).
- **Accessibility and layout:** WCAG AA contrast, keyboard navigation, and responsive from 375 px to 1,440 px+. On phones, the control rail collapses into a sheet.

---

## 9. Technology stack

| Concern | Choice | Reason |
|---|---|---|
| Build / framework | Vite + React 18 + TypeScript (strict) | Fast, static output that suits Netlify |
| Styling / motion | Tailwind CSS + Framer Motion | Glass design system; smooth animation |
| Charts | Apache ECharts (canvas, tree-shaken) | Handles 1,000+ points, bands, histograms, heatmaps, zoom and brushing smoothly |
| 3-D surface (optional) | three.js via react-three-fiber, lazy-loaded | WebGL used only where it adds insight |
| State | Zustand | Simple, fast |
| Workers | Comlink | Typed worker remote calls |
| ML | TensorFlow.js (WASM backend, lazy) | LSTM training and fine-tuning in the browser |
| CSV / storage | PapaParse (streaming) + idb-keyval (IndexedDB) | Parses 122 MB locally; persistence |
| Tests | Vitest + Playwright | Unit, parity and end-to-end tests |
| Notebook | Python 3.11, pandas, statsmodels, scikit-learn, LightGBM, TensorFlow/Keras, SciPy | Reference modelling and verification |

---

## 10. File and folder structure

```
Lab Activity - 1/
├─ store-sales-time-series-forecasting/   # raw Kaggle files (git-ignored, never deployed)
├─ notebooks/
│  └─ 01_eda_model_selection_verification.ipynb
├─ scripts/
│  └─ build_data_pack.py                  # Kaggle CSVs → curated series → compact JSON data pack
├─ docs/
│  ├─ Phase2_Implementation_Plan.md
│  └─ VERIFICATION.md                     # hand calcs, parity results, BO-vs-grid
├─ app/
│  ├─ index.html · package.json · vite.config.ts · tailwind.config.ts · tsconfig.json
│  ├─ netlify.toml
│  ├─ public/                              # icons, self-hosted fonts
│  │  └─ data/favorita-pack.json          # curated real-data pack, loaded on open
│  ├─ src/
│  │  ├─ main.tsx · App.tsx · routes.tsx
│  │  ├─ pages/          # CommandCenter, DataUpdates, Forecast, SafetyStock,
│  │  │                  # InventorySim, Optimization, Impact, Method
│  │  ├─ components/     # layout/, glass/, charts/, controls/, explain/
│  │  ├─ state/          # zustand stores
│  │  ├─ data/           # ingest, cleaning, holidays, features, demo generator
│  │  ├─ models/         # ets, ridge, lstm, ensemble, backtest, intervals, metrics
│  │  ├─ inventory/      # normal (Φ, Φ⁻¹), rng (seeded, gamma), safetyStock,
│  │  │                  # monteCarlo, policySim, costs
│  │  ├─ optimize/       # linalg, kernels, gp, acquisition, sobol, bayesOpt, sensitivity
│  │  ├─ workers/        # ingest.worker, forecast.worker, sim.worker
│  │  └─ lib/            # formatting, plain-language explainers
│  └─ tests/
│     ├─ unit/           # formula, Φ⁻¹, gamma, MC, metrics, ETS/Ridge parity, GP, BO
│     ├─ golden/         # reference numbers exported by the notebook (no raw data)
│     └─ e2e/            # Playwright smoke tests
└─ .gitignore
```

### Why a supporting Python notebook is worth it
1. **Documented EDA and data decisions** for your submission.
2. **A fair model comparison:** heavier reference models (SARIMAX, LightGBM, Keras LSTM) show whether the lighter browser models give anything up.
3. **Hyperparameter and architecture selection** for the browser models.
4. **Independent verification** with SciPy, statsmodels and scikit-learn. It exports golden numbers that the TypeScript unit tests must match: formula, Φ⁻¹, ETS/Ridge parity, metrics and Monte Carlo statistics.
5. **Bayesian optimisation vs exhaustive-grid validation.**

It runs offline and is never deployed.

---

## 11. Testing approach

| Level | What | Pass criterion |
|---|---|---|
| Unit | Φ⁻¹ vs SciPy | |error| < 1e-9 |
| Unit | Slide formula on the §1.5 numbers | SS = 39,004 ± 1; ROP = 115,062 ± 1 |
| Unit | Gamma sampler | Mean and SD within 0.5% at N = 10⁶ |
| Unit | Monte Carlo under normal-normal assumptions | Within 3% of the formula |
| Unit | Metrics vs scikit-learn | Exact to 1e-9 |
| Parity | ETS and Ridge vs statsmodels and scikit-learn on fixed synthetic fixtures | Within agreed tolerance |
| Unit | GP posterior vs scikit-learn; BO on the Branin test function | Close to the analytic optimum |
| Invariant | Policy simulation balance: on-hand + on-order − demand served is conserved | Exact |
| Integration | New data → forecasts, σ_e, SS and ROP all change; version log grows | Pass |
| End-to-end (Playwright) | Demo load, replay a week, CSV upload, Monte Carlo, BO run, page navigation, at 375 / 768 / 1,440 px | **Zero console errors** |
| Performance | Main thread never blocked > 50 ms during training, Monte Carlo or BO; Lighthouse ≥ 90 performance | Pass |
| Acceptance | Every row of the traceability table (§13) checked in Phase 4 | All ✓ |

---

## 12. Netlify deployment steps (Phase 4)

1. Build the data pack (`python scripts/build_data_pack.py`). Initialise git; confirm `.gitignore` excludes `store-sales-time-series-forecasting/`, `data/`, `*.csv`, `.env*`, `node_modules/` and `dist/`.
2. Push to the GitHub repo you provide.
3. Netlify → *Add new site → Import from Git* → select the repo.
4. Settings: base directory `app`, build command `npm run build`, publish directory `app/dist`, Node 22.
5. `netlify.toml`:
   - SPA fallback (`/* → /index.html 200`)
   - Long-cache headers for hashed assets
   - Security headers
   - Cross-origin-isolation headers (COOP/COEP) so TensorFlow.js WASM can use threads
6. Deploy. Run a smoke test on the live URL: the real data loads, then the synthetic demo. Check the console is clean. Project, repo and site name: **ReorderIQ** (`reorder-iq`).

---

## 13. Traceability: slide requirement → dashboard feature

| Slide / brief requirement | Delivered by | Verified by |
|---|---|---|
| **Objective:** predictive intelligence | Forecast page: ETS, Ridge, LSTM and adaptive ensemble | Backtest metrics vs benchmarks |
| **Objective:** adaptive decision-making | Dynamic D̄, σ_e, SS<sub>t</sub> and ROP<sub>t</sub>, recomputed on every update; adaptive ensemble weights | Integration test: new data changes ROP |
| **Objective:** real-time optimisation | In-browser Bayesian optimisation with live progress; re-run on new data | End-to-end BO run; main thread not blocked |
| **1.** ML models: time series, regression, LSTM | Holt-Winters ETS, Ridge, TensorFlow.js LSTM (plus notebook SARIMAX, LightGBM, Keras) | Model comparison table |
| **1.** Historical data, seasonality, promotions, external factors | Feature set (§3.1); demand-drivers panel; promotion planner (0–3× future promotions, re-forecast in < 1 s) | Feature-importance and uplift views; planner check (VERIFICATION §F4) |
| **1.** Keep the possibility of adding new data | Data & Updates: live replay, manual entry, CSV upload, lead-time records | End-to-end test |
| **1.** Continuously update forecasts as data arrives | Auto refit and fine-tune pipeline; version log; accuracy-over-time chart | Accuracy re-scored on unseen weeks |
| **2.** Probabilistic models for demand variability and lead-time uncertainty | Formula with σ_e and Gamma lead time; distribution selector | Unit tests; Monte Carlo agreement |
| **2.** Monte Carlo for risk scenarios and buffer requirements | Safety Stock & Risk page: histogram, stockout tail, scenarios, SS<sub>MC</sub> | Convergence check; normal-normal check |
| **2.** Formula SS = Z·√(σ_D²L + D̄²σ_L²) | Formula panel with live substitution | Hand-calc unit test (§1.5) |
| **3.** Bayesian optimisation for optimal ROP under constraints | Optimisation page: GP + constrained EI; capacity, budget, MOQ, fill-rate and dock constraints | BO vs 900-point grid |
| **3.** Replaces manual sweeps | Surrogate-based tornado chart, heatmap, frontier and shadow price | Matches direct simulation at spot checks |
| Note: no sharing of data with peers | Individual dataset and work; raw CSVs kept out of Git; site live only for the faculty review, then taken down; synthetic demo mode | Repo audit: no raw Kaggle CSVs |
| Brief: baseline vs AI comparison | *Baseline vs AI* page: 8 held-back weeks, three policies, 500 shared delay paths, constraint check | Replay on held-back real demand; no-look-ahead and warm-up unit tests (VERIFICATION §F1–F3) |
| Brief: required visuals (forecast bands, MC with stockout risk, inventory vs ROP, BO convergence, ROP sensitivity) | Pages 3, 4, 5 and 6 | Visual QA checklist |
| Brief: controls for service level, L, σ_L and constraints | Control rail and constraint panel | End-to-end test |
| Brief: plain-language explanations | Callout on every page, generated from live numbers | Review |
| Brief: responsive, fast, no console errors, UI stays responsive | Web Workers, code-splitting, responsive layout | Playwright + Lighthouse |
| Brief: formula exact, consistent units, hand cross-checks, accuracy metrics and model comparison | §4.1, §1.5, §3.3, `docs/VERIFICATION.md` | Unit tests + verification document |

---

## 14. Phase 3 build parts (each stops for your "continue")

| Part | Contents |
|---|---|
| **A** | Notebook: EDA and data decisions, golden values. Data-pack builder (built with encryption first; passphrase later removed, §2.3). App scaffold and data layer: loading, cleaning, features, demo generator. (Model-selection sections of the notebook are added in Part B, alongside the browser models.) |
| **B** | Forecasting models, backtesting and continuous-update pipeline (workers) |
| **C** | Safety stock (formula + dynamic) and Monte Carlo, with verification |
| **D** | Inventory policy simulator and Bayesian optimisation (GP, constrained EI, sensitivity), with BO-vs-grid verification |
| **E** | Dashboard UI: design system, pages and charts |
| **F** | Integration, Impact page and polish → then Phase 4 testing and deployment. *Delivered:* Baseline vs AI replay, promotion planner, lazy-loaded pages and vendor chunks, `netlify.toml`, README, project named ReorderIQ |

## 15. Assumptions to confirm
1. Lead time defaults: **L = 7 days, σ_L = 2 days**, Gamma-distributed (adjustable; replaceable by uploaded records).
2. Cost defaults: **unit cost $0.70, holding 25%/yr, ordering $120/order, stockout penalty $0.50/unit, capacity 18 days of demand** (all adjustable).
3. Service level default: **95%** (Z = 1.6449).
4. Data access in the public deployment: **decided: real Favorita pack loaded directly (no passphrase), site taken down after the faculty review, plus synthetic demo mode** (§2.3).
