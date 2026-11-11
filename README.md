# ReorderIQ — AI-powered reorder points for grocery retail

MAIB · AI in Operations · Session II exercise: *AI Applications in Reorder Point & Inventory Management*.

An interactive dashboard that enhances the classic reorder-point calculator with:
- **predictive intelligence**: demand forecasting;
- **adaptive decision-making**: dynamic safety stock and Monte Carlo risk;
- **real-time optimisation**: Bayesian optimisation under constraints.

It runs on real FMCG grocery data from the Kaggle *Store Sales – Time Series Forecasting* dataset (Corporación Favorita, Ecuador). **Every model runs in your browser.**

> **Data notice.** The Kaggle data is used for academic education under the competition rules. It is shown on the live site only for the faculty review. **After the review, take down the Netlify site and the GitHub repository** (or make the repository private). The raw Kaggle CSVs are never committed.

---

## What it does (mapped to the slide)

| Slide step | Where | How |
|---|---|---|
| **1. Demand forecasting** (time series, regression, LSTM; seasonality, promotions, external factors; add new data; continuous updates) | *Demand Forecast*, *Data & Updates* | **Models:** Holt-Winters ETS, Ridge regression with drivers and lags, and a TensorFlow.js LSTM, combined by an adaptive ensemble. **Drivers:** weekday, yearly seasonality, holidays and events, promotions, paydays, Christmas, oil price, earthquake. **Accuracy:** rolling 6 × 28-day backtest with 80/95% prediction bands. **New data:** live replay of held-back weeks, manual entry or CSV upload. Each update refits ETS and Ridge, **fine-tunes the LSTM** and re-weights the ensemble |
| **2. Dynamic safety stock** (probabilistic models; Monte Carlo; formula SS = Z·√(σ_D²·L + D̄²·σ_L²)) | *Safety Stock & Risk* | **Formula:** the slide's formula with live numbers, fed two ways (baseline: last-year statistics; AI: forecast demand and forecast-error RMSE). **Lead times:** Gamma, normal or empirical (upload order/receipt records). **Monte Carlo:** demand during the lead time using real forecast errors, with stockout risk and stress scenarios |
| **3. AI-driven sensitivity analysis** (Bayesian optimisation for optimal reorder points under constraints) | *Optimisation*, *Inventory Simulation* | **Optimiser:** Gaussian process with constrained Expected Improvement over buffer × order quantity. **Objective:** simulated holding + ordering + lost-sales cost. **Constraints:** fill rate, storage, budget, MOQ, deliveries. **Checks and views:** validation against a 900-point grid; tornado chart, heatmap, cost–service frontier and storage shadow price |
| **Objective: the enhancement** | *Baseline vs AI*, *Command Center* | Held-back weeks of **real** demand replayed through the classic, AI-formula and AI-optimised policies, with identical delivery delays |

---

## Architecture

- **Static React + TypeScript app** (Vite). No backend and no Netlify Functions.
- **All computation runs in Web Workers**, so the page stays responsive:
  - **forecast worker:** ETS, Ridge, LSTM (TensorFlow.js on the WebAssembly backend), backtests, ensemble;
  - **risk worker:** Monte Carlo;
  - **optimiser worker:** Bayesian optimisation, grid check, policy simulation.
- **Precomputed:** only the model configuration chosen offline (feature set, LSTM architecture, Ridge λ = 30) and the synthetic demo generator's parameters. Every forecast, safety stock, simulation and optimisation is computed live.
- **Offline (Python):** EDA, model selection, reference models (SARIMAX, LightGBM, Keras LSTM) and independent verification. The notebook exports reference values that the TypeScript tests must match.

## Repository layout

```
app/                     dashboard (Vite + React + TypeScript)
  public/data/           favorita-pack.json — curated real-data pack loaded on open
  src/data/              loading, schema, calendar features, new-data parsing, synthetic demo
  src/models/            ETS, Ridge, LSTM, ensemble, backtest, intervals, engine
  src/inventory/         safety stock (slide formula), lead times, Monte Carlo, policy simulation, replay
  src/optimize/          Gaussian process, Bayesian optimisation, sensitivity
  src/workers/           forecast and simulation workers
  src/pages/, src/ui/    dashboard pages and design system
  tests/                 92 unit and parity tests (Vitest)
  scripts/               offline real-data reports (backtest, safety stock, optimisation)
scripts/                 Python: data preparation, data-pack builder, reference models, parity values
notebooks/               01_eda_model_selection_verification.ipynb
docs/                    Phase2_Implementation_Plan.md, VERIFICATION.md
netlify.toml             deployment configuration
```

## Run locally

**Requirements:** Node 22+ and Python 3.11.

1. **Data pack.** This is only needed if `app/public/data/favorita-pack.json` must be rebuilt. Put the Kaggle CSVs in `store-sales-time-series-forecasting/`, then run:
   ```bash
   python scripts/build_data_pack.py
   ```
2. **Dashboard:**
   ```bash
   cd app
   npm install
   npm run dev
   ```
   Then open http://localhost:5173.
3. **Tests and production build:**
   ```bash
   npm test
   npm run build
   npm run preview
   ```
4. **Notebook** (optional; reference models need TensorFlow and LightGBM):
   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

## Deploy to Netlify

1. Push the repository to GitHub as `reorder-iq` (a private repository is recommended).
2. In Netlify, choose **Add new site → Import an existing project** and select the repository. Set the site name to `reorder-iq` (→ `reorder-iq.netlify.app`; add a suffix if that name is taken). `netlify.toml` supplies the settings:
   - base `app`
   - build `npm run build`
   - publish `dist`
   - Node 22
   - cross-origin-isolation headers
3. Deploy, open the site and check that the status line reaches "Model v… · LSTM trained".
4. **After the faculty review:** delete the Netlify site and the GitHub repository (or make it private).

## Verification

See [`docs/VERIFICATION.md`](docs/VERIFICATION.md):
- the formula hand calculation (SS 39,004.15 / ROP 115,061.63);
- Python ↔ TypeScript parity (statsmodels, scikit-learn, SciPy);
- Monte Carlo and policy-simulator NumPy re-implementations;
- Bayesian optimisation vs the 900-point grid;
- browser checks.

## Key assumptions (all adjustable in the dashboard)

| Area | Default |
|---|---|
| Lead time | Gamma, L = 7 days, σ_L = 2 days (the dataset has no supplier lead times) |
| Service level | 95% |
| Costs | unit $0.70, holding 25% per year, $120 per order, $0.50 per lost unit |
| Storage / budget | 18 / 12 days of demand |
| Policy | continuous review (R, Q) with lost sales |
