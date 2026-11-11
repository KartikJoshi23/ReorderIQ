"""
Reference outputs for the TypeScript parity tests (synthetic inputs only — no Kaggle data).

    .venv\\Scripts\\python scripts/make_parity_golden.py

Writes app/tests/golden/parity.json with:
  * Holt-Winters (statsmodels ExponentialSmoothing, additive damped trend + additive weekly season,
    fixed parameters and known initial state): one-step fitted values, final states, 14-day forecast;
  * Ridge (scikit-learn Ridge on StandardScaler-standardised inputs): coefficients and intercept;
  * metrics (MAE, RMSE) and numpy linear-interpolation quantiles.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

OUT = Path(__file__).resolve().parents[1] / "app" / "tests" / "golden" / "parity.json"
rng = np.random.default_rng(11)

# --- Holt-Winters ------------------------------------------------------------------------------
n = 140
season = np.array([0.0, -0.08, -0.02, -0.15, 0.0, 0.35, 0.42])
season -= season.mean()
y = 9.0 + 0.002 * np.arange(n) + season[np.arange(n) % 7] + rng.normal(0, 0.05, n)
params = {"alpha": 0.3, "beta": 0.05, "gamma": 0.2, "phi": 0.9}
init = {"level": 9.0, "trend": 0.001, "season": season.tolist()}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                 initialization_method="known", initial_level=init["level"],
                                 initial_trend=init["trend"], initial_seasonal=init["season"])
    fit = model.fit(smoothing_level=params["alpha"], smoothing_trend=params["beta"],
                    smoothing_seasonal=params["gamma"], damping_trend=params["phi"], optimized=False)
ets = {
    "y": y.tolist(), "params": params, "init": init,
    "fitted": np.asarray(fit.fittedvalues).tolist(),
    "finalLevel": float(fit.level[-1]), "finalTrend": float(fit.trend[-1]),
    "forecast": np.asarray(fit.forecast(14)).tolist(),
    "sse": float(fit.sse),
}

# --- Ridge -------------------------------------------------------------------------------------
rows, cols = 200, 6
X = rng.normal(0, 1, (rows, cols)) * np.array([1, 5, 0.1, 2, 1, 10]) + np.array([0, 3, 1, -2, 0, 50])
X[:, 4] = (rng.random(rows) < 0.2).astype(float)            # a 0/1 dummy like holiday flags
beta_true = np.array([0.5, -0.1, 3.0, 0.2, 0.8, 0.01])
yr = X @ beta_true + 2.0 + rng.normal(0, 0.3, rows)
Z = StandardScaler().fit_transform(X)
ridge = Ridge(alpha=3.0).fit(Z, yr)
ridge_gold = {"X": X.tolist(), "y": yr.tolist(), "lambda": 3.0, "coef": ridge.coef_.tolist(),
              "intercept": float(ridge.intercept_), "predFirst": float(ridge.predict(Z[:1])[0])}

# --- Metrics and quantiles ---------------------------------------------------------------------
a = rng.uniform(50, 150, 30)
f = a + rng.normal(0, 10, 30)
metrics = {"actual": a.tolist(), "forecast": f.tolist(), "mae": float(mean_absolute_error(a, f)),
           "rmse": float(np.sqrt(mean_squared_error(a, f))), "wape": float(np.abs(a - f).sum() / a.sum()),
           "bias": float((f - a).sum() / a.sum())}
qs = [0.025, 0.1, 0.5, 0.9, 0.975]
quant = {"values": f.tolist(), "q": qs, "expected": [float(np.quantile(f, q)) for q in qs]}

# --- Standard normal (scipy.stats.norm) ---------------------------------------------------------
from scipy.stats import norm, gamma as gamma_dist  # noqa: E402
xs = [-8, -5, -3, -1.5, -0.5, 0, 0.3, 1, 1.6448536269514722, 2.5, 3, 5, 8, 12]
ps = [1e-12, 1e-6, 0.001, 0.01, 0.2, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1 - 1e-6]
normal = {
    "x": xs, "cdf": [float(norm.cdf(x)) for x in xs], "sf": [float(norm.sf(x)) for x in xs],
    "pdf": [float(norm.pdf(x)) for x in xs],
    "loss": [float(norm.pdf(x) - x * norm.sf(x)) for x in xs],
    "p": ps, "ppf": [float(norm.ppf(p)) for p in ps],
}
shape, scale = (7 / 2) ** 2, 2 ** 2 / 7
gam = {"shape": shape, "scale": scale, "mean": float(gamma_dist.mean(shape, scale=scale)),
       "sd": float(gamma_dist.std(shape, scale=scale)), "p95": float(gamma_dist.ppf(0.95, shape, scale=scale))}

# --- Gaussian process (scikit-learn, fixed hyperparameters) --------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402
Xg = rng.random((15, 2))
yg = np.sin(6 * Xg[:, 0]) + Xg[:, 1] ** 2 + rng.normal(0, 0.05, 15)
hyper = {"lengthscales": [0.3, 0.5], "signalVar": 1.7, "noiseVar": 1e-3}
kern = (ConstantKernel(hyper["signalVar"], "fixed") * Matern(length_scale=hyper["lengthscales"], length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(hyper["noiseVar"], "fixed"))
# alpha=0: sklearn otherwise adds 1e-10 to the diagonal on top of the WhiteKernel noise.
gpr = GaussianProcessRegressor(kern, alpha=0.0, optimizer=None, normalize_y=False).fit(Xg, yg)
Xs = rng.random((6, 2))
mu, sd = gpr.predict(Xs, return_std=True)            # sd includes the white-noise variance
gp = {"X": Xg.tolist(), "y": yg.tolist(), "hyper": hyper, "Xs": Xs.tolist(), "mean": mu.tolist(),
      "sdWithNoise": sd.tolist(), "lml": float(gpr.log_marginal_likelihood_value_)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"ets": ets, "ridge": ridge_gold, "metrics": metrics, "quantile": quant,
                           "normal": normal, "gamma": gam, "gp": gp}, indent=1))
print(f"wrote {OUT}  (ETS SSE {ets['sse']:.6f}, ridge coef {np.round(ridge.coef_, 4).tolist()})")
