"""Used car prices: should the dealer buy this car?

Reproduces every number on the slides. Run: python analysis.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42          # fixed so the split (and every number downstream) is reproducible
TEST_SHARE = 0.30
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
TARGET = "Sale_Price"


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA / "used_car_sales.csv")
    df["Accident"] = (df["Accident"].str.strip().str.lower() == "yes").astype(int)  # yes=1, no=0
    return df


# ---------------------------------------------------------------------------
# Step 1: split the data 70/30
# ---------------------------------------------------------------------------
sales = load_sales()
train, test = train_test_split(sales, test_size=TEST_SHARE, random_state=SEED)
train.to_csv(OUT / "train.csv", index_label="row")
test.to_csv(OUT / "test.csv", index_label="row")

print("STEP 1: DATA SPLIT")
print(f"  Total cars: {len(sales)}  |  Training: {len(train)}  |  Test: {len(test)}  (seed={SEED})")
summary = pd.DataFrame({
    "train_mean": train[[TARGET] + FEATURES].mean(),
    "test_mean": test[[TARGET] + FEATURES].mean(),
}).round(2)
print(summary.to_string())


# ---------------------------------------------------------------------------
# Step 2: fit the linear regression on the training cars
# ---------------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def summary_text(fit) -> str:
    """statsmodels summary without its Date/Time lines, so saved files only change when the model does."""
    # Date/Time sit in the left half (first 39 chars) of lines that also carry Prob(F) and Log-Likelihood
    return "\n".join(l[:5].ljust(39) + l[39:] if l.startswith(("Date:", "Time:")) else l
                     for l in fit.summary().as_text().splitlines())


def fit_ols(data: pd.DataFrame, features: list[str]):
    X = sm.add_constant(data[features])
    return sm.OLS(data[TARGET], X).fit()


full = fit_ols(train, FEATURES)
(OUT / "model_full_summary.txt").write_text(summary_text(full))

print("\nSTEP 2: REGRESSION, ALL FIVE FEATURES (training cars)")
print(full.summary())

X_train = sm.add_constant(train[FEATURES])
vif = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(1, X_train.shape[1])],
    index=FEATURES, name="VIF").round(2)
print("\nVariance inflation factors (above ~5 = feature overlaps heavily with others):")
print(vif.to_string())
print("\nCorrelations among features (training cars):")
print(train[FEATURES].corr().round(2).to_string())

# Num_Owners has p = 0.87: once mileage and age are known, owner count adds nothing
# (it is strongly correlated with both). Refit without it and compare adjusted R².
V1_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]
v1 = fit_ols(train, V1_FEATURES)
(OUT / "model_v1_summary.txt").write_text(summary_text(v1))

print("\nREGRESSION WITHOUT Num_Owners (training cars)")
print(v1.summary())

comparison = pd.DataFrame({
    "features": [len(FEATURES), len(V1_FEATURES)],
    "R2": [full.rsquared, v1.rsquared],
    "adj_R2": [full.rsquared_adj, v1.rsquared_adj],
}, index=["all five", "without Num_Owners (v1)"]).round(4)
print("\nAdjusted R² comparison (training cars):")
print(comparison.to_string())
comparison.to_csv(OUT / "model_comparison.csv")

coef_table = pd.DataFrame({"coef": v1.params, "p_value": v1.pvalues,
                           "ci_low": v1.conf_int()[0], "ci_high": v1.conf_int()[1]}).round(4)
coef_table.to_csv(OUT / "v1_coefficients.csv")
model = v1  # the version handed to the dealer; used in every step below


# ---------------------------------------------------------------------------
# Step 3: baseline vs v1 on the test cars (never seen during fitting)
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

baseline_price = train[TARGET].mean()  # baseline: one price (training average) for every car
actual = test[TARGET]
pred_baseline = pd.Series(baseline_price, index=test.index)
pred_v1 = model.predict(sm.add_constant(test[V1_FEATURES], has_constant="add"))


def fit_stats(y, yhat):
    return {"R2": r2_score(y, yhat), "RMSE": float(np.sqrt(mean_squared_error(y, yhat)))}


test_fit = pd.DataFrame({"Baseline (average)": fit_stats(actual, pred_baseline),
                         "v1 (regression)": fit_stats(actual, pred_v1)}).T
test_fit["R2"] = test_fit["R2"].round(3)
test_fit["RMSE"] = test_fit["RMSE"].round(0)
test_fit.to_csv(OUT / "test_fit.csv")

print("\nSTEP 3: FIT ON THE 72 TEST CARS")
print(f"  Baseline prediction for every car = training average ${baseline_price:,.0f}")
print(test_fit.to_string())
print(f"  Typical error falls from ${test_fit.loc['Baseline (average)', 'RMSE']:,.0f} "
      f"to ${test_fit.loc['v1 (regression)', 'RMSE']:,.0f}")

pd.DataFrame({"actual": actual, "pred_baseline": pred_baseline.round(0),
              "pred_v1": pred_v1.round(0)}).to_csv(OUT / "test_predictions.csv", index_label="row")

# Chart: predicted vs actual, one panel per method, shared axes
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
lo, hi = 0, 50_000
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
panels = [(axes[0], pred_baseline, ORANGE, "Baseline: average price for every car", "Baseline (average)"),
          (axes[1], pred_v1, BLUE, "v1: regression model", "v1 (regression)")]
for ax, pred, color, title, key in panels:
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(actual, pred, s=46, color=color, edgecolor="white", linewidth=1.2, alpha=0.9, zorder=2)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold")
    ax.text(0.03, 0.95, f"R² = {test_fit.loc[key, 'R2']:.2f}\nTypical error = ${test_fit.loc[key, 'RMSE']:,.0f}",
            transform=ax.transAxes, va="top", fontsize=11, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_xlabel("Actual sale price")
axes[0].set_ylabel("Predicted sale price")
axes[1].text(44_000, 46_500, "perfect\nprediction", fontsize=9, color=MUTED, ha="right")
fig.suptitle("72 test cars the model never saw: predicted vs actual sale price",
             x=0.06, ha="left", fontsize=13.5, color=INK)
fig.tight_layout()
fig.savefig(OUT / "pred_vs_actual_test.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
