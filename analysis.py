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
