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
