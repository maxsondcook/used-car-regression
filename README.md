# Used car prices: should the dealer buy this car?

Linear regression that predicts a used car's retail sale price, and a buy/pass rule for wholesale auctions built on it.

## Data
- `data/used_car_sales.csv`: 240 cars the dealer sold in the last year (used to fit and test the model, 70/30 split).
- `data/auction_lots.csv`: 40 cars from one recent auction (used only to test the buy rule).

## Reproduce
```bash
pip install -r requirements.txt
python analysis.py
```
All numbers and charts on the slides are written to `outputs/`.
