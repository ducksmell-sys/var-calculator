# VaR (Value at Risk) Calculator

A Python toolkit for calculating Value at Risk (VaR) on real market data,
using three different methodologies.

## Methodologies

| Method | Description |
|---|---|
| Parametric | Assumes returns follow a normal distribution; uses mean and standard deviation |
| Historical | Uses the actual historical return distribution directly (no distribution assumption) |
| Monte Carlo | Simulates a large number of normally-distributed random scenarios |

## Tech Stack

- Python 3
- numpy — numerical computation
- pandas — time series handling
- yfinance — fetches historical/live price data from Yahoo Finance

## Files

| File | Description |
|---|---|
| `var_calculator.py` | Core VaR functions (parametric, historical, Monte Carlo) |
| `var_calculator_data.py` | Loads real data by ticker or CSV, then computes VaR |
| `var_calculator_pandas.py` | Pandas-only implementation (no numpy) |
| `var_simple.py` | Minimal single-file version — just edit the constants and run |

## Usage

```bash
pip install numpy pandas yfinance

# Fetch a ticker's data and compute VaR
python var_calculator_data.py --ticker AAPL

# Specify confidence level / portfolio value
python var_calculator_data.py --ticker 005930.KS --confidence 0.99 --portfolio-value 5000000

# Use a local CSV file instead
python var_calculator_data.py --csv prices.csv --price-col Close
```

## Sample Output

```
Data range: 2025-09-22 ~ 2026-09-18 (250 trading days)
Mean daily return: 0.1395% | Daily volatility (std dev): 1.5687%

1-Day VaR at 95% confidence, $1,000,000 portfolio
  Parametric   : $24,408.12
  Historical   : $20,101.53
  Monte Carlo  : $24,619.05
```
