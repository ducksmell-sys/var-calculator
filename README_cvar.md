# CVaR (Conditional VaR / Expected Shortfall) Calculator

A standalone Python script for calculating CVaR (also known as Expected Shortfall),
a risk measure that extends VaR by estimating the average loss in the worst-case scenarios
beyond the VaR threshold.

## Why CVaR?

VaR only tells you the loss threshold at a given confidence level (e.g. "there's a 5% chance
of losing more than $X"), but says nothing about how severe that loss could actually be.
CVaR answers that: **the average loss, given that the loss already exceeds VaR.**

Since the 2008 financial crisis, regulators (e.g. Basel III) have increasingly required
Expected Shortfall alongside or instead of VaR for exactly this reason.

## Methodologies

| Method | Description |
|---|---|
| Parametric | Closed-form Expected Shortfall under a normal distribution assumption |
| Historical | Averages the actual historical returns that fall beyond the VaR threshold |
| Monte Carlo | Simulates a large number of normal-distribution scenarios and averages the tail |

## Usage

```bash
pip install numpy
python cvar_calculator.py
```

Import the functions directly to use with your own data:

```python
from cvar_calculator import parametric_cvar, historical_cvar, monte_carlo_cvar

parametric_cvar(returns, confidence=0.95, portfolio_value=1_000_000)
historical_cvar(returns, confidence=0.95, portfolio_value=1_000_000)
monte_carlo_cvar(returns, confidence=0.95, portfolio_value=1_000_000, seed=42)
```

## Sample Output

```
Confidence 90% | Parametric: 34,112.10 | Historical: 30,000.00 | Monte Carlo: 34,303.07
Confidence 95% | Parametric: 39,654.63 | Historical: 30,000.00 | Monte Carlo: 39,851.87
Confidence 99% | Parametric: 50,516.21 | Historical: 30,000.00 | Monte Carlo: 50,912.41
```

Notice CVaR is always larger than VaR at the same confidence level — it captures the
severity of the tail, not just where the tail begins.
