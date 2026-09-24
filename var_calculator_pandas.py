import pandas as pd

# numpy 없이 pandas의 mean(), std(), quantile()만 사용

Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def parametric_var(returns, confidence=0.95, portfolio_value=1.0):
    s = pd.Series(returns)
    mean = s.mean()
    std = s.std()  # pandas 기본값이 ddof=1 (numpy와 동일)
    z = Z_SCORES[confidence]
    return (z * std - mean) * portfolio_value


def historical_var(returns, confidence=0.95, portfolio_value=1.0):
    s = pd.Series(returns)
    percentile_value = s.quantile(1 - confidence)  # np.percentile과 동일한 역할
    return -percentile_value * portfolio_value


if __name__ == "__main__":
    sample_returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.008, -0.01, 0.012, -0.025]
    portfolio_value = 1_000_000

    for confidence in (0.90, 0.95, 0.99):
        p_var = parametric_var(sample_returns, confidence, portfolio_value)
        h_var = historical_var(sample_returns, confidence, portfolio_value)
        print(f"신뢰수준 {confidence:.0%} | 모수적: {p_var:,.2f} | 역사적: {h_var:,.2f}")
