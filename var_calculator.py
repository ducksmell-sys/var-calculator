"""Value at Risk (VaR) calculator: parametric, historical, Monte Carlo methods (numpy 기반)."""

import numpy as np

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def parametric_var(returns, confidence=0.95, portfolio_value=1.0):
    """분산-공분산(모수적) 방법으로 VaR 계산."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    returns = np.asarray(returns)
    mean = returns.mean()
    std = returns.std(ddof=1)
    z = Z_SCORES[confidence]
    return (z * std - mean) * portfolio_value


def historical_var(returns, confidence=0.95, portfolio_value=1.0):
    """과거 수익률 분포를 그대로 사용하는 역사적 시뮬레이션 방법."""
    returns = np.asarray(returns)
    percentile = np.percentile(returns, (1 - confidence) * 100)
    return -percentile * portfolio_value


def monte_carlo_var(returns, confidence=0.95, portfolio_value=1.0, simulations=100_000, seed=None):
    """평균/표준편차를 정규분포로 가정하고 난수를 생성해 시뮬레이션하는 방법."""
    returns = np.asarray(returns)
    mean = returns.mean()
    std = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mean, std, simulations)
    percentile = np.percentile(simulated_returns, (1 - confidence) * 100)
    return -percentile * portfolio_value


if __name__ == "__main__":
    # 예시: 일별 수익률 샘플 데이터
    sample_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.008, -0.01, 0.012, -0.025])
    portfolio_value = 1_000_000  # 포트폴리오 평가액

    for confidence in (0.90, 0.95, 0.99):
        p_var = parametric_var(sample_returns, confidence, portfolio_value)
        h_var = historical_var(sample_returns, confidence, portfolio_value)
        m_var = monte_carlo_var(sample_returns, confidence, portfolio_value, seed=42)
        print(
            f"신뢰수준 {confidence:.0%} | 모수적: {p_var:,.2f} | "
            f"역사적: {h_var:,.2f} | 몬테카를로: {m_var:,.2f}"
        )
