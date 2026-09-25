"""CVaR (Conditional VaR / Expected Shortfall) calculator: parametric, historical, Monte Carlo methods."""

import numpy as np

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def parametric_cvar(returns, confidence=0.95, portfolio_value=1.0):
    """정규분포 가정하에 VaR을 넘는 손실들의 기댓값(Expected Shortfall)을 계산."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    returns = np.asarray(returns)
    mean = returns.mean()
    std = returns.std(ddof=1)
    z = Z_SCORES[confidence]
    alpha = 1 - confidence
    phi_z = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)  # 표준정규분포의 z 지점 확률밀도
    return (std * phi_z / alpha - mean) * portfolio_value


def historical_cvar(returns, confidence=0.95, portfolio_value=1.0):
    """과거 수익률 중 VaR을 넘는 손실들만 골라 그 평균을 계산."""
    returns = np.asarray(returns)
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_returns = returns[returns <= var_threshold]
    return -tail_returns.mean() * portfolio_value


def monte_carlo_cvar(returns, confidence=0.95, portfolio_value=1.0, simulations=100_000, seed=None):
    """몬테카를로로 시뮬레이션한 시나리오 중 VaR을 넘는 손실들의 평균을 계산."""
    returns = np.asarray(returns)
    mean = returns.mean()
    std = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mean, std, simulations)
    var_threshold = np.percentile(simulated_returns, (1 - confidence) * 100)
    tail_returns = simulated_returns[simulated_returns <= var_threshold]
    return -tail_returns.mean() * portfolio_value


if __name__ == "__main__":
    # 예시: 일별 수익률 샘플 데이터
    sample_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.008, -0.01, 0.012, -0.025])
    portfolio_value = 1_000_000  # 포트폴리오 평가액

    for confidence in (0.90, 0.95, 0.99):
        p_cvar = parametric_cvar(sample_returns, confidence, portfolio_value)
        h_cvar = historical_cvar(sample_returns, confidence, portfolio_value)
        m_cvar = monte_carlo_cvar(sample_returns, confidence, portfolio_value, seed=42)
        print(
            f"신뢰수준 {confidence:.0%} | 모수적: {p_cvar:,.2f} | "
            f"역사적: {h_cvar:,.2f} | 몬테카를로: {m_cvar:,.2f}"
        )
