import numpy as np
import yfinance as yf

# ↓↓↓ 여기 값만 원하는 대로 바꾸면 됩니다 ↓↓↓
TICKER = "AAPL"              # 종목 코드 (예: AAPL, 005930.KS, BTC-USD)
CONFIDENCE = 0.95            # 신뢰수준 (0.90, 0.95, 0.99 중 하나)
PORTFOLIO_VALUE = 1_000_000  # 포트폴리오 평가액
# ↑↑↑ 여기까지만 수정하면 됩니다 ↑↑↑

Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}

df = yf.download(TICKER, period="1y", progress=False, auto_adjust=True)
returns = df["Close"].pct_change().dropna().values.flatten()

mean = returns.mean()
std = returns.std(ddof=1)
z = Z_SCORES[CONFIDENCE]

parametric_var = (z * std - mean) * PORTFOLIO_VALUE
historical_var = -np.percentile(returns, (1 - CONFIDENCE) * 100) * PORTFOLIO_VALUE

print(f"종목: {TICKER}")
print(f"신뢰수준 {CONFIDENCE:.0%}, 포트폴리오 {PORTFOLIO_VALUE:,.0f}원 기준 1일 VaR")
print(f"  모수적(Parametric) VaR : {parametric_var:,.2f}원")
print(f"  역사적(Historical) VaR : {historical_var:,.2f}원")
