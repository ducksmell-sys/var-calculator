"""주가 데이터를 불러와 VaR을 계산하는 스크립트 (pandas + numpy).

CSV 형식 예시 (Yahoo Finance 다운로드 형식과 동일):
    Date,Close
    2024-01-02,100.0
    2024-01-03,101.5
    ...

사용법:
    python var_calculator_data.py --ticker AAPL
    python var_calculator_data.py --ticker 005930.KS --start 2023-01-01 --confidence 0.99
    python var_calculator_data.py --csv prices.csv --price-col Close --portfolio-value 5000000
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from var_calculator import historical_var, monte_carlo_var, parametric_var


def load_returns(csv_path: str, date_col: str = "Date", price_col: str = "Close") -> pd.Series:
    """CSV에서 가격을 읽어 일별 수익률(Series)로 변환."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    returns = df[price_col].pct_change().dropna()
    return returns


def load_returns_from_ticker(ticker: str, start: str = None, end: str = None) -> pd.Series:
    """yfinance로 티커의 주가를 받아와 일별 수익률(Series)로 변환."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"'{ticker}' 데이터를 받아오지 못했습니다. 티커를 확인해주세요.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # 티커를 여러 개 받은 경우 대비
        close = close.iloc[:, 0]
    returns = close.pct_change().dropna()
    returns.index.name = "Date"
    return returns


def make_sample_csv(csv_path: str, days: int = 250, seed: int = 42) -> None:
    """데모용 샘플 주가 CSV 생성 (기하 브라운 운동 근사)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    daily_returns = rng.normal(loc=0.0005, scale=0.015, size=days)
    prices = 100 * np.cumprod(1 + daily_returns)
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="주가 데이터 기반 VaR 계산기 (티커 또는 CSV)")
    parser.add_argument("--ticker", help="야후 파이낸스 티커 (예: AAPL, 005930.KS, BTC-USD)")
    parser.add_argument("--start", default=str(date.today() - timedelta(days=365)), help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD, 기본값: 오늘)")
    parser.add_argument("--csv", help="주가 CSV 파일 경로 (티커 대신 사용)")
    parser.add_argument("--date-col", default="Date", help="날짜 컬럼명")
    parser.add_argument("--price-col", default="Close", help="가격 컬럼명")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    parser.add_argument("--portfolio-value", type=float, default=1_000_000)
    args = parser.parse_args()

    if args.ticker:
        print(f"[안내] yfinance로 '{args.ticker}' 데이터를 받아옵니다 ({args.start} ~ {args.end or '오늘'}).")
        returns = load_returns_from_ticker(args.ticker, args.start, args.end)
    else:
        csv_path = args.csv or "sample_prices.csv"
        if not Path(csv_path).exists():
            print(f"[안내] '{csv_path}' 파일이 없어 데모용 샘플 CSV를 생성합니다.")
            make_sample_csv(csv_path)
        returns = load_returns(csv_path, args.date_col, args.price_col)

    print(f"데이터 기간: {returns.index.min().date()} ~ {returns.index.max().date()} "
          f"({len(returns)}개 거래일)")
    print(f"일평균 수익률: {returns.mean():.4%} | 일별 변동성(표준편차): {returns.std():.4%}\n")

    p_var = parametric_var(returns.values, args.confidence, args.portfolio_value)
    h_var = historical_var(returns.values, args.confidence, args.portfolio_value)
    m_var = monte_carlo_var(returns.values, args.confidence, args.portfolio_value, seed=42)

    print(f"신뢰수준 {args.confidence:.0%}, 포트폴리오 {args.portfolio_value:,.0f}원 기준 1일 VaR")
    print(f"  모수적(Parametric)   : {p_var:,.2f}원")
    print(f"  역사적(Historical)   : {h_var:,.2f}원")
    print(f"  몬테카를로(MonteCarlo): {m_var:,.2f}원")


if __name__ == "__main__":
    main()
