import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import thai_fund
from technical_analysis import add_all_indicators, get_signals, get_recommendation


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance download."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300)
def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df = _flatten(df)
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_price_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = _fetch_yfinance(ticker, period)
    if not df.empty:
        return df
    # Fallback: local Thai fund NAV data (never cached — always reads fresh JSON)
    return thai_fund.get_df(ticker)


@st.cache_data(ttl=300)
def fetch_ticker_summary(ticker: str) -> dict:
    try:
        df_recent = yf.download(ticker, period="1mo", progress=False, auto_adjust=True)
        df_recent = _flatten(df_recent).dropna(subset=["Close"])
        if not df_recent.empty:
            curr = float(df_recent["Close"].iloc[-1])
            prev = float(df_recent["Close"].iloc[-2]) if len(df_recent) >= 2 else curr
            change_pct = (curr - prev) / prev * 100 if prev else 0

            df_1y = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            df_1y = _flatten(df_1y).dropna(subset=["Close"])
            high_52w = float(df_1y["High"].max()) if not df_1y.empty else 0
            low_52w = float(df_1y["Low"].min()) if not df_1y.empty else 0

            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            return {
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or ticker,
                "price": round(curr, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(df_recent["Volume"].iloc[-1]) if "Volume" in df_recent else 0,
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "currency": info.get("currency", ""),
            }
    except Exception:
        pass
    # Fallback: local Thai fund NAV data (auto-fetched by scbam_fetcher for SCBAM funds)
    summary = thai_fund.get_summary(ticker)
    if summary:
        return summary
    return {"ticker": ticker, "error": "No data"}


@st.cache_data(ttl=300)
def fetch_ticker_signal(ticker: str) -> dict:
    """Return overall TA signal for a ticker (label, color, score)."""
    df = fetch_price_data(ticker, "6mo")
    if df.empty or len(df) < 26:
        return {"label": "N/A", "color": "#888", "score": 0}
    df = add_all_indicators(df)
    signals = get_signals(df)
    rec = get_recommendation(signals)
    return {"label": rec["label"], "color": rec["color"], "score": rec["score"]}


def fetch_watchlist_summary(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        data = fetch_ticker_summary(ticker)
        rows.append(data)
    return pd.DataFrame(rows)
