import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import thai_fund
import scbam_fetcher
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
    df = fetch_price_data(ticker, "1mo")
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


def _fmt_aum(v: float) -> str:
    if v is None:
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.2f} B"
    if v >= 1e6:
        return f"${v/1e6:.2f} M"
    return f"${v:,.0f}"


def _ts_to_date(v) -> str:
    try:
        return datetime.fromtimestamp(int(v)).strftime("%d %b %Y")
    except Exception:
        return "—"


def _safe_float(info: dict, key: str) -> float | None:
    v = info.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_row(row, key: str) -> float | None:
    import math
    try:
        v = row[key]
        if v is None:
            return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (KeyError, TypeError, ValueError):
        return None


@st.cache_data(ttl=300)
def fetch_ticker_fundamentals(ticker: str) -> dict:
    """Return fundamental data for a ticker as a flat dict.
    Numeric fields are float | None (None = unavailable).
    String fields fall back to '—'.
    """
    base = {
        "ticker": ticker, "name": ticker, "quote_type": "UNKNOWN", "currency": "—",
        "fund_family": "—", "inception_date": "—", "category": "—",
        "sector": "—", "industry": "—", "country": "—", "long_summary": "",
        "trailing_pe": None, "forward_pe": None, "price_to_book": None, "eps": None,
        "div_yield_pct": None, "div_rate": None, "payout_ratio": None, "ex_div_date": "—",
        "aum_or_mktcap": None, "aum_label": "—", "aum_fmt": "—",
        "beta": None, "expense_ratio": None, "change_52w_pct": None,
        "is_thai_fund": False,
    }

    # ── SCBAM / Thai mutual fund branch ──────────────────────────────────────
    if scbam_fetcher.is_scbam_fund(ticker):
        base["is_thai_fund"] = True
        base["fund_family"] = "SCBAM"
        base["category"] = "Thai Mutual Fund"
        base["currency"] = "THB"
        try:
            df = thai_fund.get_df(ticker)
            if not df.empty:
                base["name"] = ticker
                closes = df["Close"].dropna()
                if len(closes) >= 2:
                    ref_idx = max(0, len(closes) - 252)
                    base["change_52w_pct"] = (closes.iloc[-1] / closes.iloc[ref_idx] - 1) * 100
        except Exception:
            pass
        return base

    # ── Thai SET stock (.BK) branch — use thaifin for fundamentals ───────────
    if ticker.upper().endswith('.BK'):
        symbol = ticker.upper().replace('.BK', '')
        # Profile: still use yfinance (name, sector, industry, country)
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        base['quote_type'] = info.get('quoteType', 'EQUITY')
        base['name'] = info.get('shortName') or info.get('longName') or ticker
        base['currency'] = info.get('currency', 'THB') or 'THB'
        base['country'] = info.get('country', 'Thailand') or 'Thailand'
        base['sector'] = info.get('sector', '—') or '—'
        base['industry'] = info.get('industry', '—') or '—'
        base['long_summary'] = info.get('longBusinessSummary', '') or ''
        base['beta'] = _safe_float(info, 'beta')
        _pct = _safe_float(info, 'fiftyTwoWeekChangePercent')
        if _pct is not None:
            base['change_52w_pct'] = _pct               # already in %
        else:
            _dec = _safe_float(info, '52WeekChange')
            if _dec is not None:
                base['change_52w_pct'] = _dec * 100     # decimal → %
        mktcap = _safe_float(info, 'marketCap')
        if mktcap:
            base['aum_or_mktcap'] = mktcap
            base['aum_label'] = 'Market Cap'
            base['aum_fmt'] = _fmt_aum(mktcap)
        # Fundamentals: thaifin (latest fiscal year row)
        try:
            from thaifin import Stock as ThaiStock
            df_y = ThaiStock(symbol).yearly_dataframe
            if not df_y.empty:
                row = df_y.iloc[-1]
                base['trailing_pe']    = _safe_row(row, 'price_earning_ratio')
                base['price_to_book']  = _safe_row(row, 'price_book_value')
                base['eps']            = _safe_row(row, 'earning_per_share')
                dv = _safe_row(row, 'dividend_yield')
                if dv is not None:
                    base['div_yield_pct'] = dv  # thaifin returns as % already
                # Dividend rate = yield% / 100 * close price
                close = _safe_row(row, 'close')
                if dv is not None and close:
                    base['div_rate'] = round(dv / 100 * close, 4)
                # Payout ratio = div_per_share / EPS * 100
                if base['div_rate'] and base['eps'] and base['eps'] > 0:
                    base['payout_ratio'] = round(base['div_rate'] / base['eps'] * 100, 2)
                # Supplement mkt_cap from thaifin if yfinance didn't have it
                if base['aum_or_mktcap'] is None:
                    mktcap_t = _safe_row(row, 'mkt_cap')
                    if mktcap_t:
                        base['aum_or_mktcap'] = mktcap_t
                        base['aum_label'] = 'Market Cap'
                        base['aum_fmt'] = _fmt_aum(mktcap_t)
        except Exception:
            pass
        return base

    # ── yfinance branch ───────────────────────────────────────────────────────
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return base

    qt = info.get("quoteType", "UNKNOWN")
    base["quote_type"] = qt
    base["name"] = info.get("shortName") or info.get("longName") or ticker
    base["currency"] = info.get("currency", "—") or "—"
    base["country"] = info.get("country", "—") or "—"
    base["sector"] = info.get("sector", "—") or "—"
    base["industry"] = info.get("industry", "—") or "—"
    base["long_summary"] = info.get("longBusinessSummary", "") or ""

    # Fund-specific fields
    if info.get("fundFamily"):
        base["fund_family"] = info["fundFamily"]
    if info.get("fundInceptionDate"):
        base["inception_date"] = _ts_to_date(info["fundInceptionDate"])
    if info.get("category"):
        base["category"] = info["category"]

    # 52W change — fiftyTwoWeekChangePercent is already in %; 52WeekChange is a decimal fraction
    _pct = _safe_float(info, "fiftyTwoWeekChangePercent")
    if _pct is not None:
        base["change_52w_pct"] = _pct
    else:
        _dec = _safe_float(info, "52WeekChange")
        if _dec is not None:
            base["change_52w_pct"] = _dec * 100

    # INDEX assets have almost no fundamentals — return early
    if qt == "INDEX":
        return base

    # Valuation
    base["trailing_pe"] = _safe_float(info, "trailingPE")
    base["forward_pe"] = _safe_float(info, "forwardPE")
    base["price_to_book"] = _safe_float(info, "priceToBook")
    base["eps"] = _safe_float(info, "epsTrailingTwelveMonths") or _safe_float(info, "trailingEps")

    # Dividends
    if qt == "ETF":
        raw_yield = _safe_float(info, "yield")
        if raw_yield is not None:
            base["div_yield_pct"] = raw_yield * 100 if raw_yield < 1.0 else raw_yield
        base["div_rate"] = _safe_float(info, "trailingAnnualDividendRate")
        base["beta"] = _safe_float(info, "beta3Year")
        aum = _safe_float(info, "netAssets") or _safe_float(info, "totalAssets")
        if aum:
            base["aum_or_mktcap"] = aum
            base["aum_label"] = "AUM"
            base["aum_fmt"] = _fmt_aum(aum)
        exp = _safe_float(info, "netExpenseRatio") or _safe_float(info, "annualReportExpenseRatio")
        if exp is not None:
            base["expense_ratio"] = exp * 100 if exp < 1.0 else exp
    else:
        raw_yield = _safe_float(info, "dividendYield")
        if raw_yield is not None:
            base["div_yield_pct"] = raw_yield * 100 if raw_yield < 1.0 else raw_yield
        base["div_rate"] = _safe_float(info, "dividendRate") or _safe_float(info, "trailingAnnualDividendRate")
        base["beta"] = _safe_float(info, "beta")
        mktcap = _safe_float(info, "marketCap")
        if mktcap:
            base["aum_or_mktcap"] = mktcap
            base["aum_label"] = "Market Cap"
            base["aum_fmt"] = _fmt_aum(mktcap)

    base["payout_ratio"] = _safe_float(info, "payoutRatio")
    if info.get("exDividendDate"):
        base["ex_div_date"] = _ts_to_date(info["exDividendDate"])

    return base
