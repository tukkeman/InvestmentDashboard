import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, windows: list[int] = [20, 50, 200]) -> pd.DataFrame:
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list[int] = [12, 26]) -> pd.DataFrame:
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + std * num_std
    df["BB_Lower"] = mid - std * num_std
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    return df


def get_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]

    rsi = last.get("RSI", np.nan)
    if pd.isna(rsi):
        rsi_signal = ("N/A", "gray")
    elif rsi >= 70:
        rsi_signal = ("Overbought", "#ef5350")
    elif rsi <= 30:
        rsi_signal = ("Oversold", "#26a69a")
    else:
        rsi_signal = ("Neutral", "#ffa726")

    macd = last.get("MACD", 0)
    sig = last.get("MACD_Signal", 0)
    macd_signal = ("Bullish", "#26a69a") if macd > sig else ("Bearish", "#ef5350")

    price = last.get("Close", 0)
    sma20 = last.get("SMA_20", np.nan)
    sma50 = last.get("SMA_50", np.nan)
    trend = ("Uptrend", "#26a69a") if (not pd.isna(sma20) and price > sma20) else ("Downtrend", "#ef5350")

    bb_pct = last.get("BB_Pct", 0.5)
    if pd.isna(bb_pct):
        bb_pos = ("N/A", "gray")
    elif bb_pct >= 0.9:
        bb_pos = ("Near Upper", "#ef5350")
    elif bb_pct <= 0.1:
        bb_pos = ("Near Lower", "#26a69a")
    else:
        bb_pos = ("Inside Band", "#ffa726")

    return {
        "RSI": (round(rsi, 1) if not pd.isna(rsi) else "N/A", rsi_signal[0], rsi_signal[1]),
        "MACD": (round(macd, 4), macd_signal[0], macd_signal[1]),
        "Trend": (round(price, 2), trend[0], trend[1]),
        "BB": (round(bb_pct * 100, 1) if not pd.isna(bb_pct) else "N/A", bb_pos[0], bb_pos[1]),
    }


def get_recommendation(signals: dict) -> dict:
    """Aggregate indicator signals into a weighted buy/sell recommendation."""
    score = 0
    votes = []

    # RSI: oversold = buy (+1), overbought = sell (-1)
    rsi_label = signals.get("RSI", ("N/A", "N/A", "gray"))[1]
    if rsi_label == "Oversold":
        score += 1; votes.append(("RSI", "Buy", "#26a69a"))
    elif rsi_label == "Overbought":
        score -= 1; votes.append(("RSI", "Sell", "#ef5350"))
    else:
        votes.append(("RSI", "Neutral", "#777"))

    # MACD: bullish = buy (+1), bearish = sell (-1)
    macd_label = signals.get("MACD", ("N/A", "N/A", "gray"))[1]
    if macd_label == "Bullish":
        score += 1; votes.append(("MACD", "Buy", "#26a69a"))
    elif macd_label == "Bearish":
        score -= 1; votes.append(("MACD", "Sell", "#ef5350"))
    else:
        votes.append(("MACD", "Neutral", "#777"))

    # SMA Trend: uptrend = buy (+1), downtrend = sell (-1)
    trend_label = signals.get("Trend", ("N/A", "N/A", "gray"))[1]
    if trend_label == "Uptrend":
        score += 1; votes.append(("SMA Trend", "Buy", "#26a69a"))
    elif trend_label == "Downtrend":
        score -= 1; votes.append(("SMA Trend", "Sell", "#ef5350"))
    else:
        votes.append(("SMA Trend", "Neutral", "#777"))

    # Bollinger Bands: near lower = buy (+1), near upper = sell (-1)
    bb_label = signals.get("BB", ("N/A", "N/A", "gray"))[1]
    if bb_label == "Near Lower":
        score += 1; votes.append(("Bollinger", "Buy", "#26a69a"))
    elif bb_label == "Near Upper":
        score -= 1; votes.append(("Bollinger", "Sell", "#ef5350"))
    else:
        votes.append(("Bollinger", "Neutral", "#777"))

    buys     = sum(1 for _, v, _ in votes if v == "Buy")
    sells    = sum(1 for _, v, _ in votes if v == "Sell")
    neutrals = sum(1 for _, v, _ in votes if v == "Neutral")

    if score >= 3:
        label, color = "Strong Buy",  "#26a69a"
        advice = "Multiple indicators align bullishly. Consider accumulating on dips."
    elif score == 2:
        label, color = "Buy",         "#66bb6a"
        advice = "Majority of indicators lean bullish. Favorable risk/reward for entry."
    elif score == 1:
        label, color = "Weak Buy",    "#a5d6a7"
        advice = "Slight bullish bias. Wait for additional confirmation before adding exposure."
    elif score == -1:
        label, color = "Weak Sell",   "#ef9a9a"
        advice = "Slight bearish bias. Consider tightening stops or trimming position."
    elif score == -2:
        label, color = "Sell",        "#ef5350"
        advice = "Majority of indicators lean bearish. Reduce exposure or stay cautious."
    elif score <= -3:
        label, color = "Strong Sell", "#b71c1c"
        advice = "Strong bearish alignment across indicators. Consider exiting or hedging."
    else:
        label, color = "Neutral",     "#ffa726"
        advice = "Mixed signals with no clear directional bias. Hold and monitor closely."

    return {
        "score": score, "label": label, "color": color,
        "advice": advice, "votes": votes,
        "buys": buys, "sells": sells, "neutrals": neutrals,
    }
