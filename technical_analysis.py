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


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 40) -> tuple:
    """Detect bullish or bearish RSI divergence in the last `lookback` bars.
    Returns (type, score): type = 'bullish' | 'bearish' | None, score = +1 | -1 | 0.
    """
    if len(df) < lookback + 5 or "RSI" not in df.columns:
        return None, 0
    prices = df["Close"].values[-lookback:]
    rsis   = df["RSI"].values[-lookback:]
    win = 5

    troughs = [i for i in range(win, len(prices) - win)
               if prices[i] == min(prices[i - win:i + win + 1])]
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if prices[t2] < prices[t1] and rsis[t2] > rsis[t1]:
            return "bullish", 1.0

    peaks = [i for i in range(win, len(prices) - win)
             if prices[i] == max(prices[i - win:i + win + 1])]
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if prices[p2] > prices[p1] and rsis[p2] < rsis[p1]:
            return "bearish", -1.0

    return None, 0


def get_rsi_detail(df: pd.DataFrame) -> dict:
    """Return all 4 RSI components as a flat dict for hover tooltip display."""
    if len(df) < 5 or "RSI" not in df.columns:
        return {}
    rsi_series = df["RSI"].dropna()
    if len(rsi_series) < 5:
        return {}

    rsi = float(rsi_series.iloc[-1])

    # 1. Level
    if rsi >= 70:
        lv_text, lv_color = f"Overbought (RSI {rsi:.1f}) · มีโอกาสพักตัว", "#ef5350"
    elif rsi >= 60:
        lv_text, lv_color = f"RSI {rsi:.1f} · โมเมนตัมขาขึ้นแรง", "#26a69a"
    elif rsi >= 50:
        lv_text, lv_color = f"RSI {rsi:.1f} · ขาขึ้นปกติ", "#66bb6a"
    elif rsi >= 40:
        lv_text, lv_color = f"RSI {rsi:.1f} · เริ่มอ่อนแรง", "#ffa726"
    elif rsi >= 30:
        lv_text, lv_color = f"RSI {rsi:.1f} · ขาลง อ่อนแรงมาก", "#ef9a9a"
    else:
        lv_text, lv_color = f"Oversold (RSI {rsi:.1f}) · มีโอกาสรีบาวด์", "#26a69a"

    # 2. Trend (last 5 bars)
    recent = rsi_series.iloc[-5:].values
    rises  = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    falls  = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    rsi_delta = recent[-1] - recent[0]
    if rises >= 3:
        tr_text = f"RSI สูงขึ้นต่อเนื่อง (+{rsi_delta:.1f}) · แรงซื้อเพิ่มขึ้น"
        tr_color = "#26a69a"
        trend_dir = "rising"
    elif falls >= 3:
        tr_text = f"RSI ลดลงต่อเนื่อง ({rsi_delta:.1f}) · แรงซื้อเริ่มอ่อน"
        tr_color = "#ef5350"
        trend_dir = "falling"
    else:
        tr_text = "RSI เคลื่อนไหวผสม · ไม่มีทิศทางชัดเจน"
        tr_color = "#888"
        trend_dir = "flat"

    # 3. 50 line
    if rsi > 50:
        l50_text  = f"RSI > 50 · ฝั่งซื้อได้เปรียบ · แนวโน้มเป็นบวก"
        l50_color = "#26a69a"
    else:
        l50_text  = f"RSI < 50 · ฝั่งขายได้เปรียบ · แนวโน้มเป็นลบ"
        l50_color = "#ef5350"

    # 4. Divergence
    div_type, div_score = detect_rsi_divergence(df)

    # Composite score (range −3 to +3)
    _s = 0
    if rsi >= 70:   _s -= 1       # overbought → bearish
    elif rsi >= 50: _s += 0.5     # bullish territory
    elif rsi >= 30: _s -= 0.5     # bearish territory
    else:           _s += 1       # oversold → bullish
    if trend_dir == "rising":    _s += 0.5
    elif trend_dir == "falling": _s -= 0.5
    _s += 0.5 if rsi > 50 else -0.5
    _s += div_score
    if   _s >= 1.5:  sig_label, sig_color = "Strong Bull", "#00c853"
    elif _s >= 0.5:  sig_label, sig_color = "Bullish",     "#26a69a"
    elif _s >= -0.5: sig_label, sig_color = "Neutral",     "#888888"
    elif _s >= -1.5: sig_label, sig_color = "Bearish",     "#ef5350"
    else:            sig_label, sig_color = "Strong Bear",  "#b71c1c"

    return {
        "rsi_value": round(rsi, 1),
        "lv_text": lv_text, "lv_color": lv_color,
        "tr_text": tr_text, "tr_color": tr_color, "trend_dir": trend_dir,
        "l50_text": l50_text, "l50_color": l50_color,
        "divergence": div_type, "div_score": div_score,
        "score": round(_s, 2), "label": sig_label, "sig_color": sig_color,
    }


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 40) -> tuple:
    """Detect bullish or bearish MACD divergence in the last `lookback` bars.
    Returns (type, score): type = 'bullish' | 'bearish' | None, score = +1 | -1 | 0.
    """
    if len(df) < lookback + 5 or "MACD" not in df.columns:
        return None, 0
    prices = df["Close"].values[-lookback:]
    macds  = df["MACD"].values[-lookback:]
    win = 5

    troughs = [i for i in range(win, len(prices) - win)
               if prices[i] == min(prices[i - win:i + win + 1])]
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if prices[t2] < prices[t1] and macds[t2] > macds[t1]:
            return "bullish", 1.0

    peaks = [i for i in range(win, len(prices) - win)
             if prices[i] == max(prices[i - win:i + win + 1])]
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if prices[p2] > prices[p1] and macds[p2] < macds[p1]:
            return "bearish", -1.0

    return None, 0


def get_macd_detail(df: pd.DataFrame) -> dict:
    """Return all 4 MACD components as a flat dict for display and signal scoring."""
    if len(df) < 3 or "MACD" not in df.columns:
        return {}
    last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    macd  = float(last["MACD"]);        sig   = float(last["MACD_Signal"]);  hist  = float(last["MACD_Hist"])
    p_mac = float(prev["MACD"]);        p_sig = float(prev["MACD_Signal"]);  p_his = float(prev["MACD_Hist"])
    pp_his = float(prev2["MACD_Hist"])

    crossed_up   = macd > sig  and p_mac <= p_sig
    crossed_down = macd < sig  and p_mac >= p_sig

    div_type, div_score = detect_macd_divergence(df)

    return {
        "macd_value":        round(macd, 4),
        "signal_value":      round(sig,  4),
        "hist_value":        round(hist, 4),
        "above_signal":      macd > sig,
        "crossed_up":        crossed_up,
        "crossed_down":      crossed_down,
        "above_zero":        macd > 0,
        "hist_increasing":   hist > p_his,
        "hist_accelerating": hist > p_his > pp_his,
        "hist_decelerating": hist < p_his < pp_his,
        "divergence":        div_type,
        "div_score":         div_score,
    }


def get_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]

    _rd = get_rsi_detail(df)
    if _rd:
        rsi = float(_rd["rsi_value"])
        rsi_signal = (_rd["label"], _rd["sig_color"])
    else:
        rsi = last.get("RSI", np.nan)
        rsi_signal = ("N/A", "gray")

    _md = get_macd_detail(df)
    if _md:
        _s  = (1 if _md["above_signal"] else -1)
        _s += (1 if _md["above_zero"]   else -1)
        _s += (0.5 if _md["hist_increasing"] else -0.5)
        _s += _md["div_score"]
        if   _s >= 2.5:  _ml, _mc = "Strong Bull", "#00c853"
        elif _s >= 1.0:  _ml, _mc = "Bullish",     "#26a69a"
        elif _s >= -0.5: _ml, _mc = "Neutral",     "#888888"
        elif _s >= -2.0: _ml, _mc = "Bearish",     "#ef5350"
        else:            _ml, _mc = "Strong Bear",  "#b71c1c"
        macd = _md["macd_value"]
        macd_signal = (_ml, _mc)
    else:
        macd = float(last.get("MACD", 0))
        macd_signal = ("Neutral", "#888888")

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
        "MACD": (macd, macd_signal[0], macd_signal[1]),
        "Trend": (round(price, 2), trend[0], trend[1]),
        "BB": (round(bb_pct * 100, 1) if not pd.isna(bb_pct) else "N/A", bb_pos[0], bb_pos[1]),
    }


def get_recommendation(signals: dict) -> dict:
    """Aggregate indicator signals into a weighted buy/sell recommendation."""
    score = 0
    votes = []

    # RSI: bull labels = buy (+1), bear labels = sell (-1)
    rsi_label = signals.get("RSI", ("N/A", "N/A", "gray"))[1]
    if rsi_label in ("Strong Bull", "Bullish"):
        score += 1; votes.append(("RSI", "Buy", "#26a69a"))
    elif rsi_label in ("Strong Bear", "Bearish"):
        score -= 1; votes.append(("RSI", "Sell", "#ef5350"))
    else:
        votes.append(("RSI", "Neutral", "#777"))

    # MACD: bull labels = buy (+1), bear labels = sell (-1)
    macd_label = signals.get("MACD", ("N/A", "N/A", "gray"))[1]
    if macd_label in ("Strong Bull", "Bullish"):
        score += 1; votes.append(("MACD", "Buy", "#26a69a"))
    elif macd_label in ("Strong Bear", "Bearish"):
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
