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


def get_trend_detail(df: pd.DataFrame) -> dict:
    """Return 4-component Trend vs SMA20 analysis dict for hover tooltip and signal scoring."""
    if len(df) < 20 or "SMA_20" not in df.columns:
        return {}
    sma20_series = df["SMA_20"].dropna()
    if len(sma20_series) < 11:
        return {}

    last = df.iloc[-1]
    price  = float(last["Close"])
    sma20  = float(last["SMA_20"])

    # 1. SMA20 slope — compare now vs 10 bars ago
    sma20_10ago = float(sma20_series.iloc[-11])
    slope_pct = (sma20 - sma20_10ago) / sma20_10ago * 100 if sma20_10ago else 0.0
    if slope_pct > 0.3:
        slope_text  = f"SMA20 ชันขึ้น (+{slope_pct:.2f}%) · แนวโน้มขาขึ้น · มองหาจังหวะซื้อ"
        slope_color = "#26a69a"; slope_dir = "rising";  slope_score = 1
    elif slope_pct < -0.3:
        slope_text  = f"SMA20 ชันลง ({slope_pct:.2f}%) · แนวโน้มขาลง · ระวังการซื้อสวนเทรนด์"
        slope_color = "#ef5350"; slope_dir = "falling"; slope_score = -1
    else:
        slope_text  = f"SMA20 แบน ({slope_pct:+.2f}%) · Sideway · สัญญาณอาจหลอก"
        slope_color = "#ffa726"; slope_dir = "flat";    slope_score = 0

    # 2. Price position vs SMA20
    dist_pct    = (price - sma20) / sma20 * 100 if sma20 else 0.0
    above_sma20 = price > sma20
    if above_sma20:
        pos_text  = f"ราคาอยู่เหนือ SMA20 (+{dist_pct:.1f}%) · ฝั่งซื้อยังคุมเกม · แนวโน้มยังแข็งแรง"
        pos_color = "#26a69a"; pos_score = 0.5
    else:
        pos_text  = f"ราคาอยู่ต่ำกว่า SMA20 ({dist_pct:.1f}%) · ฝั่งขายเริ่มได้เปรียบ"
        pos_color = "#ef5350"; pos_score = -0.5

    # 3. Pullback to SMA20 — check if price recently touched within 2% of SMA20
    recent_touch = False
    for i in range(-5, -1):
        try:
            _p = float(df["Close"].iloc[i]); _s = float(df["SMA_20"].iloc[i])
            if _s and abs((_p - _s) / _s * 100) < 2.0:
                recent_touch = True; break
        except Exception:
            pass

    dist_abs = abs(dist_pct)
    if slope_dir == "rising" and dist_abs < 2.0 and above_sma20:
        pb_text  = f"Pullback zone · ราคาใกล้ SMA20 ({dist_pct:+.1f}%) · จังหวะเข้าซื้อ Low Risk"
        pb_color = "#26a69a"; pb_score = 1
    elif slope_dir == "rising" and recent_touch and above_sma20:
        pb_text  = "เพิ่งเด้งจาก SMA20 · ยืนยันแนวรับ · สัญญาณซื้อที่ดี"
        pb_color = "#26a69a"; pb_score = 1
    elif slope_dir == "rising" and dist_pct > 5:
        pb_text  = f"ราคาวิ่งเร็วเกิน SMA20 (+{dist_pct:.1f}%) · มีโอกาสพักฐาน · รอย่อตัวก่อนเข้า"
        pb_color = "#ffa726"; pb_score = 0
    elif slope_dir == "rising":
        pb_text  = f"ห่าง SMA20 ปานกลาง (+{dist_pct:.1f}%) · รอดูการย่อตัว"
        pb_color = "#888";    pb_score = 0
    elif slope_dir == "flat":
        pb_text  = "SMA20 แบน · จังหวะ Pullback ไม่ชัดเจน"
        pb_color = "#888";    pb_score = 0
    elif not above_sma20:
        pb_text  = f"ราคาต่ำกว่า SMA20 ({dist_pct:.1f}%) · ยังไม่มีจังหวะ Pullback"
        pb_color = "#ef5350"; pb_score = -0.5
    else:
        pb_text  = "SMA20 ชันลง · ระวังการซื้อสวนเทรนด์"
        pb_color = "#ef5350"; pb_score = -0.5

    # 4. Distance from SMA20
    if above_sma20:
        if dist_pct > 5:
            dist_text  = f"ห่าง SMA20 มาก (+{dist_pct:.1f}%) · วิ่งเร็วเกิน · อาจพักหรือแกว่งออกข้าง"
            dist_color = "#ffa726"; dist_score = -0.5
        else:
            dist_text  = f"ราคาเกาะ SMA20 (+{dist_pct:.1f}%) · แนวโน้มขึ้นต่อเนื่อง"
            dist_color = "#26a69a"; dist_score = 0.5
    else:
        if dist_pct < -5:
            dist_text  = f"ราคาต่ำกว่า SMA20 มาก ({dist_pct:.1f}%) · แนวโน้มอ่อนแอ"
            dist_color = "#ef5350"; dist_score = -1
        else:
            dist_text  = f"ราคาต่ำกว่า SMA20 ({dist_pct:.1f}%) · SMA20 กลายเป็นแนวต้าน"
            dist_color = "#ef5350"; dist_score = -0.5

    # Composite score (range −3 to +3)
    _s = slope_score + pos_score + pb_score + dist_score
    if   _s >= 1.5:  sig_label, sig_color = "Strong Bull", "#00c853"
    elif _s >= 0.5:  sig_label, sig_color = "Bullish",     "#26a69a"
    elif _s >= -0.5: sig_label, sig_color = "Neutral",     "#888888"
    elif _s >= -1.5: sig_label, sig_color = "Bearish",     "#ef5350"
    else:            sig_label, sig_color = "Strong Bear",  "#b71c1c"

    return {
        "price": round(price, 4), "sma20": round(sma20, 4), "dist_pct": round(dist_pct, 2),
        "above_sma20": above_sma20, "slope_dir": slope_dir,
        "slope_text": slope_text, "slope_color": slope_color,
        "pos_text":   pos_text,   "pos_color":   pos_color,
        "pb_text":    pb_text,    "pb_color":    pb_color,
        "dist_text":  dist_text,  "dist_color":  dist_color,
        "score": round(_s, 2), "label": sig_label, "sig_color": sig_color,
    }


def get_bb_detail(df: pd.DataFrame) -> dict:
    """Return 4-component Bollinger Band analysis dict for hover tooltip and signal scoring."""
    required = ["BB_Upper", "BB_Lower", "BB_Mid", "BB_Pct", "Close"]
    if len(df) < 20 or not all(c in df.columns for c in required):
        return {}
    bp_series = df["BB_Pct"].dropna()
    if len(bp_series) < 5:
        return {}

    last   = df.iloc[-1]
    upper  = float(last["BB_Upper"]); lower = float(last["BB_Lower"])
    mid    = float(last["BB_Mid"]);   bb_pct = float(last["BB_Pct"])

    # Band width as % of midline
    width_now = (upper - lower) / mid * 100 if mid else 0
    n_hist    = min(len(df) - 1, 20)
    _bw = ((df["BB_Upper"].iloc[-(n_hist+1):-1] - df["BB_Lower"].iloc[-(n_hist+1):-1])
           / df["BB_Mid"].iloc[-(n_hist+1):-1] * 100).dropna()
    width_avg = float(_bw.mean()) if len(_bw) > 0 else width_now
    width_std = float(_bw.std())  if len(_bw) > 1 else 0

    # RSI for mean-reversion check
    rsi_val = None
    if "RSI" in df.columns:
        _r = df["RSI"].dropna()
        if len(_r) >= 1:
            rsi_val = float(_r.iloc[-1])

    # ── 1. Band Width / Squeeze ──────────────────────────────────────────────
    is_squeeze     = (width_std > 0) and (width_now < width_avg - width_std)
    width_chg_pct  = (width_now - width_avg) / width_avg * 100 if width_avg else 0

    if is_squeeze:
        bw_text  = f"Squeeze · Band แคบมาก ({width_now:.1f}% vs avg {width_avg:.1f}%) · สะสมพลัง รอ Breakout"
        bw_color = "#ffa726"; bw_score = 0; bw_state = "squeeze"
    elif width_chg_pct > 15:
        bw_text  = f"Band ขยาย (+{width_chg_pct:.0f}%) · ความผันผวนเพิ่ม · กำลังเกิด Trend"
        bw_color = "#26a69a" if bb_pct >= 0.5 else "#ef5350"
        bw_score = 0.5 if bb_pct >= 0.5 else -0.5; bw_state = "expanding"
    elif width_chg_pct < -15:
        bw_text  = f"Band หด ({width_chg_pct:.0f}%) · ความผันผวนลด · ตลาดพักฐาน"
        bw_color = "#888"; bw_score = 0; bw_state = "contracting"
    else:
        bw_text  = f"Band ทรงตัว ({width_chg_pct:+.0f}%) · ความผันผวนปกติ"
        bw_color = "#888"; bw_score = 0; bw_state = "normal"

    # ── 2. Price Zone ────────────────────────────────────────────────────────
    bp_disp = round(bb_pct * 100, 1)
    if bb_pct > 1.0:
        zone_text  = f"เหนือ Upper Band (BB% {bp_disp:.0f}%) · ราคาทะลุขึ้น · Breakout"
        zone_color = "#26a69a"; zone_score = 0.5; zone = "above_upper"
    elif bb_pct >= 0.8:
        zone_text  = f"โซนบน (BB% {bp_disp}%) · ใกล้ Upper Band · แรงซื้อเด่น · Momentum บวก"
        zone_color = "#26a69a"; zone_score = 0.5; zone = "upper"
    elif bb_pct >= 0.5:
        zone_text  = f"ครึ่งบน (BB% {bp_disp}%) · เหนือ Middle Band · ฝั่งซื้อได้เปรียบ"
        zone_color = "#66bb6a"; zone_score = 0.25; zone = "mid_upper"
    elif bb_pct >= 0.2:
        zone_text  = f"ครึ่งล่าง (BB% {bp_disp}%) · ต่ำกว่า Middle Band · ฝั่งขายได้เปรียบ"
        zone_color = "#ffa726"; zone_score = -0.25; zone = "mid_lower"
    elif bb_pct >= 0.0:
        zone_text  = f"โซนล่าง (BB% {bp_disp}%) · ใกล้ Lower Band · แรงขายเด่น · Momentum ลบ"
        zone_color = "#ef5350"; zone_score = -0.5; zone = "lower"
    else:
        zone_text  = f"ต่ำกว่า Lower Band (BB% {bp_disp:.0f}%) · ราคาทะลุลง · Breakdown"
        zone_color = "#ef5350"; zone_score = -0.5; zone = "below_lower"

    # ── 3. Band Walk (last 5 bars) ───────────────────────────────────────────
    recent_pct      = bp_series.iloc[-5:].values
    near_upper_cnt  = sum(1 for p in recent_pct if p >= 0.7)
    near_lower_cnt  = sum(1 for p in recent_pct if p <= 0.3)

    if near_upper_cnt >= 3:
        walk_text  = f"Bullish Band Walk · เกาะ Upper Band {near_upper_cnt}/5 แท่ง · Uptrend แข็งแรงมาก"
        walk_color = "#26a69a"; walk_score = 1; walk_state = "bullish"
    elif near_lower_cnt >= 3:
        walk_text  = f"Bearish Band Walk · เกาะ Lower Band {near_lower_cnt}/5 แท่ง · Downtrend แข็งแรง"
        walk_color = "#ef5350"; walk_score = -1; walk_state = "bearish"
    else:
        walk_text  = "ไม่มี Band Walk · ราคาเคลื่อนไหวภายใน Band ปกติ"
        walk_color = "#888"; walk_score = 0; walk_state = "none"

    # ── 4. Breakout / Mean Reversion Assessment ──────────────────────────────
    mean_rev_risk = (bb_pct > 0.85 and bw_state == "contracting"
                     and (rsi_val is None or rsi_val > 75))

    if bb_pct > 1.0:
        if bw_state == "expanding":
            ass_text  = "Breakout แข็งแรง · Band ขยาย · โอกาสเริ่ม Trend ใหม่"
            ass_color = "#26a69a"; ass_score = 0.5
        else:
            ass_text  = "ระวัง Breakout หลอก · Band ไม่ขยาย · อาจกลับเข้า Band"
            ass_color = "#ffa726"; ass_score = -0.25
    elif bb_pct < 0.0:
        if bw_state == "expanding":
            ass_text  = "Breakdown แข็งแรง · Band ขยาย · โอกาสขาลงต่อ"
            ass_color = "#ef5350"; ass_score = -0.5
        else:
            ass_text  = "ระวัง Breakdown หลอก · Band ไม่ขยาย · อาจรีบาวด์"
            ass_color = "#ffa726"; ass_score = 0.25
    elif is_squeeze:
        ass_text  = "Squeeze · รอทิศทาง Breakout · ยังไม่ชัดเจน"
        ass_color = "#ffa726"; ass_score = 0
    elif mean_rev_risk:
        _r_str = f" (RSI {rsi_val:.0f})" if rsi_val else ""
        ass_text  = f"Mean Reversion เสี่ยง · ราคาสูง+Band หด{_r_str} · อาจย่อกลับ SMA20"
        ass_color = "#ffa726"; ass_score = -0.25
    elif walk_state == "bullish" and bw_state == "expanding":
        ass_text  = "Bullish Momentum ต่อเนื่อง · Band Walk+Band ขยาย · แรงซื้อแข็งแรง"
        ass_color = "#26a69a"; ass_score = 0.5
    elif walk_state == "bearish" and bw_state == "expanding":
        ass_text  = "Bearish Momentum ต่อเนื่อง · Band Walk+Band ขยาย · แรงขายแข็งแรง"
        ass_color = "#ef5350"; ass_score = -0.5
    else:
        ass_text  = "ไม่มีสัญญาณ Breakout ชัดเจน · เคลื่อนไหวปกติภายใน Band"
        ass_color = "#888"; ass_score = 0

    # Composite score (range approximately −3 to +3)
    _s = bw_score + zone_score + walk_score + ass_score
    if   _s >= 1.5:  sig_label, sig_color = "Strong Bull", "#00c853"
    elif _s >= 0.5:  sig_label, sig_color = "Bullish",     "#26a69a"
    elif _s >= -0.5: sig_label, sig_color = "Neutral",     "#888888"
    elif _s >= -1.5: sig_label, sig_color = "Bearish",     "#ef5350"
    else:            sig_label, sig_color = "Strong Bear",  "#b71c1c"

    return {
        "bb_pct": bp_disp, "upper": round(upper, 4), "mid": round(mid, 4), "lower": round(lower, 4),
        "bw_state": bw_state, "bw_text": bw_text, "bw_color": bw_color,
        "zone": zone,         "zone_text": zone_text, "zone_color": zone_color,
        "walk_state": walk_state, "walk_text": walk_text, "walk_color": walk_color,
        "ass_text": ass_text, "ass_color": ass_color,
        "score": round(_s, 2), "label": sig_label, "sig_color": sig_color,
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
    _td = get_trend_detail(df)
    if _td:
        trend = (_td["label"], _td["sig_color"])
    else:
        trend = ("Uptrend", "#26a69a") if (not pd.isna(sma20) and price > sma20) else ("Downtrend", "#ef5350")

    bb_pct = last.get("BB_Pct", 0.5)
    _bd = get_bb_detail(df)
    if _bd:
        bb_pos = (_bd["label"], _bd["sig_color"])
    elif pd.isna(bb_pct):
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

    # SMA Trend: bull labels = buy (+1), bear labels = sell (-1)
    trend_label = signals.get("Trend", ("N/A", "N/A", "gray"))[1]
    if trend_label in ("Strong Bull", "Bullish"):
        score += 1; votes.append(("SMA Trend", "Buy", "#26a69a"))
    elif trend_label in ("Strong Bear", "Bearish"):
        score -= 1; votes.append(("SMA Trend", "Sell", "#ef5350"))
    else:
        votes.append(("SMA Trend", "Neutral", "#777"))

    # Bollinger Bands: bull labels = buy (+1), bear labels = sell (-1)
    bb_label = signals.get("BB", ("N/A", "N/A", "gray"))[1]
    if bb_label in ("Strong Bull", "Bullish"):
        score += 1; votes.append(("Bollinger", "Buy", "#26a69a"))
    elif bb_label in ("Strong Bear", "Bearish"):
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
