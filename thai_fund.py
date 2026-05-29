import json
import os
import pandas as pd
from io import StringIO

NAV_FILE = os.path.join(os.path.dirname(__file__), "thai_nav_data.json")


def load_all() -> dict:
    if os.path.exists(NAV_FILE):
        with open(NAV_FILE) as f:
            return json.load(f)
    return {}


def _save(data: dict):
    with open(NAV_FILE, "w") as f:
        json.dump(data, f, indent=2)


def has_nav_data(ticker: str) -> bool:
    data = load_all()
    return bool(data.get(ticker))


def was_fetched_today(ticker: str) -> bool:
    """True if we already fetched this ticker's data today."""
    data = load_all()
    meta = data.get("_meta", {})
    last = meta.get(ticker, {}).get("last_fetched")
    if not last:
        return False
    from datetime import date
    return last == str(date.today())


def mark_fetched_today(ticker: str):
    """Record that we fetched this ticker today."""
    from datetime import date
    data = load_all()
    meta = data.setdefault("_meta", {})
    meta.setdefault(ticker, {})["last_fetched"] = str(date.today())
    _save(data)


def add_entry(ticker: str, date: str, nav: float):
    data = load_all()
    entries = data.get(ticker, [])
    entries = [e for e in entries if e["date"] != date]
    entries.append({"date": date, "nav": round(nav, 6)})
    entries.sort(key=lambda x: x["date"])
    data[ticker] = entries
    _save(data)


def delete_entry(ticker: str, date: str):
    data = load_all()
    data[ticker] = [e for e in data.get(ticker, []) if e["date"] != date]
    _save(data)


def import_csv(ticker: str, content: str) -> tuple[int, str]:
    """
    Parse CSV with flexible column detection.
    Returns (imported_count, error_message).
    Expected columns: Date + NAV (any header containing those keywords).
    """
    try:
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return 0, f"Could not parse CSV: {e}"

    if len(df.columns) < 2:
        return 0, "CSV must have at least 2 columns (Date, NAV)."

    # Flexible column detection
    date_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["date", "วัน", "ว/ด/ป", "effectivedate"])),
        df.columns[0],
    )
    nav_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["nav", "มูลค่า", "net asset", "value"])),
        df.columns[1],
    )

    data = load_all()
    entries = {e["date"]: e["nav"] for e in data.get(ticker, [])}
    count = 0
    for _, row in df.iterrows():
        try:
            date = str(pd.to_datetime(row[date_col]).date())
            nav = float(str(row[nav_col]).replace(",", ""))
            if nav > 0:
                entries[date] = round(nav, 6)
                count += 1
        except Exception:
            continue

    data[ticker] = sorted(
        [{"date": d, "nav": v} for d, v in entries.items()],
        key=lambda x: x["date"],
    )
    _save(data)
    return count, ""


def get_df(ticker: str) -> pd.DataFrame:
    """Return OHLCV-style DataFrame using NAV as Close/Open/High/Low."""
    entries = load_all().get(ticker, [])
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"nav": "Close"})
    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    df["Volume"] = 0
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_summary(ticker: str) -> dict:
    df = get_df(ticker)
    if df.empty:
        return {}
    curr = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr
    change_pct = (curr - prev) / prev * 100 if prev else 0
    return {
        "ticker": ticker,
        "name": ticker,
        "price": round(curr, 4),
        "change_pct": round(change_pct, 4),
        "volume": 0,
        "high_52w": round(df["Close"].max(), 4),
        "low_52w": round(df["Close"].min(), 4),
        "currency": "THB",
        "source": "manual_nav",
    }
