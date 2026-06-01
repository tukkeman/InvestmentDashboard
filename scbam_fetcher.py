"""
Fetches historical NAV data for SCBAM funds from www.scbam.com/th/fund/nav-historical/

The page is a large HTML dump (all ~500 funds × date range). We use regex to extract
only the target fund's data, avoiding a full DOM parse on the 16MB+ response.
"""
import re
import requests
from datetime import datetime, timedelta


_SESSION = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        })
        # Establish session cookies
        _SESSION.get("https://www.scbam.com/th/fund/nav-historical/", timeout=15)
    return _SESSION


def _to_be(dt: datetime) -> str:
    """Convert CE datetime to Thai Buddhist Era date string dd/mm/BBBB."""
    return dt.strftime("%d/%m/") + str(dt.year + 543)


def _be_to_ce(be: str) -> str:
    """Convert BE date string dd/mm/BBBB to ISO date YYYY-MM-DD."""
    d, m, y = be.split("/")
    return f"{int(y) - 543}-{m}-{d}"


_FUND_ID_CACHE: dict[str, str] = {}

def _get_fund_id(fund_name: str) -> str | None:
    """Return the checkbox ID for a fund by scraping the filter section of the page."""
    if fund_name in _FUND_ID_CACHE:
        return _FUND_ID_CACHE[fund_name]
    try:
        session = _get_session()
        r = session.get("https://www.scbam.com/th/fund/nav-historical/", timeout=15)
        escaped = re.escape(fund_name)
        # Column header structure: <span class="newFont">FUND</span> ... value="ID"
        m = re.search(
            rf'<span class="newFont">{escaped}</span>.*?value="(\d+)"',
            r.text,
            re.DOTALL,
        )
        if m:
            _FUND_ID_CACHE[fund_name] = m.group(1)
            return m.group(1)
    except Exception:
        pass
    return None


def fetch_nav(fund_name: str, days: int = 365) -> list[dict]:
    """
    Fetch NAV history for a named SCBAM fund.

    Args:
        fund_name: Exact fund name as shown on SCBAM site, e.g. "SCBSEMI(A)".
        days: How many calendar days of history to fetch (max ~365 for speed).

    Returns:
        List of {"date": "YYYY-MM-DD", "nav": float}, sorted ascending.
        Empty list if fund not found or request fails.
    """
    end_ce = datetime.today()
    start_ce = end_ce - timedelta(days=days)

    try:
        session = _get_session()
        # Include the fund's checkbox ID so SCBAM returns it in the data rows.
        # Without this, SCBAM omits the fund from the table and the regex can
        # accidentally match a neighbouring fund's NAV column.
        fund_id = _get_fund_id(fund_name)
        post_data = {"start": _to_be(start_ce), "end": _to_be(end_ce)}
        if fund_id:
            post_data[f"checkbox{fund_id}"] = "on"
        r = session.post(
            "https://www.scbam.com/th/fund/nav-historical/",
            data=post_data,
            headers={
                "Referer": "https://www.scbam.com/th/fund/nav-historical/",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=60,
        )
        r.raise_for_status()
    except Exception:
        return []

    html = r.text

    # Escape special chars in fund name for regex (handles parentheses)
    fund_escaped = re.escape(fund_name)

    # Match only within the fund's own column div — stop before the next column starts
    # so a missing NAV never bleeds into a neighbouring fund's value.
    nav_blocks = re.finditer(
        rf'<h6>{fund_escaped}</h6>(?:(?!<div class="column).)*?<p[^>]*padding-right[^>]*>\s*(?:<img[^>]*>\s*)?([0-9]+\.[0-9]+)',
        html,
        re.DOTALL,
    )

    results = []
    for match in nav_blocks:
        nav_val = float(match.group(1).strip())
        # Find the most recent date tag before this position
        preceding = html[max(0, match.start() - 5000): match.start()]
        date_tags = re.findall(r'<p>(\d{2}/\d{2}/\d{4})</p>', preceding)
        if date_tags:
            ce_date = _be_to_ce(date_tags[-1])
            results.append({"date": ce_date, "nav": nav_val})

    # Deduplicate and sort
    seen = {}
    for rec in results:
        seen[rec["date"]] = rec["nav"]
    records = [{"date": d, "nav": v} for d, v in sorted(seen.items())]

    # Drop entries where the single-day change exceeds 30% (parse artifact, not real)
    filtered = []
    for rec in records:
        if filtered:
            prev_nav = filtered[-1]["nav"]
            if prev_nav and abs(rec["nav"] - prev_nav) / prev_nav > 0.30:
                continue
        filtered.append(rec)
    return filtered


def is_scbam_fund(fund_name: str) -> bool:
    """Heuristic: SCBAM fund names start with SCB, PVD, or known SCBAM prefixes."""
    upper = fund_name.upper()
    return any(upper.startswith(prefix) for prefix in [
        "SCB", "PVDFP", "PVDTP", "PVDTPP", "PVDFPP"
    ])
