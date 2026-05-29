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
        r = session.post(
            "https://www.scbam.com/th/fund/nav-historical/",
            data={"start": _to_be(start_ce), "end": _to_be(end_ce)},
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

    # Find all fund column blocks: <h6>FUND</h6>...<p style="...padding-right...">NAV</p>
    nav_blocks = re.finditer(
        rf'<h6>{fund_escaped}</h6>.*?<p[^>]*padding-right[^>]*>\s*(?:<img[^>]*>\s*)?([0-9]+\.[0-9]+)',
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
    return [{"date": d, "nav": v} for d, v in sorted(seen.items())]


def is_scbam_fund(fund_name: str) -> bool:
    """Heuristic: SCBAM fund names start with SCB, PVD, or known SCBAM prefixes."""
    upper = fund_name.upper()
    return any(upper.startswith(prefix) for prefix in [
        "SCB", "PVDFP", "PVDTP", "PVDTPP", "PVDFPP"
    ])
