import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict

from data_fetcher import fetch_price_data, fetch_watchlist_summary, fetch_ticker_signal
from technical_analysis import add_all_indicators, get_signals, get_recommendation
from news_fetcher import fetch_news, fetch_all_news
import thai_fund
import scbam_fetcher

_WL_DND = components.declare_component(
    "watchlist_dnd",
    path=os.path.join(os.path.dirname(__file__), "watchlist_dnd"),
)
_CTX_MENU = components.declare_component(
    "context_menu",
    path=os.path.join(os.path.dirname(__file__), "context_menu"),
)


def _auto_fetch_scbam(ticker: str) -> bool:
    """Fetch/refresh SCBAM NAV data automatically. Returns True if new data was saved."""
    if not scbam_fetcher.is_scbam_fund(ticker):
        return False
    if thai_fund.was_fetched_today(ticker):
        return False  # already ran today, avoid infinite rerun loop

    has_data = thai_fund.has_nav_data(ticker)
    # SCBAM serves at most ~110 days per request; fetching more returns empty
    days = 100 if not has_data else 30
    msg = (
        f"Fetching 1 year of NAV data for {ticker} from scbam.com (first time, ~30s)…"
        if not has_data
        else f"Updating {ticker} NAV data…"
    )

    with st.spinner(msg):
        records = scbam_fetcher.fetch_nav(ticker, days=days)

    if not records:
        thai_fund.mark_fetched_today(ticker)  # avoid retrying repeatedly on failure
        return False

    data = thai_fund.load_all()
    merged = {e["date"]: e["nav"] for e in data.get(ticker, [])}
    for rec in records:
        merged[rec["date"]] = rec["nav"]
    data[ticker] = sorted(
        [{"date": d, "nav": v} for d, v in merged.items()],
        key=lambda x: x["date"],
    )
    thai_fund._save(data)
    thai_fund.mark_fetched_today(ticker)
    st.cache_data.clear()
    st.rerun()
    return True

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1a1f2e;
    border-radius: 8px;
    padding: 14px 18px;
    border-left: 3px solid;
}
.signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
}
.news-card {
    background: #1a1f2e;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
    border-left: 3px solid #00d4aa;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Watchlist persistence ─────────────────────────────────────────────────────
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

def load_watchlist() -> list[str]:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
            return data.get("tickers", [])
    return ["SPY", "QQQ"]

def load_watchlist_groups() -> tuple[dict, list]:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
            return data.get("groups", {}), data.get("group_names", [])
    return {}, []

def save_watchlist(tickers: list[str]):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({
            "tickers": tickers,
            "groups": st.session_state.get("ticker_groups", {}),
            "group_names": st.session_state.get("group_names", []),
        }, f, indent=2)

# ── Session state ─────────────────────────────────────────────────────────────
if "tickers" not in st.session_state:
    st.session_state.tickers = load_watchlist()
if "selected" not in st.session_state:
    st.session_state.selected = st.session_state.tickers[0] if st.session_state.tickers else ""
if "goto_analysis" not in st.session_state:
    st.session_state.goto_analysis = False
if "dnd_last_id" not in st.session_state:
    st.session_state.dnd_last_id = ""
if "ticker_groups" not in st.session_state:
    st.session_state.ticker_groups, st.session_state.group_names = load_watchlist_groups()
if "ctx_last_id" not in st.session_state:
    st.session_state.ctx_last_id = ""
if "grp_dnd_last_id" not in st.session_state:
    st.session_state.grp_dnd_last_id = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Investment Dashboard")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    st.markdown("### Watchlist")
    with st.form("add_ticker_form", clear_on_submit=True):
        col_in, col_btn = st.columns([3, 1])
        with col_in:
            new_ticker = st.text_input(
                "ticker_input", placeholder="Add ticker…", label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("＋", type="primary", use_container_width=True)
        if submitted:
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.tickers:
                st.session_state.tickers.append(t)
                save_watchlist(st.session_state.tickers)
                st.session_state.selected = t
                st.rerun()

    # Unified drag / click-to-select / ✕-remove / group component
    _n_groups = len({v for v in st.session_state.ticker_groups.values() if v})
    _h = max(len(st.session_state.tickers) * 48 + _n_groups * 28 + 20, 50)
    _goto = st.session_state.goto_analysis
    if _goto:
        st.session_state.goto_analysis = False  # reset before next rerun
    wl_result = _WL_DND(
        tickers=st.session_state.tickers,
        selected=st.session_state.selected,
        goto_analysis=_goto,
        groups=st.session_state.ticker_groups,
        group_names=st.session_state.group_names,
        default=None,
        key="wl_dnd",
        height=_h,
    )
    if wl_result:
        _id     = wl_result.get("_id")
        _action = wl_result.get("action")
        _order  = wl_result.get("order", st.session_state.tickers)
        _ticker = wl_result.get("ticker", "")
        if _id != st.session_state.dnd_last_id:
            st.session_state.dnd_last_id = _id
            if _action == "order" and _order != st.session_state.tickers:
                st.session_state.tickers = _order
                save_watchlist(_order)
                st.rerun()
            elif _action == "select" and _ticker in st.session_state.tickers:
                st.session_state.selected = _ticker
                st.session_state.goto_analysis = True
                st.rerun()
            elif _action == "remove" and _ticker in st.session_state.tickers:
                st.session_state.tickers = _order
                save_watchlist(_order)
                if st.session_state.selected == _ticker:
                    st.session_state.selected = _order[0] if _order else ""
                st.rerun()
            elif _action == "assign_group":
                _t = wl_result.get("ticker", "")
                _g = wl_result.get("group")
                if _t in st.session_state.tickers:
                    if _g:
                        st.session_state.ticker_groups[_t] = _g
                    else:
                        st.session_state.ticker_groups.pop(_t, None)
                    save_watchlist(st.session_state.tickers)
                    st.rerun()

    # ── Group management ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Groups")
    with st.form("add_group_form", clear_on_submit=True):
        _gi_col, _gb_col = st.columns([3, 1])
        with _gi_col:
            _new_grp = st.text_input(
                "group_name", placeholder="New group name…", label_visibility="collapsed"
            )
        with _gb_col:
            _grp_submitted = st.form_submit_button("＋", type="primary", use_container_width=True)
        if _grp_submitted and _new_grp.strip():
            _gn = _new_grp.strip()
            if _gn not in st.session_state.group_names:
                st.session_state.group_names.append(_gn)
                save_watchlist(st.session_state.tickers)
                st.rerun()

    if st.session_state.group_names:
        st.caption("Drag to reorder · ✕ to delete · right-click fund to assign")
        _gh = max(len(st.session_state.group_names) * 48 + 12, 50)
        grp_result = _WL_DND(
            tickers=st.session_state.group_names,
            selected="",
            goto_analysis=False,
            groups={},
            group_names=[],
            default=None,
            key="groups_dnd",
            height=_gh,
        )
        if grp_result:
            _gid = grp_result.get("_id")
            if _gid != st.session_state.grp_dnd_last_id:
                st.session_state.grp_dnd_last_id = _gid
                _gaction = grp_result.get("action")
                _gorder  = grp_result.get("order", st.session_state.group_names)
                _gname   = grp_result.get("ticker", "")
                if _gaction == "order" and _gorder != st.session_state.group_names:
                    st.session_state.group_names = _gorder
                    save_watchlist(st.session_state.tickers)
                    st.rerun()
                elif _gaction == "remove" and _gname in st.session_state.group_names:
                    st.session_state.group_names = [g for g in _gorder if g != _gname]
                    for _t in list(st.session_state.ticker_groups):
                        if st.session_state.ticker_groups[_t] == _gname:
                            del st.session_state.ticker_groups[_t]
                    save_watchlist(st.session_state.tickers)
                    st.rerun()
    else:
        st.caption("No groups yet.")

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="font-size:0.75em; color: #888; line-height:1.6em;">
    <b>Thai SET funds:</b> add <code>.BK</code><br>
    Example: <code>TDEX.BK</code>, <code>ESET.BK</code><br><br>
    Thai mutual funds (open-end) may not be available via Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
if not st.session_state.tickers:
    st.info("Add a ticker in the sidebar to get started.")
    st.stop()

# ── Right-click context menu (invisible, handles Overview fund rows) ──────────
ctx_result = _CTX_MENU(
    group_names=st.session_state.group_names,
    key="ctx_menu",
    default=None,
    height=0,
)
if ctx_result:
    _cid = ctx_result.get("_id")
    if _cid != st.session_state.ctx_last_id:
        st.session_state.ctx_last_id = _cid
        if ctx_result.get("action") == "assign_group":
            _ct = ctx_result.get("ticker", "")
            _cg = ctx_result.get("group")
            if _ct in st.session_state.tickers:
                if _cg:
                    st.session_state.ticker_groups[_ct] = _cg
                else:
                    st.session_state.ticker_groups.pop(_ct, None)
                save_watchlist(st.session_state.tickers)
                st.rerun()

tab_overview, tab_analysis, tab_news = st.tabs(["📊 Overview", "📉 Technical Analysis", "📰 News"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Watchlist Overview")

    # Auto-fetch SCBAM NAV data for any SCB funds in watchlist
    for _t in st.session_state.tickers:
        _auto_fetch_scbam(_t)

    with st.spinner("Fetching market data…"):
        summary_df = fetch_watchlist_summary(st.session_state.tickers)

    if summary_df.empty:
        st.warning("Could not load data. Check your tickers and internet connection.")
    else:
        # Top metrics row
        valid = summary_df[~summary_df.get("error", pd.Series(dtype=str)).notna()] if "error" in summary_df else summary_df
        gainers = int((summary_df.get("change_pct", pd.Series(dtype=float)) > 0).sum())
        losers = int((summary_df.get("change_pct", pd.Series(dtype=float)) < 0).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Watchlist Size", len(st.session_state.tickers))
        m2.metric("Gainers Today", gainers, delta=f"{gainers} ↑", delta_color="normal")
        m3.metric("Losers Today", losers, delta=f"{losers} ↓", delta_color="inverse")

        st.divider()

        # ── Build grouped rows ────────────────────────────────────────────────
        UNGROUPED = "— Ungrouped"
        grouped_rows: dict = defaultdict(list)
        for row_i, row in summary_df.iterrows():
            g = st.session_state.ticker_groups.get(row.get("ticker", ""), UNGROUPED)
            if g not in st.session_state.group_names:
                g = UNGROUPED
            grouped_rows[g].append((row_i, row))

        display_order = [g for g in st.session_state.group_names if g in grouped_rows]
        if UNGROUPED in grouped_rows:
            display_order.append(UNGROUPED)
        show_headers = bool(st.session_state.group_names)

        if show_headers:
            st.caption("Right-click any fund row to assign its group.")

        # ── Compact fund list — CSS targets the container via :has() sentinel ─
        st.markdown("""
<style>
[data-testid="stVerticalBlock"]:has(.ov-list){gap:.15rem!important}
[data-testid="stVerticalBlock"]:has(.ov-list) [data-testid="stHorizontalBlock"]{align-items:center}
[data-testid="stVerticalBlock"]:has(.ov-list) [data-testid="stBaseButton-secondary"]{padding:2px 8px!important;min-height:0!important;height:auto!important;line-height:1.5!important;font-size:0.82em!important;font-weight:700!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}
[data-testid="stVerticalBlock"]:has(.ov-list) hr{margin:0!important}
</style>
<div class="ov-list"></div>""", unsafe_allow_html=True)

        with st.container():
            for group_name in display_order:
                rows_in_group = grouped_rows[group_name]

                if show_headers:
                    st.markdown(
                        f"<div style='padding:6px 0 4px 0;margin-top:6px;border-bottom:1px solid #2a3050'>"
                        f"<span style='font-weight:700;font-size:0.9em;color:#e0e0e0'>{group_name}</span>"
                        f"<span style='font-size:0.75em;color:#888;margin-left:8px'>"
                        f"{len(rows_in_group)} fund{'s' if len(rows_in_group) != 1 else ''}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                for row_i, row in rows_in_group:
                    if "error" in row and pd.notna(row.get("error")):
                        ticker_name = row["ticker"]
                        with st.expander(f"⚠️ {ticker_name} — data unavailable"):
                            if "(" in ticker_name:
                                st.caption(
                                    f"**{ticker_name}** — NAV data not yet loaded.  \n"
                                    "Go to the **Technical Analysis** tab to trigger an automatic fetch from scbam.com."
                                )
                            else:
                                st.caption(
                                    f"Could not load data for **{ticker_name}**. "
                                    "Click 🔄 Refresh Data in the sidebar and try again. "
                                    "If the problem persists, verify the ticker on finance.yahoo.com."
                                )
                        continue

                    chg = row.get("change_pct", 0)
                    price = row.get("price", 0)
                    chg_color = "#26a69a" if chg >= 0 else "#ef5350"
                    chg_arrow = "▲" if chg >= 0 else "▼"
                    currency = row.get("currency", "")

                    high = row.get("high_52w", 0)
                    low = row.get("low_52w", 0)
                    if high and low and high != low:
                        pct_in_range = min(max((price - low) / (high - low) * 100, 0), 100)
                        range_html = (
                            f"<div style='font-size:0.72em;color:#888;line-height:1.2'>{low:.2f} – {high:.2f}</div>"
                            f"<div style='background:#333;border-radius:2px;height:3px;margin-top:2px'>"
                            f"<div style='background:{chg_color};height:3px;width:{pct_in_range:.0f}%;border-radius:2px'></div>"
                            f"</div>"
                        )
                    else:
                        range_html = "<span style='font-size:0.72em;color:#888'>52W N/A</span>"

                    sig = fetch_ticker_signal(row["ticker"])
                    sig_label = sig["label"]
                    sig_color = sig["color"]

                    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 2.5, 1.5, 1.2, 1.5, 1.8])
                    with c1:
                        st.markdown(
                            f'<div class="fund-row-marker" data-ticker="{row["ticker"]}" style="display:none"></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(row["ticker"], key=f"goto_{row_i}"):
                            st.session_state.selected = row["ticker"]
                            st.session_state.goto_analysis = True
                            st.rerun()
                    with c2:
                        label = row.get('name', row['ticker'])
                        cur_tag = f" <span style='color:#888;font-size:0.8em'>{currency}</span>" if currency else ""
                        st.markdown(f"<span style='font-size:0.88em'>{label}{cur_tag}</span>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**{price:,.4f}**")
                    with c4:
                        st.markdown(
                            f"<span style='color:{chg_color};font-weight:700'>{chg_arrow} {abs(chg):.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    with c5:
                        st.markdown(range_html, unsafe_allow_html=True)
                    with c6:
                        st.markdown(
                            f"<span style='background:{sig_color}22;color:{sig_color};border:1px solid {sig_color}44;"
                            f"border-radius:10px;padding:2px 8px;font-size:0.72em;font-weight:600'>{sig_label}</span>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<hr style='border:none;border-top:1px solid #2a3050'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    if not st.session_state.tickers:
        st.info("Add a ticker in the sidebar to get started.")
        st.stop()

    col_fund, col_period = st.columns([3, 1])
    with col_fund:
        idx = st.session_state.tickers.index(st.session_state.selected) if st.session_state.selected in st.session_state.tickers else 0
        ticker = st.selectbox("Fund", st.session_state.tickers, index=idx, label_visibility="collapsed")
        if ticker != st.session_state.selected:
            st.session_state.selected = ticker
            st.rerun()
    with col_period:
        period = st.selectbox(
            "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2, label_visibility="collapsed"
        )

    # ── Auto-fetch SCBAM NAV if needed (no user interaction required) ────────
    _auto_fetch_scbam(ticker)

    with st.spinner(f"Loading {ticker}…"):
        df = fetch_price_data(ticker, period)

    is_thai = thai_fund.has_nav_data(ticker)

    if df.empty:
        if scbam_fetcher.is_scbam_fund(ticker):
            st.error(f"Could not fetch NAV data for **{ticker}** from scbam.com. Check your internet connection.")
        else:
            st.error(
                f"No data found for **{ticker}**. "
                "This fund is not available on Yahoo Finance and is not an SCBAM fund. "
                "Only SET-listed ETFs (e.g. `TDEX.BK`) and SCBAM funds (e.g. `SCBSEMI(A)`) are supported automatically."
            )
        st.stop()

    df = add_all_indicators(df)
    signals = get_signals(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # ── Latest price header ──────────────────────────────────────────────────
    price = last["Close"]
    chg = price - prev["Close"]
    chg_pct = chg / prev["Close"] * 100 if prev["Close"] else 0
    price_color = "#26a69a" if chg >= 0 else "#ef5350"
    chg_arrow = "▲" if chg >= 0 else "▼"
    price_date = df.index[-1].strftime("%d %b %Y")
    currency = "THB" if (is_thai or scbam_fetcher.is_scbam_fund(ticker) or ticker.upper().endswith(".BK")) else "USD"
    st.markdown(
        f"""<div style="padding:10px 4px 4px 2px">
          <div style="font-size:0.78em;color:#888;letter-spacing:.06em;margin-bottom:2px">{ticker} · {price_date}</div>
          <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
            <span style="font-size:2.2em;font-weight:800;color:#e0e0e0">{price:,.4f}</span>
            <span style="font-size:0.82em;color:#aaa">{currency}</span>
            <span style="font-size:1.1em;font-weight:600;color:{price_color}">
              {chg_arrow} {abs(chg):.4f} ({abs(chg_pct):.2f}%)
            </span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Signal cards ────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    cards = [
        ("RSI", signals.get("RSI", ("N/A", "N/A", "gray"))),
        ("MACD", signals.get("MACD", ("N/A", "N/A", "gray"))),
        ("Trend vs SMA20", signals.get("Trend", ("N/A", "N/A", "gray"))),
        ("Bollinger %", signals.get("BB", ("N/A", "N/A", "gray"))),
    ]
    for col, (label, (val, text, color)) in zip([s1, s2, s3, s4], cards):
        with col:
            st.markdown(
                f"""<div class="metric-card" style="border-color:{color}">
                <div style="font-size:0.75em;color:#888">{label}</div>
                <div style="font-size:1.4em;font-weight:700;color:{color}">{val}</div>
                <div style="font-size:0.8em;color:{color}">{text}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Recommendation banner ─────────────────────────────────────────────────
    rec = get_recommendation(signals)
    vote_tags = "".join(
        f"<span style='background:{vc}22;color:{vc};border:1px solid {vc}44;"
        f"border-radius:12px;padding:2px 10px;font-size:0.75em;font-weight:600'>"
        f"{name}: {vote}</span>"
        for name, vote, vc in rec["votes"]
    )
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,{rec['color']}14,{rec['color']}06);
            border:1.5px solid {rec['color']}55;border-radius:10px;padding:14px 18px;margin:10px 0 4px 0">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
            <div>
              <div style="font-size:0.72em;color:#888;letter-spacing:.06em;margin-bottom:3px">OVERALL SIGNAL</div>
              <div style="font-size:1.7em;font-weight:800;color:{rec['color']};letter-spacing:.01em">{rec['label']}</div>
              <div style="font-size:0.82em;color:#bbb;margin-top:5px;max-width:420px">{rec['advice']}</div>
            </div>
            <div style="display:flex;gap:18px;text-align:center;flex-shrink:0">
              <div><div style="font-size:1.5em;font-weight:700;color:#26a69a">{rec['buys']}</div>
                   <div style="font-size:0.68em;color:#888;letter-spacing:.05em">BUY</div></div>
              <div><div style="font-size:1.5em;font-weight:700;color:#888">{rec['neutrals']}</div>
                   <div style="font-size:0.68em;color:#888;letter-spacing:.05em">NEUTRAL</div></div>
              <div><div style="font-size:1.5em;font-weight:700;color:#ef5350">{rec['sells']}</div>
                   <div style="font-size:0.68em;color:#888;letter-spacing:.05em">SELL</div></div>
            </div>
          </div>
          <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">{vote_tags}</div>
          <div style="margin-top:8px;font-size:0.68em;color:#555">
            ⚠ For informational purposes only — not financial advice.
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Chart ────────────────────────────────────────────────────────────────
    st.markdown("")
    overlay_opts = st.multiselect(
        "Overlays",
        ["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26", "Bollinger Bands"],
        default=["SMA 20", "SMA 50", "Bollinger Bands"],
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["", "RSI (14)", "MACD (12/26/9)"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        ),
        row=1, col=1,
    )

    overlay_map = {
        "SMA 20":  ("SMA_20",  "#ffa726", 1.5, "solid"),
        "SMA 50":  ("SMA_50",  "#ab47bc", 1.5, "solid"),
        "SMA 200": ("SMA_200", "#42a5f5", 1.5, "solid"),
        "EMA 12":  ("EMA_12",  "#ff7043", 1,   "dot"),
        "EMA 26":  ("EMA_26",  "#66bb6a", 1,   "dot"),
    }
    for name, (col_name, color, width, dash) in overlay_map.items():
        if name in overlay_opts and col_name in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col_name], name=name,
                           line=dict(color=color, width=width, dash=dash)),
                row=1, col=1,
            )

    if "Bollinger Bands" in overlay_opts and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                       line=dict(color="rgba(100,100,200,0.6)", width=1, dash="dash"),
                       showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                       line=dict(color="rgba(100,100,200,0.6)", width=1, dash="dash"),
                       fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
                       showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid",
                       line=dict(color="rgba(100,100,200,0.4)", width=1),
                       showlegend=False),
            row=1, col=1,
        )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                   line=dict(color="#ce93d8", width=2)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.6)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.6)", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)

    # MACD histogram
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
               marker_color=hist_colors, opacity=0.6),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                   line=dict(color="#29b6f6", width=1.5)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                   line=dict(color="#ff7043", width=1.5)),
        row=3, col=1,
    )

    fig.update_layout(
        height=780,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", bgcolor="rgba(0,0,0,0)",
            yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)", row=i, col=1)

    st.plotly_chart(fig, width="stretch")

    # ── Raw data expander ────────────────────────────────────────────────────
    with st.expander("View raw data"):
        display_cols = ["Open", "High", "Low", "Close", "Volume",
                        "SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
        cols_present = [c for c in display_cols if c in df.columns]
        st.dataframe(df[cols_present].tail(50).iloc[::-1].round(4), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEWS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.subheader("Market News")

    news_scope = st.radio(
        "Show news for",
        [f"Selected: {st.session_state.selected}", "All watchlist"],
        horizontal=True,
    )

    tickers_for_news = (
        [st.session_state.selected]
        if news_scope.startswith("Selected")
        else st.session_state.tickers
    )

    with st.spinner("Loading news…"):
        all_news = fetch_all_news(tickers_for_news, max_per_ticker=5)

    if not all_news:
        st.info("No news found. Yahoo Finance RSS may not cover all tickers.")
    else:
        for item in all_news:
            title = item.get("title", "")
            summary = item.get("summary", "")
            link = item.get("link", "#")
            published = item.get("published", "")
            ticker_tag = item.get("ticker", "")

            st.markdown(
                f"""<div class="news-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px">
                    <span style="background:#00d4aa22; color:#00d4aa; padding:2px 8px;
                           border-radius:10px; font-size:0.75em; font-weight:600">{ticker_tag}</span>
                    <span style="font-size:0.75em; color:#888">{published[:22] if published else ""}</span>
                </div>
                <a href="{link}" target="_blank"
                   style="font-size:0.95em; font-weight:600; color:#fafafa;
                          text-decoration:none; line-height:1.4em">
                   {title}
                </a>
                <p style="font-size:0.8em; color:#aaa; margin-top:6px; margin-bottom:0">{summary}</p>
                </div>""",
                unsafe_allow_html=True,
            )

