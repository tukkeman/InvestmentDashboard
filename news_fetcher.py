import feedparser
import streamlit as st


@st.cache_data(ttl=1800)
def fetch_news(ticker: str, max_items: int = 6) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:max_items]:
            summary = entry.get("summary", "")
            news.append({
                "ticker": ticker,
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": summary[:250] + "..." if len(summary) > 250 else summary,
            })
        return news
    except Exception:
        return []


def fetch_all_news(tickers: list[str], max_per_ticker: int = 4) -> list[dict]:
    all_news = []
    for ticker in tickers:
        all_news.extend(fetch_news(ticker, max_per_ticker))
    return all_news
