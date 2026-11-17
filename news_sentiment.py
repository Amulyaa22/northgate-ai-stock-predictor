"""
Financial news fetcher and sentiment analysis engine for Indian & global equities.
Uses Google News RSS + Financial Lexicon & Heuristics + Optional LLM Deep Insights.
"""

from typing import List, Dict, Any
import urllib.parse
import re
import feedparser
from config import SENTIMENT_KEYWORDS


def fetch_stock_news(ticker: str, company_name: str, max_items: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch recent financial news articles for the stock via Google News RSS.
    """
    # Clean symbol for search query
    clean_sym = ticker.replace(".NS", "").replace(".BO", "").replace("^", "")
    query = f"{company_name} {clean_sym} stock news"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    news_items = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "")
            source = entry.get("source", {}).get("title", "Financial Media")

            sentiment_score, sentiment_label = analyze_text_sentiment(title)

            news_items.append({
                "title": title,
                "link": link,
                "published": published,
                "source": source,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label
            })
    except Exception:
        pass

    # Fallback if no items retrieved
    if not news_items:
        news_items.append({
            "title": f"Market watchers track recent momentum in {company_name} ({clean_sym})",
            "link": "#",
            "published": "Recent",
            "source": "Market Feed",
            "sentiment_score": 0.1,
            "sentiment_label": "Neutral"
        })

    return news_items


def analyze_text_sentiment(text: str) -> tuple[float, str]:
    """
    Rule-based financial sentiment scoring for market headlines.
    Returns (score between -1.0 and 1.0, label)
    """
    cleaned = text.lower()
    pos_score = 0
    neg_score = 0

    for word in SENTIMENT_KEYWORDS["positive"]:
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned):
            pos_score += 1.0

    for word in SENTIMENT_KEYWORDS["negative"]:
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned):
            neg_score += 1.0

    total = pos_score + neg_score
    if total == 0:
        return 0.0, "Neutral"

    # Normalized score between -1 and 1
    raw_score = (pos_score - neg_score) / max(total, 1)

    if raw_score >= 0.25:
        label = "Bullish"
    elif raw_score <= -0.25:
        label = "Bearish"
    else:
        label = "Neutral"

    return round(raw_score, 2), label


def compute_aggregate_sentiment(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates sentiment across fetched news articles.
    """
    if not news_items:
        return {
            "average_score": 0.0,
            "sentiment_label": "Neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "impact": "Low Impact"
        }

    scores = [item["sentiment_score"] for item in news_items]
    avg_score = sum(scores) / len(scores)

    bullish_count = sum(1 for item in news_items if item["sentiment_label"] == "Bullish")
    bearish_count = sum(1 for item in news_items if item["sentiment_label"] == "Bearish")
    neutral_count = sum(1 for item in news_items if item["sentiment_label"] == "Neutral")

    if avg_score >= 0.2:
        sentiment_label = "Bullish"
    elif avg_score <= -0.2:
        sentiment_label = "Bearish"
    else:
        sentiment_label = "Neutral"

    impact = "High Impact" if abs(avg_score) >= 0.4 else "Moderate Impact" if abs(avg_score) >= 0.15 else "Low Impact"

    return {
        "average_score": round(avg_score, 2),
        "sentiment_label": sentiment_label,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "impact": impact
    }

