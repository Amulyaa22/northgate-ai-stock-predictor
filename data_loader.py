"""
Data acquisition and caching module for Indian and global equities.
"""

from typing import Dict, Any, Optional, Tuple
import datetime
import pandas as pd
import yfinance as yf


def fetch_stock_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch historical OHLCV data for a given ticker.
    Returns: (DataFrame, error_message)
    """
    try:
        clean_ticker = ticker.strip().upper()
        # Automatically append .NS if user entered a plain Indian stock symbol without dot
        if not clean_ticker.startswith("^") and "." not in clean_ticker:
            # Check if likely an Indian stock symbol
            clean_ticker = f"{clean_ticker}.NS"

        stock = yf.Ticker(clean_ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)

        if df.empty or len(df) < 15:
            # Fallback attempt if .NS was appended and empty, try .BO (Bombay Stock Exchange)
            if clean_ticker.endswith(".NS"):
                alt_ticker = clean_ticker.replace(".NS", ".BO")
                df_alt = yf.Ticker(alt_ticker).history(period=period, interval=interval, auto_adjust=True)
                if not df_alt.empty and len(df_alt) >= 15:
                    return df_alt, None

            return None, f"Insufficient or no historical data found for '{clean_ticker}'."

        # Standardize column names
        df.columns = [col.capitalize() for col in df.columns]
        df = df.sort_index()
        # Drop rows with NaN in Close
        df = df.dropna(subset=["Close"])

        return df, None
    except Exception as e:
        return None, f"Failed to fetch data: {str(e)}"


def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch summary metadata, fundamentals, and company overview.
    """
    try:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker.startswith("^") and "." not in clean_ticker:
            clean_ticker = f"{clean_ticker}.NS"

        stock = yf.Ticker(clean_ticker)
        info = stock.info or {}

        summary = {
            "symbol": clean_ticker,
            "shortName": info.get("shortName") or info.get("longName") or clean_ticker,
            "currency": info.get("currency", "INR"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open") or info.get("regularMarketOpen"),
            "dayLow": info.get("dayLow"),
            "dayHigh": info.get("dayHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "No company profile available.")
        }
        return summary
    except Exception:
        return {
            "symbol": ticker,
            "shortName": ticker,
            "currency": "INR",
            "currentPrice": None,
            "sector": "N/A",
            "industry": "N/A",
            "summary": "Data unavailable."
        }

