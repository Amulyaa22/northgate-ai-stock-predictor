"""
Northgate AI Stock Predictor - FastAPI Web Application & API Server
Ultra-fast, responsive FinTech Web UI powered by Plotly.js and Tailwind CSS.
"""

from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from config import DEFAULT_INDIAN_TICKERS, TIME_PERIODS, FORECAST_HORIZONS
from core_engine import (
    fetch_market_data,
    calculate_sma,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr,
    compute_pivots_support_resistance,
    train_and_forecast_pure,
    get_stock_news,
    synthesize_trade_recommendation,
    backtest_strategy
)

app = FastAPI(title="Northgate AI Stock Predictor")


@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Northgate AI Stock Predictor API",
        "engine": "Hybrid Quant & ML Time-Series Engine",
        "version": "1.0.0"
    }


@app.get("/api/tickers")
@app.get("/tickers")
def get_tickers():
    return [{"symbol": k, "name": v} for k, v in DEFAULT_INDIAN_TICKERS.items()]


@app.get("/api/analysis")
@app.get("/analysis")
def get_analysis(
    ticker: str = Query("RELIANCE.NS"),
    period: str = Query("1y"),
    horizon: int = Query(5)
):
    market_data = fetch_market_data(ticker, period=period)
    if "error" in market_data:
        return {"error": market_data["error"]}

    bars = market_data["bars"]
    dates = [b["date"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    # Technical Indicators
    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50)
    sma200 = calculate_sma(closes, 200)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(closes, 20)
    rsi14 = calculate_rsi(closes, 14)
    macd, macd_signal, macd_hist = calculate_macd(closes)
    atr = calculate_atr(highs, lows, closes, 14)
    pivots = compute_pivots_support_resistance(highs, lows, closes)

    # ML Forecast
    ml_forecast = train_and_forecast_pure(
        closes, highs, lows, volumes,
        sma20, sma50, rsi14, macd, macd_signal, atr,
        horizon=horizon
    )

    # News & Sentiment
    news = get_stock_news(ticker, market_data["company_name"])

    # Hybrid Recommendation
    recommendation = synthesize_trade_recommendation(
        current_price=closes[-1],
        ml_forecast=ml_forecast,
        rsi=rsi14[-1] if rsi14[-1] is not None else 50.0,
        macd_hist=macd_hist[-1],
        sma20=sma20[-1] or closes[-1],
        sentiment=news,
        pivots=pivots,
        atr=atr[-1]
    )

    # Backtesting
    backtest = backtest_strategy(closes, dates, sma20, rsi14, macd_hist)

    # Metrics
    curr_c = closes[-1]
    prev_c = market_data["previous_close"]
    chg = round(curr_c - prev_c, 2)
    chg_pct = round((chg / prev_c) * 100, 2) if prev_c else 0.0

    return {
        "symbol": market_data["symbol"],
        "company_name": market_data["company_name"],
        "currency": market_data["currency"],
        "current_price": curr_c,
        "previous_close": prev_c,
        "change": chg,
        "change_pct": chg_pct,
        "day_high": max(highs[-1], curr_c),
        "day_low": min(lows[-1], curr_c),
        "year_high": max(highs),
        "year_low": min(lows),
        "volume": volumes[-1],
        "bars": bars,
        "indicators": {
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "pivots": pivots
        },
        "ml_forecast": ml_forecast,
        "news": news,
        "recommendation": recommendation,
        "backtest": backtest
    }


import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
def serve_dashboard():
    if not TEMPLATE_PATH.exists():
        # Fallback to local relative path
        alt_path = Path("templates") / "index.html"
        if alt_path.exists():
            return alt_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>Northgate AI Stock Predictor</h1><p>Dashboard template loading...</p>")
    return TEMPLATE_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

