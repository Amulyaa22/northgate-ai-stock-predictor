"""
Configuration settings and Indian stock tickers for Northgate AI Stock Predictor.
"""

DEFAULT_INDIAN_TICKERS = {
    # NIFTY 50 Heavyweights
    "RELIANCE.NS": "Reliance Industries Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "INFY.NS": "Infosys Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "BHARTIARTL.NS": "Bharti Airtel Ltd",
    "SBIN.NS": "State Bank of India",
    "LT.NS": "Larsen & Toubro Ltd",
    "ITC.NS": "ITC Ltd",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd",
    "TATAMOTORS.NS": "Tata Motors Ltd",
    "M&M.NS": "Mahindra & Mahindra Ltd",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries",
    "BAJFINANCE.NS": "Bajaj Finance Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd",
    "TITAN.NS": "Titan Company Ltd",
    "ADANIENT.NS": "Adani Enterprises Ltd",
    "ADANIPORTS.NS": "Adani Ports and SEZ",
    "MARUTI.NS": "Maruti Suzuki India Ltd",
    "NTPC.NS": "NTPC Ltd",
    "ONGC.NS": "Oil & Natural Gas Corp Ltd",
    "POWERGRID.NS": "Power Grid Corporation",
    "TATASTEEL.NS": "Tata Steel Ltd",
    "ASIANPAINT.NS": "Asian Paints Ltd",
    "WIPRO.NS": "Wipro Ltd",
    # Benchmark Indices
    "^NSEI": "NIFTY 50 Index",
    "^NSEBANK": "NIFTY BANK Index",
    "^BSESN": "BSE SENSEX Index"
}

# Historical data lookback presets
TIME_PERIODS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

INTERVALS = {
    "Daily (1d)": "1d",
    "Weekly (1wk)": "1wk",
    "Hourly (1h)": "1h"
}

# ML Parameters
FORECAST_HORIZONS = {
    "Next 1 Trading Day": 1,
    "Next 5 Trading Days (1 Week)": 5,
    "Next 10 Trading Days (2 Weeks)": 10,
    "Next 20 Trading Days (1 Month)": 20
}

# Financial positive/negative keywords for rule-based Indian market sentiment
SENTIMENT_KEYWORDS = {
    "positive": [
        "surge", "jump", "rally", "gain", "profit", "growth", "record high", "bullish",
        "beat estimates", "outperform", "dividend", "order win", "expansion", "upgrade",
        "buy", "strong quarterly", "revenue rise", "partnership", "all-time high",
        "fii buying", "dii buying", "green", "breakout"
    ],
    "negative": [
        "fall", "drop", "plunge", "loss", "decline", "bearish", "miss estimates",
        "downgrade", "selloff", "probe", "fraud", "penalty", "slump", "debt",
        "inflation", "recession", "fii selling", "down", "crash", "weak guidance",
        "underperform", "lawsuit", "default"
    ]
}

