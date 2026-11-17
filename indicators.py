"""
Technical indicator computation engine for equities.
Supports SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Stochastic, and OBV.
"""

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and appends technical indicators to the OHLCV DataFrame.
    """
    data = df.copy()

    # Ensure required columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # 1. Simple Moving Averages (SMA)
    data["SMA_20"] = close.rolling(window=20).mean()
    data["SMA_50"] = close.rolling(window=50).mean()
    data["SMA_200"] = close.rolling(window=200).mean()

    # 2. Exponential Moving Averages (EMA)
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # 3. MACD and Signal Line
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # 4. Relative Strength Index (RSI - 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    # Smooth RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data["RSI_14"] = data["RSI_14"].fillna(50)

    # 5. Bollinger Bands (20-day, 2 std dev)
    rolling_mean_20 = close.rolling(window=20).mean()
    rolling_std_20 = close.rolling(window=20).std()
    data["BB_Upper"] = rolling_mean_20 + (rolling_std_20 * 2)
    data["BB_Lower"] = rolling_mean_20 - (rolling_std_20 * 2)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / rolling_mean_20

    # 6. Average True Range (ATR - 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR_14"] = true_range.rolling(window=14).mean()

    # 7. Stochastic Oscillator (%K, %D)
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    denom = high_14 - low_14
    data["Stoch_K"] = 100 * ((close - low_14) / denom.replace(0, np.nan))
    data["Stoch_D"] = data["Stoch_K"].rolling(window=3).mean()

    # 8. Volume Moving Average & On-Balance Volume (OBV)
    data["Vol_SMA_20"] = volume.rolling(window=20).mean()
    obv_direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    data["OBV"] = (volume * obv_direction).cumsum()

    # 9. Daily & Multi-day Returns
    data["Daily_Return"] = close.pct_change()
    data["Return_5d"] = close.pct_change(5)
    data["Return_10d"] = close.pct_change(10)
    data["Return_20d"] = close.pct_change(20)

    # 10. Historical Volatility (20-day annualized)
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

    return data


def compute_support_resistance(df: pd.DataFrame, window: int = 30) -> dict:
    """
    Computes key local support and resistance pivot levels based on recent highs and lows.
    """
    recent = df.tail(window)
    if recent.empty:
        return {"support_1": 0, "support_2": 0, "resistance_1": 0, "resistance_2": 0}

    current_price = recent["Close"].iloc[-1]
    highs = recent["High"]
    lows = recent["Low"]

    # Pivot Point formulas
    last_high = highs.max()
    last_low = lows.min()
    pivot = (last_high + last_low + current_price) / 3

    r1 = (2 * pivot) - last_low
    s1 = (2 * pivot) - last_high
    r2 = pivot + (last_high - last_low)
    s2 = pivot - (last_high - last_low)

    return {
        "pivot": round(pivot, 2),
        "resistance_1": round(r1, 2),
        "resistance_2": round(r2, 2),
        "support_1": round(s1, 2),
        "support_2": round(s2, 2)
    }

