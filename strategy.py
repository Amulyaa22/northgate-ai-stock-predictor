"""
Hybrid Fusion Engine: Combines Technical Indicators, Machine Learning Forecasts,
and Financial News Sentiment into the Northgate Trade Recommendation.
"""

from typing import Dict, Any
import pandas as pd


def generate_technical_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates rule-based technical indicator signals.
    """
    latest = df.iloc[-1]
    score = 0.0
    signals = []

    # 1. Trend: Moving Average Alignment (Golden Cross / Bullish stack)
    close = latest["Close"]
    sma_20 = latest.get("SMA_20", close)
    sma_50 = latest.get("SMA_50", close)
    sma_200 = latest.get("SMA_200", close)

    if close > sma_20 > sma_50:
        score += 25
        signals.append("Bullish Trend: Price trading above 20 & 50 SMA")
    elif close < sma_20 < sma_50:
        score -= 25
        signals.append("Bearish Trend: Price trading below 20 & 50 SMA")
    else:
        signals.append("Trend: Moving averages showing consolidation")

    # 2. RSI (14)
    rsi = latest.get("RSI_14", 50)
    if rsi < 30:
        score += 25
        signals.append(f"RSI Oversold ({rsi:.1f}): Potential reversal buying zone")
    elif rsi > 70:
        score -= 25
        signals.append(f"RSI Overbought ({rsi:.1f}): Potential pullback caution")
    elif 45 <= rsi <= 60:
        score += 10
        signals.append(f"RSI Healthy Momentum ({rsi:.1f})")

    # 3. MACD Crossover
    macd = latest.get("MACD", 0)
    signal = latest.get("MACD_Signal", 0)
    hist = latest.get("MACD_Hist", 0)
    if macd > signal and hist > 0:
        score += 25
        signals.append("MACD Bullish: MACD Line above Signal with expanding histogram")
    elif macd < signal and hist < 0:
        score -= 25
        signals.append("MACD Bearish: MACD Line below Signal")

    # 4. Bollinger Bands
    bb_upper = latest.get("BB_Upper", close * 1.05)
    bb_lower = latest.get("BB_Lower", close * 0.95)
    if close <= bb_lower:
        score += 15
        signals.append("Bollinger Bands: Price touching lower band (Value Zone)")
    elif close >= bb_upper:
        score -= 15
        signals.append("Bollinger Bands: Price touching upper band (Extended)")

    return {
        "raw_score": score,
        "signals": signals
    }


def synthesize_northgate_recommendation(
    technical_signal: Dict[str, Any],
    ml_result: Dict[str, Any],
    sentiment_result: Dict[str, Any],
    support_resistance: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes Technicals (40%), ML Forecast (40%), and News Sentiment (20%)
    into a comprehensive Northgate AI Signal with Risk/Reward.
    """
    # 1. Technical Score (-100 to +100) -> Normalized to -1.0 to 1.0
    tech_score = max(-1.0, min(1.0, technical_signal["raw_score"] / 75.0))

    # 2. ML Score (Bullish prob from 0% to 100% -> Normalized to -1.0 to 1.0)
    ml_prob = ml_result["bullish_probability"] / 100.0
    ml_score = (ml_prob - 0.5) * 2.0  # e.g., 0.8 prob -> +0.6 score

    # 3. Sentiment Score (-1.0 to +1.0)
    sent_score = sentiment_result.get("average_score", 0.0)

    # Hybrid Weighted Composite Score: 40% ML + 40% Tech + 20% News
    composite_score = (0.40 * ml_score) + (0.40 * tech_score) + (0.20 * sent_score)
    confidence_pct = int(min(98, max(52, 50 + (abs(composite_score) * 48))))

    current_price = ml_result["current_price"]
    forecast_price = ml_result["forecast_price"]
    atr = max(current_price * 0.015, support_resistance.get("pivot", current_price) * 0.015)

    if composite_score >= 0.35:
        action = "STRONG BUY"
        action_color = "#00C853"
        stop_loss = round(max(0.1, current_price - (1.5 * atr)), 2)
        target_1 = round(max(forecast_price, current_price + (1.8 * atr)), 2)
        target_2 = round(max(target_1 * 1.02, support_resistance.get("resistance_2", target_1 * 1.03)), 2)
    elif composite_score >= 0.12:
        action = "BUY"
        action_color = "#64DD17"
        stop_loss = round(max(0.1, current_price - (1.4 * atr)), 2)
        target_1 = round(max(forecast_price, current_price + (1.5 * atr)), 2)
        target_2 = round(target_1 * 1.02, 2)
    elif composite_score <= -0.35:
        action = "STRONG SELL"
        action_color = "#D50000"
        stop_loss = round(current_price + (1.5 * atr), 2)
        target_1 = round(min(forecast_price, current_price - (1.8 * atr)), 2)
        target_2 = round(min(target_1 * 0.98, support_resistance.get("support_2", target_1 * 0.97)), 2)
    elif composite_score <= -0.12:
        action = "SELL"
        action_color = "#FF1744"
        stop_loss = round(current_price + (1.4 * atr), 2)
        target_1 = round(min(forecast_price, current_price - (1.5 * atr)), 2)
        target_2 = round(target_1 * 0.98, 2)
    else:
        action = "HOLD / NEUTRAL"
        action_color = "#FFAB00"
        stop_loss = round(max(0.1, current_price - (1.2 * atr)), 2)
        target_1 = round(current_price + (1.2 * atr), 2)
        target_2 = round(current_price + (2.0 * atr), 2)

    # Risk to Reward
    risk = abs(current_price - stop_loss)
    reward = abs(target_1 - current_price)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 1.0

    return {
        "action": action,
        "action_color": action_color,
        "confidence_pct": confidence_pct,
        "composite_score": round(composite_score, 3),
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_reward_ratio": f"1 : {rr_ratio}",
        "summary": f"Northgate AI synthesizes a {action} stance for the chosen horizon with {confidence_pct}% model conviction."
    }

