"""
Verification test script for Northgate AI Stock Predictor pipeline.
"""

import sys
from config import DEFAULT_INDIAN_TICKERS
from data_loader import fetch_stock_history, fetch_stock_info
from indicators import add_technical_indicators, compute_support_resistance
from news_sentiment import fetch_stock_news, compute_aggregate_sentiment
from ml_engine import train_and_forecast
from strategy import generate_technical_signal, synthesize_northgate_recommendation
from backtester import run_backtest


def main():
    ticker = "RELIANCE.NS"
    print(f"=== Testing Northgate AI Pipeline for {ticker} ===")

    # 1. Test data loader
    print("\n1. Fetching historical data...")
    df, err = fetch_stock_history(ticker, period="1y", interval="1d")
    if err or df is None or df.empty:
        print(f"FAILED data fetch: {err}")
        sys.exit(1)
    print(f"SUCCESS: Fetched {len(df)} rows. Latest Close: ₹{df['Close'].iloc[-1]:.2f}")

    # 2. Test indicators
    print("\n2. Computing technical indicators...")
    df = add_technical_indicators(df)
    sr = compute_support_resistance(df)
    print(f"SUCCESS: Indicators added. RSI={df['RSI_14'].iloc[-1]:.2f}, MACD={df['MACD'].iloc[-1]:.2f}")
    print(f"Support/Resistance: Pivot=₹{sr['pivot']}, S1=₹{sr['support_1']}, R1=₹{sr['resistance_1']}")

    # 3. Test ML engine
    print("\n3. Training ML forecast models...")
    ml_result = train_and_forecast(df, horizon_days=5)
    print(f"SUCCESS: Forecast Price in 5 Days: ₹{ml_result['forecast_price']} ({ml_result['predicted_return_pct']}%)")
    print(f"Trend: {ml_result['trend']} (Bullish Prob: {ml_result['bullish_probability']}%)")
    print(f"Backtest Accuracy: {ml_result['test_accuracy']}%")

    # 4. Test News Sentiment
    print("\n4. Testing News & Sentiment...")
    news = fetch_stock_news(ticker, "Reliance Industries")
    sentiment = compute_aggregate_sentiment(news)
    print(f"SUCCESS: Fetched {len(news)} news items. Sentiment Score: {sentiment['average_score']} ({sentiment['sentiment_label']})")

    # 5. Test Strategy Recommendation
    print("\n5. Synthesizing Northgate Recommendation...")
    tech_sig = generate_technical_signal(df)
    rec = synthesize_northgate_recommendation(tech_sig, ml_result, sentiment, sr)
    print(f"SUCCESS: Action: {rec['action']} (Conviction: {rec['confidence_pct']}%)")
    print(f"Stop Loss: ₹{rec['stop_loss']} | Target 1: ₹{rec['target_1']} | Target 2: ₹{rec['target_2']} | R:R: {rec['risk_reward_ratio']}")

    # 6. Test Backtester
    print("\n6. Running Strategy Backtest...")
    bt = run_backtest(df)
    print(f"SUCCESS: Strategy Return: {bt['strategy_return_pct']}% vs Benchmark: {bt['benchmark_return_pct']}%")
    print(f"Win Rate: {bt['win_rate_pct']}% | Max Drawdown: {bt['max_drawdown_pct']}%")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()

