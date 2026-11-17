"""
Backtesting engine for evaluating historical performance of the Northgate AI strategy vs Benchmark (Buy & Hold).
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    Runs a chronological backtest over historical price action.
    Generates cumulative return curves, win rate, and drawdown metrics.
    """
    data = df.copy()
    if len(data) < 50:
        raise ValueError("Insufficient data points for backtesting (minimum 50 bars required).")

    # Strategy signals based on Technical Indicators & Momentum:
    # Buy when RSI < 60 and Close > SMA_20 and MACD > MACD_Signal
    # Sell when RSI > 70 or Close < SMA_20 or MACD < MACD_Signal

    buy_condition = (
        (data["Close"] > data["SMA_20"]) &
        (data["MACD"] > data["MACD_Signal"]) &
        (data["RSI_14"] < 65)
    )

    sell_condition = (
        (data["Close"] < data["SMA_20"]) |
        (data["MACD"] < data["MACD_Signal"]) |
        (data["RSI_14"] > 75)
    )

    # Position: 1 (Invested) or 0 (Cash)
    position = pd.Series(index=data.index, dtype=float)
    current_pos = 0

    for idx, (is_buy, is_sell) in enumerate(zip(buy_condition, sell_condition)):
        if is_buy:
            current_pos = 1
        elif is_sell:
            current_pos = 0
        position.iloc[idx] = current_pos

    # Shift position by 1 day to prevent lookahead bias (trade executed on next bar open/close)
    data["Position"] = position.shift(1).fillna(0)
    data["Daily_Return"] = data["Close"].pct_change().fillna(0)

    # Strategy Daily Return
    data["Strategy_Daily_Return"] = data["Position"] * data["Daily_Return"]

    # Equity curves
    data["Buy_Hold_Equity"] = initial_capital * (1 + data["Daily_Return"]).cumprod()
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Daily_Return"]).cumprod()

    # Performance Metrics
    total_bh_return = (data["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1) * 100
    total_strat_return = (data["Strategy_Equity"].iloc[-1] / initial_capital - 1) * 100

    # Drawdown
    strat_peak = data["Strategy_Equity"].cummax()
    strat_dd = (data["Strategy_Equity"] - strat_peak) / strat_peak
    max_dd = strat_dd.min() * 100

    # Trade stats
    trade_entries = (data["Position"].diff() == 1).sum()
    daily_active_returns = data.loc[data["Position"] == 1, "Daily_Return"]
    winning_days = (daily_active_returns > 0).sum()
    total_active_days = len(daily_active_returns)
    win_rate = (winning_days / total_active_days * 100) if total_active_days > 0 else 50.0

    # Annualized Sharpe (assuming 252 trading days)
    mean_ret = data["Strategy_Daily_Return"].mean()
    std_ret = data["Strategy_Daily_Return"].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in data.index],
        "strategy_equity": [round(val, 2) for val in data["Strategy_Equity"].tolist()],
        "benchmark_equity": [round(val, 2) for val in data["Buy_Hold_Equity"].tolist()],
        "strategy_return_pct": round(total_strat_return, 2),
        "benchmark_return_pct": round(total_bh_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": int(trade_entries),
        "sharpe_ratio": round(sharpe, 2)
    }

