"""
Northgate AI Stock Predictor - Interactive Dashboard
Streamlit + Plotly + Machine Learning + Technical Analysis + News Sentiment
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import DEFAULT_INDIAN_TICKERS, TIME_PERIODS, INTERVALS, FORECAST_HORIZONS
from data_loader import fetch_stock_history, fetch_stock_info
from indicators import add_technical_indicators, compute_support_resistance
from news_sentiment import fetch_stock_news, compute_aggregate_sentiment
from ml_engine import train_and_forecast
from strategy import generate_technical_signal, synthesize_northgate_recommendation
from backtester import run_backtest


# Page Configuration
st.set_page_config(
    page_title="Northgate AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Fintech Dark Aesthetic)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #161f30 0%, #111726 100%);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #24324d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .signal-box {
        padding: 18px 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151d2d;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Northgate AI")
st.sidebar.caption("Intelligent Indian Equities Forecasting & Quant Platform")

# 1. Ticker Selector
ticker_options = list(DEFAULT_INDIAN_TICKERS.keys())
selected_preset = st.sidebar.selectbox(
    "Select Indian Ticker / Index:",
    ticker_options,
    index=0,
    format_func=lambda x: f"{x} - {DEFAULT_INDIAN_TICKERS[x]}"
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Custom Symbol (e.g. TATACHEM, SBICARD, AAPL):",
    value=""
)
target_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_preset

# Ensure Indian extension if entered bare
if not target_ticker.startswith("^") and "." not in target_ticker:
    target_ticker = f"{target_ticker}.NS"

# 2. Timeframe & Interval
selected_period_label = st.sidebar.selectbox("Historical Lookback:", list(TIME_PERIODS.keys()), index=4)  # 2 Years
selected_period = TIME_PERIODS[selected_period_label]

# 3. Forecast Horizon
horizon_label = st.sidebar.selectbox("Prediction Horizon:", list(FORECAST_HORIZONS.keys()), index=1)  # 5 Days
forecast_days = FORECAST_HORIZONS[horizon_label]

# 4. Chart Overlay Toggles
st.sidebar.markdown("---")
st.sidebar.subheader("Chart Overlays")
show_sma20 = st.sidebar.checkbox("SMA 20 (Short-Term)", value=True)
show_sma50 = st.sidebar.checkbox("SMA 50 (Medium-Term)", value=True)
show_sma200 = st.sidebar.checkbox("SMA 200 (Long-Term)", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", value=False)
show_sr = st.sidebar.checkbox("Support & Resistance Levels", value=True)

# Main Dashboard Execution
company_info = fetch_stock_info(target_ticker)
company_name = company_info.get("shortName", target_ticker)

# Title & Top Header
col_header, col_clock = st.columns([3, 1])
with col_header:
    st.title(f"{company_name} ({target_ticker})")
    st.caption(f"Sector: {company_info.get('sector')} | Industry: {company_info.get('industry')} | Currency: {company_info.get('currency', 'INR')}")
with col_clock:
    st.markdown(f"<div style='text-align: right; color: #94a3b8; padding-top: 15px;'>Engine: <b>Hybrid XGBoost + Quant</b></div>", unsafe_allow_html=True)

# Fetch Data
with st.spinner(f"Loading market data for {target_ticker}..."):
    raw_df, err = fetch_stock_history(target_ticker, period=selected_period, interval="1d")

if err or raw_df is None or raw_df.empty:
    st.error(f"Error fetching data: {err}")
    st.stop()

# Compute Indicators
df = add_technical_indicators(raw_df)
sr_levels = compute_support_resistance(df)

# Key Price Metrics Bar
current_price = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
price_change = current_price - prev_close
price_change_pct = (price_change / prev_close) * 100 if prev_close else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    delta_color = "normal" if price_change >= 0 else "inverse"
    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
        f"{price_change:+,.2f} ({price_change_pct:+.2f}%)"
    )
with m2:
    st.metric("Day High / Low", f"₹{df['High'].iloc[-1]:,.2f}", f"Low: ₹{df['Low'].iloc[-1]:,.2f}")
with m3:
    st.metric("52-Week High / Low", f"₹{df['High'].max():,.2f}", f"Low: ₹{df['Low'].min():,.2f}")
with m4:
    vol_str = f"{df['Volume'].iloc[-1]:,.0f}"
    st.metric("Volume", vol_str)
with m5:
    pe_val = company_info.get("trailingPE")
    pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A"
    st.metric("P/E Ratio", pe_str)

st.markdown("---")

# Run ML & News Engine
with st.spinner("Executing Northgate AI Model & News Sentiment Analysis..."):
    ml_result = train_and_forecast(df, horizon_days=forecast_days)
    news_items = fetch_stock_news(target_ticker, company_name)
    sentiment_result = compute_aggregate_sentiment(news_items)
    tech_signal = generate_technical_signal(df)
    recommendation = synthesize_northgate_recommendation(tech_signal, ml_result, sentiment_result, sr_levels)

# Northgate AI Recommendation Banner
c_signal, c_targets, c_conviction = st.columns([1.5, 2, 1.5])
with c_signal:
    st.markdown(
        f"""
        <div class="signal-box" style="background-color: {recommendation['action_color']};">
            <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Northgate AI Signal</span>
            <div style="font-size: 32px; font-weight: 800; margin: 4px 0;">{recommendation['action']}</div>
            <div style="font-size: 14px; opacity: 0.9;">Horizon: {horizon_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_targets:
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"<div class='metric-card'><span style='color:#94a3b8;font-size:12px;'>STOP LOSS</span><div style='font-size:20px;font-weight:700;color:#ff5252;'>₹{recommendation['stop_loss']:,.2f}</div></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='metric-card'><span style='color:#94a3b8;font-size:12px;'>TARGET 1</span><div style='font-size:20px;font-weight:700;color:#4caf50;'>₹{recommendation['target_1']:,.2f}</div></div>", unsafe_allow_html=True)
    with t3:
        st.markdown(f"<div class='metric-card'><span style='color:#94a3b8;font-size:12px;'>TARGET 2</span><div style='font-size:20px;font-weight:700;color:#00e676;'>₹{recommendation['target_2']:,.2f}</div></div>", unsafe_allow_html=True)

with c_conviction:
    st.markdown(
        f"""
        <div class="metric-card" style="text-align: center;">
            <span style="color:#94a3b8;font-size:12px;">MODEL CONVICTION & RISK</span>
            <div style="font-size:22px; font-weight:bold; color: #38bdf8; margin: 4px 0;">{recommendation['confidence_pct']}% Conviction</div>
            <div style="font-size:13px; color:#cbd5e1;">Risk/Reward: <b>{recommendation['risk_reward_ratio']}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Interactive Tabs
tab_charts, tab_tech, tab_ml, tab_news, tab_backtest = st.tabs([
    "📊 Price & AI Projection",
    "📈 Technical Indicators",
    "🤖 ML Model Diagnostics",
    "📰 News & Sentiment",
    "⚙️ Strategy Backtesting"
])

# ----------------- TAB 1: CHARTS & PROJECTIONS -----------------
with tab_charts:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{company_name} Price Action & AI Forecast Corridor", "Volume")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Moving Averages
    if show_sma20 and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    if show_sma50 and "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
    if show_sma200 and "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], mode="lines", name="SMA 200", line=dict(color="#ec4899", width=2)), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="BB Lower", fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)), row=1, col=1)

    # Support & Resistance
    if show_sr:
        last_date = df.index[-1]
        fig.add_hline(y=sr_levels["resistance_1"], line_dash="dash", line_color="#ef4444", annotation_text=f"R1: ₹{sr_levels['resistance_1']}", row=1, col=1)
        fig.add_hline(y=sr_levels["support_1"], line_dash="dash", line_color="#10b981", annotation_text=f"S1: ₹{sr_levels['support_1']}", row=1, col=1)

    # Future Projection Cone
    last_dt = df.index[-1]
    future_dt = last_dt + pd.Timedelta(days=int(forecast_days * 1.4))  # Account for weekends roughly
    
    # Projection line from current price to forecast price
    fig.add_trace(
        go.Scatter(
            x=[last_dt, future_dt],
            y=[current_price, ml_result["forecast_price"]],
            mode="lines+markers",
            name=f"AI Forecast ({horizon_label})",
            line=dict(color="#38bdf8", width=3, dash="dot"),
            marker=dict(size=[4, 8], color=["#38bdf8", "#38bdf8"])
        ),
        row=1, col=1
    )

    # Forecast Target Range Box
    fig.add_trace(
        go.Scatter(
            x=[last_dt, future_dt, future_dt, last_dt],
            y=[current_price, ml_result["upper_bound"], ml_result["lower_bound"], current_price],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(56, 189, 248, 0.3)"),
            name="Confidence Interval",
            hoverinfo="skip"
        ),
        row=1, col=1
    )

    # Volume Bar Chart
    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.8),
        row=2, col=1
    )
    if "Vol_SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Vol_SMA_20"], mode="lines", name="Vol SMA 20", line=dict(color="#f59e0b", width=1)),
            row=2, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=20, r=20, t=40, b=20),
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Projection summary box below chart
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.info(f"📍 **Expected Price ({horizon_label})**: ₹{ml_result['forecast_price']:,.2f}")
    with p2:
        st.info(f"📈 **Expected Return**: {ml_result['predicted_return_pct']:+,.2f}%")
    with p3:
        st.info(f"🎯 **Lower Bound**: ₹{ml_result['lower_bound']:,.2f}")
    with p4:
        st.info(f"🚀 **Upper Bound**: ₹{ml_result['upper_bound']:,.2f}")

# ----------------- TAB 2: TECHNICAL INDICATORS -----------------
with tab_tech:
    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        # RSI Subplot
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode="lines", name="RSI (14)", line=dict(color="#a855f7", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            title="Relative Strength Index (RSI 14)", height=240, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Subplot
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="#38bdf8", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal", line=dict(color="#f97316", width=1.5)))
        macd_colors = ["#26a69a" if val >= 0 else "#ef5350" for val in df["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=macd_colors))
        fig_macd.update_layout(
            template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            title="Moving Average Convergence Divergence (MACD)", height=240, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_macd, use_container_width=True)

    with col_t2:
        st.subheader("Technical Signal Breakdown")
        for sig in tech_signal["signals"]:
            st.markdown(f"✔ **{sig}**")

        st.markdown("---")
        st.subheader("Pivot Points & Key Levels")
        st.write(f"• **Resistance 2 (R2)**: ₹{sr_levels['resistance_2']:,.2f}")
        st.write(f"• **Resistance 1 (R1)**: ₹{sr_levels['resistance_1']:,.2f}")
        st.write(f"• **Central Pivot (P)**: ₹{sr_levels['pivot']:,.2f}")
        st.write(f"• **Support 1 (S1)**: ₹{sr_levels['support_1']:,.2f}")
        st.write(f"• **Support 2 (S2)**: ₹{sr_levels['support_2']:,.2f}")

# ----------------- TAB 3: ML MODEL DIAGNOSTICS -----------------
with tab_ml:
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("Model Backtest Accuracy", f"{ml_result['test_accuracy']}%", "Out-of-sample testing")
    with c_m2:
        st.metric("Model Precision", f"{ml_result['test_precision']}%", "True Positive Ratio")
    with c_m3:
        st.metric("Bullish Direction Probability", f"{ml_result['bullish_probability']}%", f"Trend: {ml_result['trend']}")

    st.markdown("---")
    st.subheader("Top Predictive Features (Feature Importance)")
    imp_df = pd.DataFrame(ml_result["feature_importances"])
    if not imp_df.empty:
        fig_imp = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(color="#38bdf8")
        ))
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- TAB 4: NEWS & SENTIMENT -----------------
with tab_news:
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        st.subheader("Sentiment Summary")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sentiment Index: <span style="color:#38bdf8;">{sentiment_result['sentiment_label']}</span></h4>
            <p>Score: <b>{sentiment_result['average_score']}</b> (Scale: -1.0 to +1.0)</p>
            <p>Impact Level: <b>{sentiment_result['impact']}</b></p>
            <hr style="border-color:#24324d;">
            <p>🟢 Bullish Articles: <b>{sentiment_result['bullish_count']}</b></p>
            <p>⚪ Neutral Articles: <b>{sentiment_result['neutral_count']}</b></p>
            <p>🔴 Bearish Articles: <b>{sentiment_result['bearish_count']}</b></p>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.subheader(f"Recent Financial News: {company_name}")
        for item in news_items:
            tag_color = "#4caf50" if item["sentiment_label"] == "Bullish" else "#f44336" if item["sentiment_label"] == "Bearish" else "#9e9e9e"
            st.markdown(
                f"""
                <div style="background-color: #161f30; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {tag_color};">
                    <div style="font-weight: 600; font-size: 15px;">
                        <a href="{item['link']}" target="_blank" style="color: #f1f5f9; text-decoration: none;">{item['title']}</a>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                        Source: {item['source']} &bull; {item['published']} &bull; Sentiment: <b style="color:{tag_color};">{item['sentiment_label']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------- TAB 5: STRATEGY BACKTESTING -----------------
with tab_backtest:
    st.subheader("Northgate AI Strategy vs Benchmark (Buy & Hold)")
    st.caption("Chronological backtest over the selected historical period without lookahead bias.")

    with st.spinner("Simulating backtest..."):
        try:
            bt_results = run_backtest(df)

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.metric("Strategy Return", f"{bt_results['strategy_return_pct']:+,.2f}%", f"Benchmark: {bt_results['benchmark_return_pct']:+,.2f}%")
            with b_col2:
                st.metric("Win Rate", f"{bt_results['win_rate_pct']}%", f"Total Trades: {bt_results['total_trades']}")
            with b_col3:
                st.metric("Max Drawdown", f"{bt_results['max_drawdown_pct']:.2f}%", "Risk preservation")
            with b_col4:
                st.metric("Sharpe Ratio", f"{bt_results['sharpe_ratio']:.2f}", "Risk-adjusted return")

            # Equity Curve Chart
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt_results["dates"],
                y=bt_results["strategy_equity"],
                mode="lines",
                name="Northgate AI Strategy",
                line=dict(color="#10b981", width=2.5)
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_results["dates"],
                y=bt_results["benchmark_equity"],
                mode="lines",
                name="Buy & Hold Benchmark",
                line=dict(color="#94a3b8", width=1.5, dash="dash")
            ))
            fig_bt.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                title="Portfolio Equity Growth (Initial: ₹100,000)",
                yaxis_title="Portfolio Value (₹)",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not complete backtest: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px;'>"
    "⚠️ <b>Disclaimer:</b> Northgate AI Stock Predictor is built for research and educational purposes. "
    "Stock markets involve risk of capital loss. Past performance does not guarantee future results."
    "</div>",
    unsafe_allow_html=True
)

