# 📈 Northgate AI Stock Predictor

> **AI-Based Stock Market Prediction and Portfolio Optimization System using Python, Machine Learning, Deep Learning, Sentiment Analysis and Streamlit.**

[![Live Demo on Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://northgate-ai-stock-predictor.vercel.app)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visuals-Plotly.js-3F4F75?style=for-the-badge&logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

**Northgate AI Stock Predictor** is an enterprise-grade quantitative finance platform engineered for intelligent stock market forecasting, momentum analysis, and risk-managed trading signals. Designed with a primary focus on Indian Equities (NSE / BSE) and major benchmark indices (NIFTY 50, SENSEX, BANK NIFTY), the platform fuses **Machine Learning Time-Series Ensembles**, **Technical Indicator Momentum**, and **Real-Time Financial News Sentiment** into an actionable **Northgate Trade Recommendation**.

---

## 🚀 Key Features

### 1. 🇮🇳 Comprehensive Indian Equities Support
- **NIFTY 50 & SENSEX Leaders**: Pre-configured tickers including Reliance Industries (`RELIANCE.NS`), TCS (`TCS.NS`), HDFC Bank (`HDFCBANK.NS`), Infosys (`INFY.NS`), Tata Motors (`TATAMOTORS.NS`), SBI (`SBIN.NS`), and benchmark indices (`^NSEI`, `^BSESN`, `^NSEBANK`).
- **Dynamic Search**: Instantly query any NSE (`.NS`), BSE (`.BO`), or global ticker.

### 2. 🤖 Hybrid AI Prediction Engine
- **Time-Series ML Ensemble**: Feature engineering on lagged returns, moving average distances, volatility ratios, and momentum oscillators.
- **Multi-Horizon Forecasts**: Directional probability (Bullish / Bearish / Neutral) and expected returns for **1-Day, 5-Day (1W), 10-Day (2W), and 20-Day (1M)** periods.
- **Confidence Corridors**: Dynamic upper and lower price boundaries adjusted for Average True Range (ATR) and model residuals.

### 3. 📈 Quantitative Technical Analysis
- **Trend Indicators**: Simple Moving Averages (SMA 20, 50, 200), Exponential Moving Averages (EMA 12, 26).
- **Momentum & Volatility**: Relative Strength Index (RSI 14) with overbought/oversold boundaries, MACD & Signal Line with momentum histogram, Bollinger Bands (20, 2), and Average True Range (ATR 14).
- **Support & Resistance**: Automatic pivot calculations (Central Pivot, R1, R2, S1, S2).

### 4. 📰 Real-Time News & Sentiment Engine
- **Live Financial RSS Aggregator**: Streams breaking headlines from Google News India and top financial publications.
- **Lexical Sentiment Classifier**: Analyzes financial sentiment with categorical tagging (Bullish, Neutral, Bearish) and sentiment impact weighting.

### 5. 🎯 Northgate Unified Trade Signals
- **Actionable Decisions**: Synthesizes Technicals (40%) + ML (40%) + Sentiment (20%) into **`STRONG BUY`**, **`BUY`**, **`HOLD`**, **`SELL`**, or **`STRONG SELL`**.
- **Risk Management**: Computes dynamic **Target 1**, **Target 2**, **Stop-Loss**, and **Risk-to-Reward Ratio**.

### 6. ⚙️ Walk-Forward Strategy Backtesting
- Simulates the Northgate strategy against a Buy-and-Hold benchmark across historical price data.
- Reports **Total Return (%)**, **Benchmark Return (%)**, **Win Rate (%)**, **Maximum Drawdown (%)**, and **Sharpe Ratio**.

---

## 🏗️ System Architecture

Northgate AI Stock Predictor provides a dual-interface architecture:

```
┌────────────────────────────────────────────────────────┐
│                   Data Ingestion                       │
│    Yahoo Finance API  •  Google Financial News RSS     │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                    Hybrid AI Engine                    │
│   Quant Indicators  •  ML Ensembles  •  NLP Sentiment  │
│             Strategy Synthesis & Backtesting           │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
┌─────────────▼────────────┐  ┌────────────▼─────────────┐
│    FastAPI Web Server    │  │   Streamlit Dashboard    │
│  Plotly.js + Tailwind UI │  │   Interactive Analytics  │
│  (Vercel Cloud Deploy)   │  │   (Local Research Hub)   │
└──────────────────────────┘  └──────────────────────────┘
```

---

## 📂 Project Structure

```
NORTHGATE AI STOCK PREDICTOR/
│
├── api/
│   └── index.py            # Vercel ASGI Serverless Entrypoint
├── templates/
│   └── index.html          # High-performance FinTech Dark UI with Plotly.js
├── server.py               # FastAPI application (/api/health, /api/tickers, /api/analysis)
├── core_engine.py          # Pure-Python quant indicators, ML forecasting, and backtesting
├── app.py                  # Streamlit Interactive Dashboard
├── config.py               # Tickers, timeframes, and sentiment keywords
├── ml_engine.py            # XGBoost & Random Forest time-series training pipeline
├── indicators.py           # Technical indicators calculation engine
├── data_loader.py          # Data ingestion and caching module
├── news_sentiment.py       # Financial news scraper and sentiment scorer
├── strategy.py             # Hybrid fusion recommendation generator
├── backtester.py           # Historical strategy backtest simulator
├── run.bat                 # One-click Windows application launcher
├── requirements.txt        # Complete project dependencies
├── requirements-serverless.txt # Lean dependencies for serverless deployment
├── .vercelignore           # Excluded files for Vercel builds
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variables template
├── DEPLOYMENT.md           # Cloud deployment documentation
└── README.md               # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/amulyaaa/northgate-ai-stock-predictor.git
cd northgate-ai-stock-predictor
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Modern FinTech Web Interface (FastAPI + Plotly.js)
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> On Windows, you can also double-click **`run.bat`** for one-click launch!

### Option B: Streamlit Analytical Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🌐 Cloud Deployment

The FastAPI application and FinTech terminal are deployed on **Vercel Serverless**:
- **Live Production URL**: [https://northgate-ai-stock-predictor.vercel.app](https://northgate-ai-stock-predictor.vercel.app)
- **API Health Check**: [https://northgate-ai-stock-predictor.vercel.app/api/health](https://northgate-ai-stock-predictor.vercel.app/api/health)

For complete cloud deployment instructions, refer to [`DEPLOYMENT.md`](./DEPLOYMENT.md).

---

## 🔒 Security & Best Practices

- **Zero Hardcoded Secrets**: All configurations are managed cleanly via environment variables.
- **Safe Version Control**: Virtual environments, secrets, temporary caches, and local binaries are excluded via `.gitignore`.

---

## ⚠️ Disclaimer

*Northgate AI Stock Predictor is developed strictly for research, educational, and analytical purposes. Stock and equity markets carry financial risk. Always perform your own due diligence or consult a SEBI-registered financial advisor before making investment decisions.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
