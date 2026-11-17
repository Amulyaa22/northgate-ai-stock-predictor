# 🚀 Northgate AI Stock Predictor — Vercel Deployment Guide

This documentation details the production deployment architecture for **Northgate AI Stock Predictor** hosted on **Vercel Serverless** directly from your local environment without using GitHub or any Git provider.

---

## 1. 🏗️ Architecture & Hosting Overview

Northgate AI Stock Predictor is structured as a dual-interface financial engineering platform:

| Component | Framework | Vercel Compatibility | Deployment Strategy |
| :--- | :--- | :--- | :--- |
| **Production Web UI & API** | FastAPI + Plotly.js + Tailwind CSS | ✅ **100% Native Vercel Serverless** | Deployed via `api/index.py` with `@vercel/python` |
| **Quant & Predictive Engine** | Pure-Python Time-Series Ensemble | ✅ **100% Native Vercel Serverless** | Bundled in `core_engine.py` (ultra-fast cold start, <15MB) |
| **Market Data Pipeline** | Yahoo Finance / Google RSS API | ✅ **100% Native Vercel Serverless** | Real-time stateless JSON & RSS fetcher |
| **Research Dashboard** | Streamlit (`app.py`) | ⚠️ **Requires Persistent Container** | Maintained locally or on Streamlit Community Cloud / Railway / Render |

> [!IMPORTANT]
> **Streamlit Serverless Limitation:**
> Streamlit requires an uninterrupted, stateful WebSocket connection and a long-lived Python background process with an in-memory session manager. Because Vercel functions are stateless HTTP request-response handlers with cold starts and timeout boundaries, Streamlit cannot run natively as an ordinary Vercel serverless function. 
> To provide an uninterrupted cloud experience, the **FastAPI Web Terminal** (`templates/index.html` + `server.py`) serves the full interactive dashboard and all quantitative features natively on Vercel, while your local Streamlit app (`app.py`) remains preserved for local research.

---

## 2. 📁 Vercel Deployment Structure

```
NORTHGATE AI STOCK PREDICTOR/
│
├── api/
│   └── index.py            # Vercel Serverless ASGI entry point
├── templates/
│   └── index.html          # High-performance FinTech Dark UI with Plotly.js
├── server.py               # FastAPI application (/api/health, /api/tickers, /api/analysis)
├── core_engine.py          # Pure-Python quant indicators, ML forecasting, news scraper
├── config.py               # Tickers and market parameters
├── vercel.json             # Vercel URL rewrite rules routing all traffic to api/index.py
├── requirements.txt        # Lean serverless Python dependencies (fastapi, feedparser, pydantic)
├── requirements-local.txt  # Full local stack (Streamlit, XGBoost, Scikit-learn, Pandas)
├── .vercelignore           # Excludes .venv, cache, and test scripts
├── .gitignore              # Git ignore rules
└── .env.example            # Environment variables template
```

---

## 3. 🛠️ Step-by-Step Vercel CLI Deployment (No GitHub)

### Step 1: Install Vercel CLI
If not already installed:
```powershell
npm install -g vercel
```
Verify:
```powershell
vercel --version
```

### Step 2: Authenticate Without GitHub
Run:
```powershell
vercel login
```
Select **Continue with Email** or authenticate via the browser device authorization URL (e.g. `https://vercel.com/oauth/device`). **Do not select GitHub.**

### Step 3: Link Project to Vercel
In the project directory:
```powershell
vercel link
```
- Select your Vercel personal account or team.
- Link to an existing project or create a new one (e.g., `northgate-ai-stock-predictor`).

### Step 4: Create Preview Deployment
Deploy and test without affecting production:
```powershell
vercel deploy
```
Inspect the build output and verify the preview URL:
- Health check: `https://<preview-domain>/api/health`
- Tickers API: `https://<preview-domain>/api/tickers`
- Main UI: `https://<preview-domain>/`

### Step 5: Deploy to Production
```powershell
vercel deploy --prod
```

---

## 4. 🔒 Environment Variables & Security

Configure environment variables in your Vercel Project Dashboard under **Settings > Environment Variables**, or via the CLI:
```powershell
vercel env add GEMINI_API_KEY production
```

Available variables:
- `ENV`: `production`
- `DEBUG`: `false`
- `GEMINI_API_KEY`: *(Optional)* For advanced LLM fundamental insights.

---

## 5. 🌐 Live Production Deployment

- **Production URL**: [https://northgate-ai-stock-predictor.vercel.app](https://northgate-ai-stock-predictor.vercel.app)
- **Deployment Status**: Active & Verified (HTTP 200)
- **Health Check**: `https://northgate-ai-stock-predictor.vercel.app/api/health`
- **Tickers Endpoint**: `https://northgate-ai-stock-predictor.vercel.app/api/tickers`
- **Analysis Endpoint**: `https://northgate-ai-stock-predictor.vercel.app/api/analysis?ticker=RELIANCE.NS&period=3mo&horizon=5`

---

## 6. 🔄 How to Push Future Updates (Without GitHub)
- `GEMINI_API_KEY`: *(Optional)* For advanced LLM fundamental insights.

> [!NOTE]
> `.env` files are excluded from deployment by `.vercelignore` to prevent leaking secrets.

---

## 5. 🔄 How to Push Future Updates (Without GitHub)

Whenever you make changes to your codebase locally, simply run:
```powershell
vercel deploy --prod
```
The Vercel CLI will package your local files and deploy them directly to your live production URL in seconds without touching GitHub.

