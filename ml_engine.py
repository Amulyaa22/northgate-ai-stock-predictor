"""
Machine Learning engine for stock trend prediction and price forecasting.
Uses Time-Series feature engineering with XGBoost / Random Forest and chronological cross-validation.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_percentage_error
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURE_COLUMNS = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "Stoch_K", "Stoch_D", "Volatility_20",
    "Dist_SMA_20", "Dist_SMA_50", "Dist_SMA_200",
    "Vol_Ratio", "HL_Ratio", "ATR_Ratio",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d"
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts predictive technical and momentum features from historical data.
    """
    data = df.copy()

    # Relative distances to key moving averages
    data["Dist_SMA_20"] = (data["Close"] - data["SMA_20"]) / data["SMA_20"]
    data["Dist_SMA_50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"]
    data["Dist_SMA_200"] = (data["Close"] - data["SMA_200"]) / data["SMA_200"]

    # Volume & Price Action Ratios
    data["Vol_Ratio"] = data["Volume"] / data["Vol_SMA_20"].replace(0, np.nan)
    data["HL_Ratio"] = (data["High"] - data["Low"]) / data["Close"]
    data["ATR_Ratio"] = data["ATR_14"] / data["Close"]

    # Lagged Returns
    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_3d"] = data["Close"].pct_change(3)
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Return_10d"] = data["Close"].pct_change(10)

    return data


def train_and_forecast(
    df: pd.DataFrame,
    horizon_days: int = 5,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Trains ML models on chronological historical data and forecasts future trend and target price.
    """
    feat_df = prepare_features(df)

    # Define targets
    # Future return over horizon_days
    future_return = feat_df["Close"].shift(-horizon_days) / feat_df["Close"] - 1.0
    feat_df["Target_Return"] = future_return
    feat_df["Target_Direction"] = (future_return > 0).astype(int)

    # Clean dataset for training (exclude the most recent horizon_days rows which don't have future targets)
    train_pool = feat_df.dropna(subset=FEATURE_COLUMNS + ["Target_Return", "Target_Direction"])

    if len(train_pool) < 60:
        raise ValueError("Not enough historical data points to train machine learning models reliably (minimum 60 trading days required).")

    X = train_pool[FEATURE_COLUMNS]
    y_reg = train_pool["Target_Return"]
    y_clf = train_pool["Target_Direction"]

    # Chronological Time-Series Split
    split_idx = int(len(train_pool) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

    # Train Classifier (Directional Probability)
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    clf.fit(X_train, y_train_clf)
    y_pred_clf = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test_clf, y_pred_clf)
    test_precision = precision_score(y_test_clf, y_pred_clf, zero_division=0)

    # Train Regressor (Expected Magnitude of Return)
    if HAS_XGB:
        reg = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    reg.fit(X_train, y_train_reg)
    y_pred_reg = reg.predict(X_test)

    # Residual standard deviation for confidence intervals
    residuals = y_test_reg - y_pred_reg
    residual_std = float(residuals.std()) if not residuals.empty else 0.03

    # Feature Importance
    importances = clf.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(FEATURE_COLUMNS, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Predict for Current / Latest Market Bar
    latest_features = feat_df[FEATURE_COLUMNS].iloc[[-1]]
    if latest_features.isna().any().any():
        latest_features = latest_features.bfill().ffill().fillna(0)

    # Direction probability
    prob_up = float(clf.predict_proba(latest_features)[0][1])
    predicted_return = float(reg.predict(latest_features)[0])

    current_price = float(df["Close"].iloc[-1])
    forecast_price = current_price * (1 + predicted_return)

    # Volatility & ATR adjusted confidence bands
    recent_atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns and not pd.isna(df["ATR_14"].iloc[-1]) else current_price * 0.02
    band_margin = (residual_std * np.sqrt(horizon_days / 5) * current_price) + (recent_atr * 0.5)

    lower_bound = max(0.0, forecast_price - band_margin)
    upper_bound = forecast_price + band_margin

    trend = "BULLISH" if prob_up >= 0.55 else "BEARISH" if prob_up <= 0.45 else "NEUTRAL"

    return {
        "current_price": round(current_price, 2),
        "horizon_days": horizon_days,
        "forecast_price": round(forecast_price, 2),
        "predicted_return_pct": round(predicted_return * 100, 2),
        "bullish_probability": round(prob_up * 100, 1),
        "trend": trend,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "test_accuracy": round(test_accuracy * 100, 1),
        "test_precision": round(test_precision * 100, 1),
        "feature_importances": feat_imp[:8]
    }

