@echo off
title Northgate AI Stock Predictor
echo =======================================================
echo          NORTHGATE AI STOCK PREDICTOR
echo     Intelligent Indian Equities Forecasting Suite
echo =======================================================
echo.
echo Starting Web Dashboard on http://localhost:8000 ...
echo.

start "" "http://localhost:8000"
python server.py

pause

