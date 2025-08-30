# Stock Market Predictor

[![Live Streamlit Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?logo=streamlit)](https://scalping-stock-market-predictor.streamlit.app/)

A machine learning model predicting stock price increases for the purpose of scalping, with a database-backed cache, backtesting, and interactive performance visualizations.

## Live demo

Try the app live: **[Open the Stock Market Predictor](https://scalping-stock-market-predictor.streamlit.app/)**.

> Publicly hosted on Streamlit.

## Features
- Random Forest predictions for index daily moves  
- SQLite cache of Yahoo Finance data (auto-updates missing days)  
- Backtesting, precision metrics, and a calendar view of prediction accuracy  
- Lightweight Streamlit UI for quick interactive use

## Backtesting & evaluation
A walk-forward backtest retrains the model on past data and evaluates on the next time block, which avoids look-ahead bias and provides realistic performance metrics. The app reports precision and highlights correct vs. incorrect predictions in a calendar view.

## Data split & model choice
Data is split in a time-series fashion: older data is used for training, and the most recent 100 days are reserved for testing (rather than a typical 70/30 split). Random Forest was chosen for its speed and robustness, handling non-linear relationships well with engineered rolling/window features.

## File Overview
- `apptest.py` — main Streamlit app  
- `model.py` — data preparation, predictor features, Random Forest logic, backtesting  
- `database.py` — SQLite caching, incremental updates  
- `notebook/` — optional exploratory notebooks  

## Local Setup

### Installation

To set up this project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/lbischoffhashem/stock-market-predictor
   cd stock-market-predictor

2. **Install the following**
- JupyterLab
- Python 3.8+
- Required libraries in `requirements.txt`:
  - To install, run:
  ```bash
  pip install -r requirements.txt
