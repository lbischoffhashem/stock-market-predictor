import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import time
from database import load_ticker, update_ticker

from model import (
    load_and_prepare_data, 
    add_predictor_columns, 
    predict, 
    backtest,
    split_training_testing_data,
)

st.set_page_config(
    page_title="Index Fund Predictor",
    page_icon="📈",
    layout="wide"
)

def get_display_for_session(ticker: str, db_version: int, days=100):
    """Generates and caches model predictions and display data for selected ticker"""
    key = f"{ticker}__{db_version}__{days}"
    if "display_cache" not in st.session_state:
        st.session_state["display_cache"] = {}
    cache = st.session_state["display_cache"]
    if key in cache:
        return cache[key]

    #Displaying a loading message while model is training
    loading_placeholder = st.empty()
    loading_placeholder.info("Computing predictions...")

    df = load_fund_data(ticker, db_version)
    if df is None or df.empty:
        display = {"error": "no data"}
    else:
        if "Target" not in df.columns:
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        predictors = add_predictor_columns(df)
        df_clean = df.dropna(subset=predictors)

        #Training and backtesting model
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50,random_state=1, n_jobs=-1)
        model.fit(df_clean[predictors], df_clean["Target"])
        train, test = split_training_testing_data(df_clean)
        test_predictions = predict(train, test, predictors, model)
        historical_predictions = backtest(df_clean, model, predictors)

        #Testing precision
        test_precision = precision_score(test_predictions["Target"], test_predictions["Predictions"])
        historical_precision = precision_score(historical_predictions["Target"], historical_predictions["Predictions"])

        latest_data = df_clean.iloc[-1:].copy()
        next_day_row = predict(df_clean, latest_data, predictors, model).iloc[0]
        next_day_pred = int(next_day_row["Predictions"])
        next_day_prob = float(next_day_row["Predicted_Prob"])
        merged_data = pd.merge(historical_predictions, df[["Close"]], left_index=True, right_index=True, how='left')
        recent_predictions = merged_data.iloc[-days:].copy()

        #Creating calendar
        calendar_data = []
        for date, row in recent_predictions.iterrows():
            day_name = calendar.day_name[date.weekday()]
            day_num = date.day
            if row["Predictions"] == 1:
                if row["Target"] == 1:
                    color = "green"; result = "Correct ↑"
                else:
                    color = "red"; result = "Wrong ↓"
            else:
                color = "gray"; result = "No prediction"
            calendar_data.append({"date": date, "day": day_num, "day_name": day_name, "color": color, "result": result, "close": row["Close"]})
        display = {
            "df_close": df["Close"],
            "test_predictions": test_predictions,
            "historical_predictions": historical_predictions,
            "test_precision": test_precision,
            "historical_precision": historical_precision,
            "next_day_pred": next_day_pred,
            "next_day_prob": next_day_prob,
            "calendar_data": calendar_data
        }

    cache[key] = display

    #Remove loading message
    loading_placeholder.empty()  
    return display

#Caching data loading to improve speed
@st.cache_data 
def load_fund_data(ticker: str, db_version: int = 0) -> pd.DataFrame:
    """
    Load ticker from SQLite. If DB is empty for that ticker, 
    fall back to the load_and_prepare_data function and then incrementally 
    update DB via update_ticker.
    """
    #Trying to load from DB
    df = load_ticker(ticker)

    if df is None or df.empty:
        #Retreive data from yfinance store in DB
        fetched = load_and_prepare_data(ticker)
        df = fetched[0]
        #If database.update_ticker expects to insert new rows, call it to populate DB.
        try:
            update_ticker(ticker)
            #Reread from DB after update
            df = load_ticker(ticker)
        except Exception:
            #If DB write fails continue using data from yfinance
            pass

    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    return df

def auto_update_db_for_ticker(ticker):
    """
    Automatically attempt an incremental update when ticker is selected for the first time in a session.
    Returns number of rows added (0 if none or failed). Sets session flag.
    """
    key = f"_auto_updated_{ticker}"
    if st.session_state.get(key):
        return 0

    #Using a spinner to show user that updating is in progress
    with st.spinner("Checking DB and fetching any missing days from Yahoo Finance..."):
        try:
            added = update_ticker(ticker)
        except Exception as e:
            st.warning(f"Automatic DB update failed: {e}")
            added = 0

    #Avoids repeating for each rerun
    st.session_state[key] = True

    #Bumping so load_fund_data rereads from DB if rows were added
    if added:
        st.session_state.db_version = int(time.time())

    return added

index_funds = {
    "S&P 500 (^GSPC)": {
        "ticker": "^GSPC",
        "description": "The S&P 500 is a stock market index tracking the stock performance of 500 large companies listed on stock exchanges in the United States."
    },
    "Dow Jones Industrial Average (^DJI)": {
        "ticker": "^DJI",
        "description": "The Dow Jones Industrial Average is a price-weighted measurement stock market index of 30 prominent companies listed on stock exchanges in the United States."
    },
    "NASDAQ Composite (^IXIC)": {
        "ticker": "^IXIC",
        "description": "The NASDAQ Composite is a stock market index that includes almost all stocks listed on the NASDAQ stock exchange."
    },
    "Russell 2000 (^RUT)": {
        "ticker": "^RUT",
        "description": "The Russell 2000 Index is a small-cap stock market index of the smallest 2,000 companies in the Russell 3000 Index."
    }
}


def main():
    st.title("Index Fund Predictor")
    
    #Creating sidebar for navigation
    st.sidebar.header("Navigation")
    selected_fund_name = st.sidebar.selectbox(
        "Select Index Fund",
        list(index_funds.keys())
    )
    
    #Getting selected fund data
    selected_fund = index_funds[selected_fund_name]
    ticker = selected_fund["ticker"]
    
    #Ensuring the cache token exists
    if "db_version" not in st.session_state:
        st.session_state.db_version = 0

    #Always auto update DB end once per session per ticker
    auto_update_db_for_ticker(ticker)

    display_data = get_display_for_session(ticker, st.session_state.db_version)
    if "error" in display_data:
        st.error(f"Error loading data: {display_data['error']}")
        return
    
    st.header(selected_fund_name)
    st.write(selected_fund["description"])
    
    st.subheader("Price History")
    st.line_chart(display_data["df_close"])
    
    st.subheader("Tomorrow's Prediction")
    if display_data["next_day_pred"] == 1:
        st.success(f"The model predicts the price will increase with {display_data['next_day_prob']:.2%} confidence")
    else:
        st.info("The model does not predict a price increase for tomorrow")
    
    #Displaying model accuracy
    st.subheader("Model Performance")
    st.write(f"Test Set Precision (last 100 days): {display_data['test_precision']:.2%}")
    st.write(f"Historical Precision (backtest): {display_data['historical_precision']:.2%}")

    #Creating and displaying calendar
    st.subheader("Recent Performance Calendar")
    calendar_data = display_data["calendar_data"]
    for i in range(0, len(calendar_data), 7):
        week_data = calendar_data[i:i+7]
        cols = st.columns(7)
        
        for j, day_data in enumerate(week_data):
            with cols[j]:
                #Displays days with color coded backgrounds
                day_str = f"""
                <div style="background-color: {day_data['color']}; 
                            padding: 10px; 
                            border-radius: 5px; 
                            text-align: center;
                            color: white;">
                    <b>{day_data['day_name']}</b><br>
                    <b>{day_data['day']}</b><br>
                    {day_data['result']}<br>
                    ${day_data['close']:.2f}
                </div>
                """
                st.markdown(day_str, unsafe_allow_html=True)
    
    #Legend
    st.markdown("""
    **Color Legend:**
    - <span style="color:green">Green</span>: Correct prediction of price increase
    - <span style="color:red">Red</span>: Incorrect prediction of price increase
    - <span style="color:gray">Gray</span>: Did not predict a price increase
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()