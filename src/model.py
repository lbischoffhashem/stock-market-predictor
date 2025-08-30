import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, fbeta_score, precision_recall_curve
import pandas as pd
import numpy as np

def load_and_prepare_data(*stock_tickers):
    # min_date="1990-01-01"
    # min_start_date = pd.Timestamp(min_date) 

    tickers = []
    for stock_ticker in stock_tickers:
        ticker_data = yf.Ticker(stock_ticker).history(period="max")

        ticker_data.index = pd.to_datetime(ticker_data.index)
        # ticker_data.index = ticker_data.index.tz_localize(None)
        # ticker_start_date = max(ticker_data.index.min(), "1990-01-01")
        ticker_data = ticker_data.loc["1990-01-01":].copy()


        del ticker_data["Dividends"]
        del ticker_data["Stock Splits"]

        ticker_data["Tomorrow"] = ticker_data["Close"].shift(-1)

        ticker_data["Target"] = (ticker_data["Tomorrow"] > ticker_data["Close"]).astype(int)

        tickers.append(ticker_data)

    return tickers


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    # preds = model.predict(test[predictors])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.57] = 1
    preds[preds <.57] = 0
    probs = model.predict_proba(test[predictors])[:, 1]
    probs = pd.Series(probs, index=test.index, name="Predicted_Prob")
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds, probs],axis=1)
    return combined

def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def add_predictor_columns(data):
    # Percentage change from open to close (daily return)
    '''data["OC_Pct_Change"] = (data["Close"] - data["Open"]) / data["Open"]

    # Volatility (Range as a percentage of the open)
    data["Daily_Volatility"] = (data["High"] - data["Low"]) / data["Open"]

    # Where did the price close relative to its daily range
    data["Close_Relative_To_Range"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])

    predictors = ["OC_Pct_Change","Daily_Volatility","Close_Relative_To_Range"]'''
    predictors = []
    horizons = [2,5,60,250,1000]

    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        '''exponential_moving_averages = data["Close"].ewm(span=horizon, adjust=False).mean()
        
        ema_column = f"Close_EMA_{horizon}"
        data[ema_column] = exponential_moving_averages

        EMA_ratio = f"Close_Ratio_EMA_{horizon}"
        data[EMA_ratio] = data["Close"] / data[ema_column]'''

        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        predictors += [ratio_column, trend_column]#, ema_column, EMA_ratio]

    return predictors

def split_training_testing_data(data):
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    return train, test

def optimize_fbeta_score(predictions):
    best_fbeta = 0
    best_beta = 0.25  
    best_thresh = 0

    for thresh in np.arange(0.25, 1.01, 0.01):
        y_pred = (predictions["Predicted_Prob"] >= thresh).astype(int)
        score = fbeta_score(predictions["Target"], y_pred, beta=best_beta, zero_division=0)
        
        if score > best_fbeta:
            best_fbeta = score
            best_thresh = thresh
    return best_thresh, best_fbeta

def precision_recall_analysis(predictions):
    precision_vals, recall_vals, thresholds = precision_recall_curve(predictions["Target"], predictions["Predicted_Prob"])
    
    max_precision_idx = np.argmax(precision_vals[:-1])
    max_precision_threshold = thresholds[max_precision_idx]

    max_recall_idx = np.argmax(recall_vals[:-1])
    max_recall_threshold = thresholds[max_recall_idx]
    
    max_beta_threshold, max_fbeta = optimize_fbeta_score(predictions)
    precision_at_max_beta = precision_score(predictions["Target"], (predictions["Predicted_Prob"] >= max_beta_threshold).astype(int), zero_division=0)
    
    return {
        "max_precision": (precision_vals[max_precision_idx], max_precision_threshold, calculate_trading_days(predictions, max_precision_threshold)),
        "max_recall": (recall_vals[max_recall_idx], max_recall_threshold, calculate_trading_days(predictions, max_recall_threshold)),
        "max_fbeta": (max_fbeta, max_beta_threshold, precision_at_max_beta, calculate_trading_days(predictions, max_beta_threshold))
    }

def calculate_trading_days(predictions, threshold):
    return (predictions["Predicted_Prob"] >= threshold).astype(int).sum()

# Temporary main for testing and development
if __name__ == "__main__":
    #Dow Jones Industrial Average, S&P 500, Nasdaq Composite, Russell 2000
    indexes = load_and_prepare_data("^DJI", "^GSPC", "^IXIC", "^RUT")

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, n_jobs=-1)

    for index in indexes:
        predictors = add_predictor_columns(index)
        index = index.dropna(subset=index.columns[index.columns != "Tomorrow"])
        train, test = split_training_testing_data(index)
        predictions = predict(train, test, predictors, model)
        print(precision_score(predictions["Target"], predictions["Predictions"]))
        predictions = backtest(index, model, predictors)
        print(precision_score(predictions["Target"], predictions["Predictions"]))


    
