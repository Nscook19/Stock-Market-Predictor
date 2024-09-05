import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load S&P 500 data
sp500 = yf.Ticker("^GSPC").history(period="max").loc["1990-01-01":]

# Prepare data: add target column
sp500["Target"] = (sp500["Close"].shift(-1) > sp500["Close"]).astype(int)

# Define model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, n_jobs=-1)

# Backtesting functionpy
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i]
        test = data.iloc[i:i+step]
        model.fit(train[predictors], train["Target"])
        predictions = model.predict_proba(test[predictors])[:, 1]
        predictions = (predictions >= 0.6).astype(int)
        all_predictions.append(pd.Series(predictions, index=test.index, name="Predictions"))
    return pd.concat(all_predictions)

# Adding rolling averages and trends
horizons = [2, 5, 60, 250, 1000]
for horizon in horizons:
    rolling_averages = sp500["Close"].rolling(horizon).mean()
    sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_averages
    sp500[f"Trend_{horizon}"] = sp500["Target"].shift(1).rolling(horizon).sum()

# Drop rows with NaNs after feature engineering
sp500.dropna(inplace=True)

# Define new predictors
new_predictors = [f"Close_Ratio_{horizon}" for horizon in horizons] + [f"Trend_{horizon}" for horizon in horizons]

# Run backtest
predictions = backtest(sp500, model, new_predictors)

# Show performance
print("Predictions Value Counts:\n", predictions.value_counts())
print("Precision Score:", precision_score(sp500.loc[predictions.index, "Target"], predictions))
