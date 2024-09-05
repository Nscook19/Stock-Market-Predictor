# Stock Market Prediction Using Random Forest Classifier

This project uses historical S&P 500 data to predict stock market trends using a Random Forest Classifier. The model is built with Python, utilizing the `yfinance` library to fetch data and `scikit-learn` for machine learning algorithms.

## Overview

- **Data Source**: S&P 500 historical data from 1990 onwards, fetched using the `yfinance` library.
- **Target**: The model predicts whether the closing price of the S&P 500 will increase the following day.
- **Features**: Rolling averages and trends are used as predictors for various horizons (2, 5, 60, 250, and 1000 days).
- **Model**: A Random Forest Classifier is trained and backtested on historical data with precision evaluation.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install yfinance pandas scikit-learn
