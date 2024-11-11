# financial_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load historical data for TSLA, BND, SPY
def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']  # Return only adjusted closing prices

# Basic data cleaning, type checking, and ensuring appropriate data types
def clean_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Fill or interpolate missing values
    df = df.interpolate(method='linear').fillna(method='bfill')
    
    # Ensure all columns have appropriate data types (e.g., float for numerical data)
    for column in df.columns:
        if not pd.api.types.is_float_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # After conversion, re-check for any new NaN values and fill them
    df = df.fillna(method='bfill').fillna(method='ffill')  # Ensure no remaining NaNs
    
    return df, missing_values


# Generate basic statistics to understand the data distribution
def data_summary(df):
    stats = df.describe()
    return stats

# Normalize the data
def normalize_data(df):
    normalized_df = (df - df.mean()) / df.std()
    return normalized_df

# Visualize closing price over time
def plot_closing_prices(df):
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.title("Adjusted Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

# Calculate daily percentage change and plot volatility
def plot_daily_percentage_change(df):
    pct_change = df.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for column in pct_change.columns:
        plt.plot(pct_change[column], label=f'{column} Daily % Change')
    plt.title("Daily Percentage Change")
    plt.xlabel("Date")
    plt.ylabel("Percentage Change")
    plt.legend()
    plt.show()
    return pct_change

# Calculate rolling means and standard deviations for volatility analysis
def plot_rolling_stats(df, window=20):
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        plt.plot(rolling_mean, label=f'{column} {window}-Day Rolling Mean')
        plt.plot(rolling_std, linestyle='--', label=f'{column} {window}-Day Rolling Std Dev')
    plt.title(f"{window}-Day Rolling Mean and Standard Deviation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Outlier detection for extreme returns
def detect_outliers(df, threshold=3):
    outliers = pd.DataFrame()
    for column in df.columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers[column] = np.where(z_scores.abs() > threshold, df[column], np.nan)
    return outliers

# Decompose the time series to analyze trend, seasonality, and residuals
def decompose_time_series(df, column, model='multiplicative', period=252):
    decomposition = seasonal_decompose(df[column], model=model, period=period)
    decomposition.plot()
    plt.show()
    return decomposition

# Calculate Value at Risk (VaR) and Sharpe Ratio
def calculate_risk_metrics(df, risk_free_rate=0.02):
    daily_returns = df.pct_change().dropna()
    # VaR at 5% confidence interval
    var_95 = daily_returns.quantile(0.05)
    # Sharpe Ratio (risk-adjusted return)
    sharpe_ratio = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std()
    return var_95, sharpe_ratio

# ARIMA Model
def arima_model(train_data, test_data):
    try:
        arima_model = ARIMA(train_data, order=(5, 1, 0))
        arima_model_fit = arima_model.fit()

        arima_predictions = arima_model_fit.forecast(steps=len(test_data))
        arima_predictions_series = pd.Series(arima_predictions, index=test_data.index)
        
        mae_arima = mean_absolute_error(test_data, arima_predictions_series)
        rmse_arima = np.sqrt(mean_squared_error(test_data, arima_predictions_series))
        mape_arima = np.mean(np.abs((test_data - arima_predictions_series) / test_data)) * 100

        return mae_arima, rmse_arima, mape_arima

    except ValueError as e:
        print(f"ARIMA model error: {e}")
        return None, None, None

# SARIMA Model
def sarima_model(train_data, test_data):
    try:
        # Ensure numeric values
        train_data = pd.to_numeric(train_data, errors='coerce').dropna()
        test_data = pd.to_numeric(test_data, errors='coerce').dropna()

        if len(train_data) < 12 or len(test_data) < 12:
            print("Insufficient data for SARIMA.")
            return None, None, None
        
        sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        sarima_model_fit = sarima_model.fit(disp=False)
        
        sarima_predictions = sarima_model_fit.forecast(steps=len(test_data))
        
        mae_sarima = mean_absolute_error(test_data, sarima_predictions)
        rmse_sarima = np.sqrt(mean_squared_error(test_data, sarima_predictions))
        mape_sarima = np.mean(np.abs((test_data - sarima_predictions) / test_data)) * 100

        return mae_sarima, rmse_sarima, mape_sarima

    except Exception as e:
        print(f"SARIMA model error: {e}")
        return None, None, None

# Prepare LSTM Data
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X).reshape((len(X), look_back, 1))
    y = np.array(y)
    
    return X, y, scaler

# LSTM Model
# def lstm_model(train_data, test_data, look_back=60, epochs=10, batch_size=32):
#     X_train, y_train, scaler = prepare_lstm_data(train_data, look_back)
#     X_test, y_test, _ = prepare_lstm_data(test_data, look_back)
    
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
#     lstm_predictions = scaler.inverse_transform(model.predict(X_test))
#     y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
#     mae_lstm = mean_absolute_error(y_test_actual, lstm_predictions)
#     rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
#     mape_lstm = np.mean(np.abs((y_test_actual - lstm_predictions) / y_test_actual)) * 100

#     return mae_lstm, rmse_lstm, mape_lstm
def lstm_model(train_data, test_data, look_back=60, epochs=10, batch_size=32):
    X_train, y_train, scaler = prepare_lstm_data(train_data, look_back)
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Return only the model and scaler
    return model, scaler

def forecast_arima(train_data, forecast_period=180, order=(5, 1, 0)):
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=forecast_period)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    return forecast_values, confidence_intervals
def forecast_sarima(train_data, forecast_period=180, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=forecast_period)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    return forecast_values, confidence_intervals
def forecast_lstm(model, data, scaler, look_back=60, forecast_period=180):
    # Prepare initial input for forecast generation
    inputs = data[-look_back:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    forecast_values = []

    for _ in range(forecast_period):
        X_input = np.array(inputs[-look_back:]).reshape(1, look_back, 1)
        predicted_value = model.predict(X_input)
        forecast_values.append(predicted_value[0, 0])
        inputs = np.append(inputs, predicted_value)[-look_back:]

    # Inverse scale the forecast values
    forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()
    
    return forecast_values

def forecast_and_analyze(train_data, model_type="arima", forecast_period=180):
    if model_type.lower() == "arima":
        forecast_values, confidence_intervals = forecast_arima(train_data, forecast_period)
    elif model_type.lower() == "sarima":
        forecast_values, confidence_intervals = forecast_sarima(train_data, forecast_period)
    elif model_type.lower() == "lstm":
        model, scaler = lstm_model(train_data, train_data)
        forecast_values = forecast_lstm(model, train_data, scaler, forecast_period=forecast_period)
        confidence_intervals = None

    # Plotting and Analysis
    plt.figure(figsize=(12, 6))
    train_data.plot(label='Historical Data')
    
    forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=forecast_period)
    forecast_series = pd.Series(forecast_values, index=forecast_index)
    forecast_series.plot(label=f'{model_type.upper()} Forecast', color='orange')

    if confidence_intervals is not None:
        plt.fill_between(forecast_index, 
                         confidence_intervals.iloc[:, 0], 
                         confidence_intervals.iloc[:, 1], 
                         color='pink', alpha=0.3)

    plt.title(f'{model_type.upper()} Forecast for Tesla Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Analysis
    print("Trend Analysis:")
    trend_direction = "upward" if forecast_values[-1] > forecast_values[0] else "downward"
    print(f"The trend over the forecast period is {trend_direction}.")

    print("\nVolatility and Risk Analysis:")
    if confidence_intervals is not None:
        print("The forecast includes confidence intervals, indicating expected price fluctuation ranges.")
    else:
        print("Confidence intervals are unavailable for the LSTM model.")
    
    print("\nMarket Opportunities and Risks:")
    if trend_direction == "upward":
        print("Potential market opportunity due to an expected price increase.")
    else:
        print("Potential market risk due to an expected price decrease.")
