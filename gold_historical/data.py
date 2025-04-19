import pandas as pd
import numpy as np

# Helper Functions
def clean_CPI(cpi_csv):
    """
    Load and clean CPI data from a CSV file.

    Parameters:
        cpi_csv (str): Path to the CSV file containing CPI data
       
    Returns:
        DataFrame: Pandas DataFrame with loaded data
    """
    
    df = pd.read_csv(cpi_csv)

    # Function to combine Year and Period into Date
    def combine_year_period(row):
        if row['Period'].startswith('M'):
            month = int(row['Period'][1:])
            return pd.to_datetime(f"{row['Year']}-{month}-01")
        else:
            return None


    # Apply the function to create Date column
    df['Date'] = df.apply(combine_year_period, axis=1)

    # Drop rows with None values in Date column
    df = df.dropna(subset=['Date'])

    # Drop Year and Period columns
    df = df.drop(columns=['Year', 'Period', 'Series ID'])
    df.set_index('Date', inplace=True)
    return df

# Relative Strength Index (RSI)
def calculate_rsi(price_series, window=14):
    # Calculate price changes
    delta = price_series.diff()
    
    # Create copies for gain and loss
    gain = delta.copy()
    loss = delta.copy()
    
    # Set gains to 0 where price decreased
    gain[gain < 0] = 0
    
    # Set losses to 0 where price increased (and make losses positive)
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss over the specified window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Bollinger Bands
def add_bollinger_bands(price_series, window=20, num_std_dev=2):
    # Calculate the moving average (middle band)
    sma = price_series.rolling(window=window).mean()
    
    # Calculate the rolling standard deviation
    std_dev = price_series.rolling(window=window).std()
    
    # Calculate the upper and lower Bollinger Bands
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    
    return upper_band, lower_band

# ========== #
# Final Data #
# ========== #
def load_data():
    
    # Gold Prices
    data = pd.read_csv('./data/Prices_cleaned.csv',index_col = 0, parse_dates=True)
    data['Price'] = data['Price'].str.replace(',', '').astype(float)
    data = data.loc[(data.index >= "1998-01-01") & (data.index < "2025-01-01")]
    data.sort_index(inplace=True)

    ## Macro Features
    # USD Rates
    dxy = pd.read_csv('./data/dxy.csv',index_col = 0, parse_dates=True)

    # Real Yields
    real_yields = pd.read_csv('./data/real_yields.csv',index_col = 0, parse_dates=True)

    # VIX
    vix = pd.read_csv('./data/vix.csv',index_col = 0, parse_dates=True)

    # CPI Data
    cpi_data = clean_CPI('./data/CPI_report.csv')
    cpi_data.columns = ["CPI"]

    ## Sentiment Features
    # News Data
    merged_data = data.join([dxy, real_yields, vix, cpi_data], how='left').ffill() # .bfill()

    sentiment_score_data = pd.read_csv("./data/gold-daily-sentiment-with-weighting.csv", index_col = 0, parse_dates = True)
    merged_data = merged_data.join(sentiment_score_data[["Sentiment_Score", "Exponential_Weighted_Score"]], how="left")

    ## Technicals Features
    # Exponential Moving Average with alpha = 0.9
    merged_data["EMA30"] = merged_data[["Price"]].ewm(alpha=0.9, min_periods=30, adjust=False).mean()
    merged_data["EMA252"] = merged_data[["Price"]].ewm(alpha=0.9, min_periods=252, adjust=False).mean()

    # RSI
    merged_data["RSI"] = calculate_rsi(merged_data["Price"])
    
    # Bollinger bands
    merged_data["Upper_Band"], merged_data["Lower_Band"] = add_bollinger_bands(merged_data["Price"])
    merged_data = merged_data.dropna()

    return merged_data