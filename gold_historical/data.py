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

def load_sentiment_data(sentiment_csv):
    """
    Load and clean sentiment analysis data from a CSV file.

    Parameters:
        sentiment_csv (str): Path to the CSV file containing sentiment analysis data
       
    Returns:
        DataFrame: Pandas DataFrame with loaded data
    """
    df = pd.read_csv(sentiment_csv)
    def correct_date_format(date_str):
        parts = date_str.split('/')
        if len(parts) > 1:
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        else:
            # If parsing fails, try to correct the format
            parts = date_str.split('-')
            if len(parts) == 3:
                if len(parts[0]) > 2:
                    # Correct the corrupted year format by moving the first part to the end
                    corrected_date_str = f"{parts[2]}-{parts[1]}-{parts[0][1]}{parts[0][0]}{parts[0][2:]}"
                    return pd.to_datetime(corrected_date_str, dayfirst=True)
                else:
                    # Try to parse the date with day/month/year format
                    corrected_date_str = f"{parts[0]}/{parts[1]}/{parts[2]}"
                    return pd.to_datetime(corrected_date_str, dayfirst=True)
            else:
                raise
    df['Dates'] = df['Dates'].apply(correct_date_format)

    # Create Sentiment_Numeric column
    df['Sentiment_Numeric'] = df['Price Sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0, 'none': 0})

    # Set Dates column as index and calculate average Sentiment_Numeric for each date
    df.set_index('Dates', inplace=True)
    df_avg_sentiment = df.groupby('Dates')['Sentiment_Numeric'].mean()

    # Create a new DataFrame with only Dates and Avg_Sentiment_Numeric columns
    df_final = df_avg_sentiment.reset_index()
    df_final.set_index('Dates', inplace=True)
    return df_final

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
    #interest_rates = pd.read_csv('.data/real-long-term-rates-2000-2024.csv',index_col = 0, parse_dates=True)

    # CPI Data
    cpi_data = clean_CPI('./data/CPI_report.csv')
    cpi_data.columns = ["CPI"]

    ## Sentiment Features
    # News Data
    sentiment_data = load_sentiment_data('./data/gold-dataset-sinha-khandait.csv')

    merged_data = data.join([dxy, real_yields, vix, cpi_data, sentiment_data], how='left').ffill() # .bfill()

    ## Technicals Features
    # Exponential Moving Average with alpha = 0.9
    merged_data["EMA30"] = merged_data[["Price"]].ewm(alpha=0.9, min_periods=30, adjust=False).mean()
    merged_data["EMA252"] = merged_data[["Price"]].ewm(alpha=0.9, min_periods=252, adjust=False).mean()

    # RSI
    merged_data["RSI"] = calculate_rsi(merged_data["Price"])

    return merged_data