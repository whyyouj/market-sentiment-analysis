import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime, timedelta
from google.cloud import bigquery
import math

# pre-load model and tokenizer for sentiment scores
model_name = "LHF/finbert-regressor"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# News Sentiment Score from Transformer
def calculate_sentiment_score(news):
    if news is None or len(news) == 0:
        return 0
    
    # load cached model and tokenizer
    global tokenizer
    global model

    inputs = tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()

    return score

# Exponential Moving Average for Price
def ema_price(price_df, alpha=0.9):
    ema30 = price_df.ewm(alpha=alpha, span=30, min_periods=30, adjust=False).mean()
    ema252 = price_df.ewm(alpha=alpha, span=252, min_periods=252, adjust=False).mean()
    return ema30, ema252

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

# Bollinger Bands and Spread
def add_bollinger_bands(price_series, window=20, num_std_dev=2):
    # Calculate the moving average (middle band)
    sma = price_series.rolling(window=window).mean()
    
    # Calculate the rolling standard deviation
    std_dev = price_series.rolling(window=window).std()
    
    # Calculate the upper and lower Bollinger Bands
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)

    # Calculate the band spread
    band_spread = upper_band - lower_band
    
    return band_spread

def exp_weighting_sentiment_score(sentiment_score, execution_date, hist_df, window=30):
    # Ensure execution_date is a datetime object
    if isinstance(execution_date, str):
        execution_date = datetime.strptime(execution_date, '%Y-%m-%d')

    # Get window of past days
    exp_window_df = hist_df.sort_values(by=['Date']).iloc[-window:]

    # Iterate each past day to do exponential weighting of sentiment scores
    exp_sentiment = sentiment_score

    for i, row in exp_window_df.iterrows():
        days_diff = (execution_date - row['Date']).days
        weight = math.exp(-days_diff / window)
        exp_sentiment += weight * row['Sentiment_Score']

    return exp_sentiment