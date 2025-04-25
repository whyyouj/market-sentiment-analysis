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
    ema30 = price_df.ewm(alpha=alpha, min_periods=30, adjust=False).mean()
    ema252 = price_df.ewm(alpha=alpha, min_periods=252, adjust=False).mean()
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

# # Exponential weighting of news sentiment scores
# def exp_weighting_sentiment_score(bq_config, bq_hook, sentiment_score, execution_date, window=30):
#     # Ensure execution_date is a datetime object
#     if isinstance(execution_date, str):
#         execution_date = datetime.strptime(execution_date, '%Y-%m-%d')
    
#     # Compute end date (n days after execution_date, where n = window)
#     end_date = execution_date + timedelta(days=window)
    
#     # Format dates for BigQuery
#     execution_date_str = execution_date.strftime('%Y-%m-%d')
#     end_date_str = end_date.strftime('%Y-%m-%d')
    
#     # Fetch existing data from BigQuery
#     project_id = bq_config['project_id']
#     dataset_id = bq_config['dataset_id']
#     exp_table_id = bq_config['exp_weighted_sentiment_table']
    
#     query = f"""
#     SELECT Date, Sentiment_Score, Exponential_Weighted_Score
#     FROM `{project_id}.{dataset_id}.{exp_table_id}`
#     WHERE Date >= '{execution_date_str}' AND Date <= '{end_date_str}'
#     ORDER BY Date ASC
#     """
    
#     client = bq_hook.get_client()
    
#     # Run query to get existing data
#     existing_data_df = client.query(query).to_dataframe()
#     existing_data_df['Date'] = pd.to_datetime(existing_data_df['Date'])
    
#     # Create a complete date range from execution_date to end_date
#     date_range = pd.date_range(start=execution_date, end=end_date, freq='D')
#     date_df = pd.DataFrame({'Date': date_range})
    
#     # If existing data is empty, create a new DataFrame with zeros
#     if existing_data_df.empty:
#         scores_df = pd.DataFrame({
#             'Date': date_range,
#             'Sentiment_Score': [0] * len(date_range),
#             'Exponential_Weighted_Score': [0] * len(date_range)
#         })
#     else:
#         # Merge existing data with complete date range, filling missing values with 0
#         scores_df = pd.merge(date_df, existing_data_df, on='Date', how='left')
#         scores_df.fillna(0, inplace=True)
    
#     # Compute new exponential weights for each day
#     scores_df['days_diff'] = (scores_df['Date'] - execution_date).dt.days
#     scores_df['weights'] = np.exp(-scores_df['days_diff'] / window)
    
#     # Update exponential weighted scores
#     scores_df['Exponential_Weighted_Score'] += sentiment_score * scores_df['weights']
    
#     # Set sentiment score for execution date
#     execution_idx = scores_df[scores_df['Date'] == execution_date].index
#     if len(execution_idx) > 0:
#         scores_df.loc[execution_idx, 'Sentiment_Score'] = sentiment_score
    
#     # Prepare final dataframe for insertion
#     result_df = scores_df[['Date', 'Sentiment_Score', 'Exponential_Weighted_Score']]
    
#     # Convert date to string format for BigQuery
#     result_df['Date'] = result_df['Date'].dt.strftime('%Y-%m-%d')
    
#     # # Insert data into BigQuery, overwriting existing data
#     # # First delete existing records for the date range
#     # delete_query = f"""
#     # DELETE FROM `{project_id}.{dataset_id}.{exp_table_id}`
#     # WHERE Date >= '{execution_date_str}' AND Date <= '{end_date_str}'
#     # """
#     # client.query(delete_query).result()
    
#     # # Insert new records
#     # table_ref = f"{project_id}.{dataset_id}.{exp_table_id}"
#     # table = client.get_table(table_ref)
    
#     # # Convert final dataframe to list of dictionaries for insertion
#     # rows_to_insert = result_df.to_dict('records')
    
#     # # Insert rows from final dataframe into BigQuery
#     # errors = client.insert_rows_json(table, rows_to_insert)
    
#     # if errors:
#     #     raise Exception(f"[exp_weighting_sentiment_score] Error inserting rows into BigQuery: {errors}")
#     # else:
#     #     print(f"[exp_weighting_sentiment_score] Successfully updated {len(rows_to_insert)} rows of exponentially weighted scores")

#     # Upload to staging table
#     staging_table_id = f"{project_id}.{dataset_id}.temp_exp_weighted_sentiment"
#     job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
#     load_job = client.load_table_from_dataframe(result_df, staging_table_id, job_config=job_config)
#     load_job.result()  # Wait for the job to finish

#     # Merge staging table into main table
#     merge_query = f"""
#     MERGE `{project_id}.{dataset_id}.{exp_table_id}` T
#     USING `{staging_table_id}` S
#     ON T.Date = S.Date
#     WHEN MATCHED THEN
#     UPDATE SET T.Sentiment_Score = S.Sentiment_Score,
#                 T.Exponential_Weighted_Score = S.Exponential_Weighted_Score
#     WHEN NOT MATCHED THEN
#     INSERT (Date, Sentiment_Score, Exponential_Weighted_Score)
#     VALUES(S.Date, S.Sentiment_Score, S.Exponential_Weighted_Score)
#     """

#     try:
#         client.query(merge_query).result()
#         print(f"[exp_weighting_sentiment_score] Successfully updated {len(result_df)} rows of exponentially weighted scores")
#     except Exception as e:
#         print(f"[exp_weighting_sentiment_score] Error inserting rows into BigQuery: {e}")
#         raise
    
#     # Return the exp weighted score on execution_date
#     return scores_df.loc[execution_idx, 'Exponential_Weighted_Score'].iloc[0]

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