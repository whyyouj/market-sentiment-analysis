import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime
import os
import json
import sys
from dags.modelling.lstm import train_lstm
from dags.modelling.transformer import train_transformer
import pickle

# Fetch DAG configuration file
config_file_path = "./config/gold_dag_config.json"
DAG_CONFIG = None
with open(config_file_path, 'r') as f:
    DAG_CONFIG = json.load(f)

# Configure BigQuery client
project_id = DAG_CONFIG['bigquery']['project_id']
dataset_id = DAG_CONFIG['bigquery']['dataset_id']
gold_market_data_table = DAG_CONFIG['bigquery']['gold_market_data_table']
model_training_status_table = DAG_CONFIG['bigquery']['model_training_status_table']
model_predictions_table = DAG_CONFIG['bigquery']['model_predictions_table']

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials, project=project_id)

def fetch_predictions():
    """Fetch latest predictions from BigQuery"""
    query = f"""
    SELECT * FROM `{project_id}.{dataset_id}.{model_predictions_table}`
    ORDER BY Date DESC
    LIMIT 30
    """
    return client.query(query).to_dataframe()

def fetch_training_status():
    """Get current model training status"""
    query = f"""
    SELECT * FROM `{project_id}.{dataset_id}.{model_training_status_table}`
    WHERE status_id = 'model_training_status'
    """
    return client.query(query).to_dataframe().iloc[0]

def update_training_status(status, triggered_by):
    """Update training status in BigQuery"""
    status_table = f"{project_id}.{dataset_id}.{model_training_status_table}"
    now = datetime.now().isoformat()
    
    if status == 'started':
        query = f"""
        UPDATE `{status_table}`
        SET training_status = 'in_progress',
            start_time = '{now}',
            end_time = NULL,
            updated_at = '{now}',
            triggered_by = '{triggered_by}'
        WHERE status_id = 'model_training_status'
        """
    elif status == 'completed':
        query = f"""
        UPDATE `{status_table}`
        SET training_status = 'completed',
            end_time = '{now}',
            updated_at = '{now}'
        WHERE status_id = 'model_training_status'
        """
    elif status == 'idle':
        query = f"""
        UPDATE `{status_table}`
        SET training_status = 'idle',
            updated_at = '{now}'
        WHERE status_id = 'model_training_status'
        """
    
    client.query(query)

def save_scaler(scaler):
    # Create scaler directory if it does not exist
    scalers_path = os.path.join(os.getcwd(), 'dags', 'trained_models', 'scalers')
    if not os.path.exists(scalers_path):
        os.makedirs(scalers_path)
    
    # Save with pickle
    scaler_path = os.path.join(scalers_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Saved scaler to {scaler_path}")
    return scaler_path
    
def retrain_models():
    """Retrain both LSTM and transformer models with latest data"""
    try:
        # Update status to in_progress
        update_training_status('started', 'streamlit_app')
        
        # Fetch latest training data
        query = f"""
        SELECT * FROM `{project_id}.{dataset_id}.{gold_market_data_table}`
        ORDER BY Date ASC
        """
        training_data = client.query(query).to_dataframe()
        
        for model_config in DAG_CONFIG['models']:
            if model_config['type'] == 'lstm':
                # Train LSTM
                lstm_scaler_dict = train_lstm(training_data, model_config['optimal_hp'], model_config['features'], model_config['name'])
            elif model_config['type'] == 'transformer':
                # Train Transformer 
                transformer_scaler_dict = train_transformer(training_data, model_config['optimal_hp'], model_config['features'], model_config['name'])

        # Save any one of the scaler_dict outputted from training
        scaler_path = save_scaler(transformer_scaler_dict)
        
        # Update status to completed
        update_training_status('completed', 'streamlit_app')
        
        return True, "Models trained successfully"
    except Exception as e:
        # If error, update status to idle
        update_training_status('idle', 'streamlit_app')
        return False, f"Error training models: {str(e)}"

# Streamlit UI
st.title("Gold Price Prediction Dashboard")

# Predictions display
st.header("Latest Predictions")
predictions_df = fetch_predictions()
st.dataframe(predictions_df)

# Model training section
st.header("Model Training")
training_status = fetch_training_status()
status_label = training_status['training_status']

st.write(f"Current training status: **{status_label}**")
if status_label == 'in_progress':
    st.write(f"Training started at: {training_status['start_time']}")
    st.write(f"Triggered by: {training_status['triggered_by']}")
    st.warning("Models are currently being trained. Please wait.")
else:
    if status_label == 'completed':
        st.success(f"Last training completed at: {training_status['end_time']}")
    
# Only show retraining button if not currently training
if st.button("Retrain Models with Latest Data"):
    with st.spinner("Retraining models..."):
        success, message = retrain_models()
        if success:
            st.success(message)
        else:
            st.error(message)