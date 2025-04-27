import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime
import os
import json
from dags.modelling.lstm import train_lstm
from dags.modelling.transformer import train_transformer, GoldPriceTransformer, TransformerEncoderLayer, PositionalEncoding
from dags.modelling.utils import create_sequences, load_scaler, analyse_feature_importance
import pickle
import tensorflow as tf

# Fetch DAG configuration file
config_file_path = "./config/gold_dag_config.json"
DAG_CONFIG = None
with open(config_file_path, 'r') as f:
    DAG_CONFIG = json.load(f)

# Initialise variables from configuration file
project_id = DAG_CONFIG['bigquery']['project_id']
dataset_id = DAG_CONFIG['bigquery']['dataset_id']
gold_market_data_table = DAG_CONFIG['bigquery']['gold_market_data_table']
model_training_status_table = DAG_CONFIG['bigquery']['model_training_status_table']
model_predictions_table = DAG_CONFIG['bigquery']['model_predictions_table']
feature_importances_table = DAG_CONFIG['bigquery']['feature_importances_table']
models_config = DAG_CONFIG['models']

# Configure BigQuery client
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials, project=project_id)

# Initialise session state variables
if 'last_train_date' not in st.session_state:
    st.session_state['last_train_date'] = '2015-11-13'
if 'model_path_dict' not in st.session_state:
    st.session_state['model_path_dict'] = {
        model_config['name']: os.path.join(os.getcwd(), 'dags', 'trained_models', f"{model_config['name']}.keras") 
        for model_config in models_config
    }

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

        model_path_dict = {}
        
        for model_config in models_config:
            if model_config['type'] == 'lstm':
                # Train LSTM
                lstm_scaler_dict, abs_model_path = train_lstm(training_data, model_config['optimal_hp'], model_config['features'], model_config['name'])

            elif model_config['type'] == 'transformer':
                # Train Transformer 
                transformer_scaler_dict, abs_model_path = train_transformer(training_data, model_config['optimal_hp'], model_config['features'], model_config['name'])
            
            # Save absolute path of retrained model
            model_path_dict[model_config['name']] = abs_model_path

        # Save any one of the scaler_dict outputted from training
        scaler_path = save_scaler(transformer_scaler_dict)

        # Update last train date
        st.session_state['last_train_date'] = training_data.iloc[-1]['Date']
        
        # Update status to completed
        update_training_status('completed', 'streamlit_app')
        
        return True, "Models trained successfully"
    except Exception as e:
        # If error, update status to idle
        update_training_status('idle', 'streamlit_app')
        return False, f"Error training models: {str(e)}"   
    
def analyse_models(last_train_date, models_config):
    # Iterate through all models in the model configuration list and do analysis
    for model_config in models_config:
        model_analyse_feature_importance(last_train_date, model_config)
    
def model_analyse_feature_importance(last_train_date, model_config):
    # Fetch data after last_train_date for analysis
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{gold_market_data_table}
    WHERE Date > '{last_train_date}'
    ORDER BY Date ASC
    """

    test_data = client.query(query).to_dataframe()

    # Load the scaler_dict
    scaler_dict = load_scaler()

    # Prepare data for feature importance computation
    features = model_config['features']
    seq_length = model_config['seq_length']

    # Apply the same scalers to test data
    data_scaled = test_data[features].copy()
    for column in data_scaled.columns:
        data_scaled[column] = scaler_dict[column].transform(
            data_scaled[column].values.reshape(-1, 1)
        )

    # Create sequences
    X_test, y_test = create_sequences(data_scaled, seq_length)

    # Load the model
    model_name = model_config['name']
    model_path = st.session_state['model_path_dict'][model_name]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[model_analyse] Model {model_name} not found at {model_path}")
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'GoldPriceTransformer': GoldPriceTransformer,
            'TransformerEncoderLayer': TransformerEncoderLayer,
            'PositionalEncoding': PositionalEncoding
        }
    )

    # Analyse feature importances with model
    mean_importances = analyse_feature_importance(model, X_test, y_test, features)
    mean_importances['Date'] = test_data.iloc[-1]['Date']
    mean_importances['Model'] = model_name

    print(f"[model_analyse] Mean feature importance scores for {model_name} on {mean_importances['Date']} computed: {mean_importances}")

    # Store feature importances in BigQuery table
    # insert_row_to_bigquery(feature_importances_table, mean_importances)
    print(f"[model_analyse] Successfully stored mean feature importance scores for {model_name} on {mean_importances['Date']}")

def insert_row_to_bigquery(table_name, row):
    query = f"""
        SELECT * FROM `{project_id}.{dataset_id}.{table_name}`
        WHERE Date = {row['Date']}
    """
    existing_data = client.query(query).to_dataframe()

    if not existing_data.empty:
        delete_query = f"""
            DELETE FROM `{project_id}.{dataset_id}.{table_name}`
            WHERE Date = {row['Date']}
        """
        client.query(delete_query)

    try:
        table_ref = f"{project_id}.{dataset_id}.{table_name}"
        errors = client.insert_rows_json(table_ref, [row])
        if errors:
            print(f"Encountered errors while inserting rows to {table_ref}:")
            for error in errors:
                print(error)
        else:
            print(f"Successfully inserted row into {table_ref}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def fetch_feature_importances():
    # Fetch and display feature importances
    try:
        feature_importances_query = f"""
        SELECT * FROM `{project_id}.{dataset_id}.{feature_importances_table}`
        ORDER BY Date DESC
        LIMIT 50
        """
        feature_importances_df = client.query(feature_importances_query).to_dataframe()
        return feature_importances_df

    except Exception as e:
        st.error(f"Failed to load feature importances: {e}")

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

# Feature importances display
st.header("Latest Feature Importances")
feature_importances_df = fetch_feature_importances()
st.dataframe(feature_importances_df)

# Button to trigger feature importance computation
if st.button("Compute Feature Importances"):
    with st.spinner("Computing feature importances..."):
        try:
            last_train_date = st.session_state['last_train_date']
            analyse_models(last_train_date, models_config)
            st.success("Feature importances computed successfully!")
        except Exception as e:
            st.error(f"Failed to compute feature importances: {e}")