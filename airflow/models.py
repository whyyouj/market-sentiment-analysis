import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import utils
from google.oauth2 import service_account
from google.cloud import bigquery
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

def app():
    # Initialize session state for training status
    if 'training_status' not in st.session_state:
        st.session_state['training_status'] = 'completed'
    if 'last_trained_time' not in st.session_state:
        st.session_state['last_trained_time'] = None

    # Get authenticated BigQuery client
    client = utils.get_bigquery_client()

    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return

    # Load data
    with st.spinner("Loading data from BigQuery..."):
        query_feature_importance = """
        SELECT Model, Date, Price, DXY, DFII10, VIX, CPI, Sentiment_Score, Exponential_Weighted_Score, EMA30, EMA252, RSI, Band_Spread
        FROM IS3107_Project.feature_importances
        ORDER BY Date
        """
        data_feature_importance = client.query(query_feature_importance).to_dataframe()

        query_model_pred = """
        SELECT Date, transformer_full, lstm_full
        FROM IS3107_Project.model_predictions
        ORDER BY Date
        """
        data_model_pred = client.query(query_model_pred).to_dataframe()

        query_true_price = """
        SELECT Date, Price
        FROM IS3107_Project.gold_market_data
        ORDER BY Date
        """
        data_true_price = client.query(query_true_price).to_dataframe()

    # Ensure Date is in datetime format
    data_model_pred['Date'] = pd.to_datetime(data_model_pred['Date'])
    data_true_price['Date'] = pd.to_datetime(data_true_price['Date'])

    merged_data = pd.merge(data_model_pred, data_true_price, on="Date", how="left")

    st.title("Gold Price: Model Validation")

    title_col, button_col, col_spacer = st.columns([6, 2, 2])

    with title_col:
        st.subheader("Comparing Actual Prices with Transformer and LSTM Predictions")

    with button_col:
        if st.button("Retrain Models with Latest Data", key="retrain_models"):
            retrain_models()

    # --- Display current training status ---
    st.header("Model Training")

    # Create two placeholders: one for status, one for messages
    status_placeholder = st.empty()
    message_placeholder = st.empty()

    # Display training status
    status_placeholder.write(f"Current training status: **{st.session_state['training_status']}**")
    if st.session_state['last_trained_time']:
        message_placeholder.success(f"Last training completed at: {st.session_state['last_trained_time']}")
    
    # Remove the future prediction point with NaN Price for the visualization
    viz_data = merged_data.dropna(subset=['Price']).copy()
    future_pred = merged_data[merged_data['Price'].isna()].copy()

    # Prepare data for visualization
    dates_list = viz_data['Date'].dt.strftime('%Y-%m-%d').tolist()
    selected_idx = len(viz_data) - 1
    selected_date = viz_data['Date'].iloc[selected_idx]
    filtered_data = viz_data.iloc[:selected_idx + 1]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Price'], mode='lines', name='Actual Price', line=dict(color='#22c55e', width=3)))
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['transformer_full'], mode='lines', name='Transformer Prediction', line=dict(color='#3b82f6', width=3)))
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['lstm_full'], mode='lines', name='LSTM Prediction', line=dict(color='#f59e0b', width=3)))

    if selected_idx == len(viz_data) - 1 and not future_pred.empty:
        last_historical_date = filtered_data['Date'].iloc[-1]
        last_historical_transformer = filtered_data['transformer_full'].iloc[-1]
        last_historical_lstm = filtered_data['lstm_full'].iloc[-1]

        transformer_future_x = [last_historical_date] + future_pred['Date'].tolist()
        transformer_future_y = [last_historical_transformer] + future_pred['transformer_full'].tolist()
        lstm_future_x = [last_historical_date] + future_pred['Date'].tolist()
        lstm_future_y = [last_historical_lstm] + future_pred['lstm_full'].tolist()

        fig.add_trace(go.Scatter(x=transformer_future_x, y=transformer_future_y, mode='lines', name='Transformer (Future)', line=dict(color='#3b82f6', width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=lstm_future_x, y=lstm_future_y, mode='lines', name='LSTM (Future)', line=dict(color='#f59e0b', width=3, dash='dash')))

        fig.add_shape(type="line", xref="x", yref="paper", x0=last_historical_date, y0=0, x1=last_historical_date, y1=1, line=dict(color="gray", width=2, dash="dot"))

    y_values = pd.concat([filtered_data['Price'], filtered_data['transformer_full'], filtered_data['lstm_full']])
    if selected_idx == len(viz_data) - 1 and not future_pred.empty:
        y_values = pd.concat([y_values, future_pred['transformer_full'], future_pred['lstm_full']])

    y_min, y_max = y_values.min(), y_values.max()
    buffer = (y_max - y_min) * 0.1

    fig.update_layout(height=600, hovermode="x unified", xaxis=dict(title="Date", tickformat="%Y-%m-%d"), yaxis=dict(title="Gold Price ($)", range=[y_min - buffer, y_max + buffer]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)"), margin=dict(l=60, r=60, t=80, b=80), plot_bgcolor='rgba(240, 242, 246, 0.8)')

    st.plotly_chart(fig, use_container_width=True)
    
    current_date = dates_list[selected_idx]
    st.subheader(f"Model Performance on {current_date}")
    curr_cols = st.columns(3)
    current_actual = viz_data['Price'].iloc[selected_idx]
    current_transformer = viz_data['transformer_full'].iloc[selected_idx]
    current_lstm = viz_data['lstm_full'].iloc[selected_idx]

    with curr_cols[0]:
        st.metric("Actual Gold Price", f"${current_actual:.2f}")
    with curr_cols[1]:
        transformer_diff = current_transformer - current_actual
        transformer_pct = (transformer_diff / current_actual) * 100
        st.metric("Transformer Prediction", f"${current_transformer:.2f}", f"{transformer_diff:.2f} ({transformer_pct:.2f}%)", delta_color="inverse")
    with curr_cols[2]:
        lstm_diff = current_lstm - current_actual
        lstm_pct = (lstm_diff / current_actual) * 100
        st.metric("LSTM Prediction", f"${current_lstm:.2f}", f"{lstm_diff:.2f} ({lstm_pct:.2f}%)", delta_color="inverse")

    # Calculate metrics
    transformer_errors = filtered_data['transformer_full'] - filtered_data['Price']
    lstm_errors = filtered_data['lstm_full'] - filtered_data['Price']
    transformer_mse = np.mean(transformer_errors**2)
    lstm_mse = np.mean(lstm_errors**2)
    transformer_rmse = np.sqrt(transformer_mse)
    lstm_rmse = np.sqrt(lstm_mse)
    transformer_mae = np.mean(np.abs(transformer_errors))
    lstm_mae = np.mean(np.abs(lstm_errors))

    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'],
        'Transformer': [transformer_mse, transformer_rmse, transformer_mae],
        'LSTM': [lstm_mse, lstm_rmse, lstm_mae]
    })
    st.table(metrics_df.set_index('Metric').style.format('{:.4f}'))

    if selected_idx == len(viz_data) - 1 and not future_pred.empty:
        st.subheader("Future Price Prediction")
        future_date = future_pred['Date'].iloc[0].strftime('%Y-%m-%d')
        future_cols = st.columns(2)
        with future_cols[0]:
            st.metric("Transformer Prediction", f"${future_pred['transformer_full'].iloc[0]:.2f}")
            st.markdown(f"**Date**: {future_date}")
        with future_cols[1]:
            st.metric("LSTM Prediction", f"${future_pred['lstm_full'].iloc[0]:.2f}")
        st.caption("Note: This is a future prediction where actual price data is not yet available")

    col1, col2, col_spacer = st.columns([3, 3, 6])
    with col1:
        st.header("Feature Importance")
    with col2:
        st.write("")
        if st.button("Compute Feature Importances", key="compute_fi"):
            last_train_date = st.session_state['last_train_date']
            analyse_models(last_train_date, models_config)

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
    insert_row_to_bigquery(feature_importances_table, mean_importances)
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

if __name__ == "__main__":
    app()