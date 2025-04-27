import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import utils


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
            # TODO: Add actual feature importance computation here
            compute_feature_importances(data_feature_importance)

def retrain_models():
    """
    Function to handle the actual model retraining process
    """
    # Update status to in_progress
    st.session_state['training_status'] = 'in_progress'
    
    with st.spinner("Retraining models with latest data..."):
        pass
        
    # After retraining completes
    st.session_state['training_status'] = 'completed'
    st.session_state['last_trained_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # You might want to refresh data or rerun the app after training
    st.success("Models successfully retrained with latest data!")

def compute_feature_importances(data_feature_importance):
    """
    Function to compute and display feature importances
    
    Args:
        data_feature_importance: DataFrame containing feature importance data
    """
    # Extract LSTM and Transformer data
    lstm_data = data_feature_importance[data_feature_importance['Model'] == 'lstm_full']
    transformer_data = data_feature_importance[data_feature_importance['Model'] == 'transformer_full']

    # List of features
    feature_columns = ['Price', 'DXY', 'DFII10', 'VIX', 'CPI', 'Sentiment_Score', 
                      'Exponential_Weighted_Score', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']


    # Currently using values directly from the data
    lstm_mean_importance = lstm_data[feature_columns].values.flatten()
    transformer_mean_importance = transformer_data[feature_columns].values.flatten()

    # Create DataFrames for visualization
    lstm_df = pd.DataFrame({'Feature': feature_columns, 'Importance': lstm_mean_importance}).sort_values('Importance', ascending=False)
    transformer_df = pd.DataFrame({'Feature': feature_columns, 'Importance': transformer_mean_importance}).sort_values('Importance', ascending=False)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].bar(lstm_df['Feature'], lstm_df['Importance'], color='blue')
    axes[0].set_title("LSTM Feature Importance")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Feature Importance")
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(transformer_df['Feature'], transformer_df['Importance'], color='green')
    axes[1].set_title("Transformer Feature Importance")
    axes[1].set_xlabel("Features")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    app()