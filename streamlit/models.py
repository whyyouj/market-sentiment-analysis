import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
import utils

def app():
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

        query_true_price = """
        SELECT Date, Price
        FROM IS3107_Project.gold_market_data
        ORDER BY Date
        """
        data_model_pred = client.query(query_model_pred).to_dataframe()
        data_true_price = client.query(query_true_price).to_dataframe()

    # Ensure Date is in datetime format
    data_model_pred['Date'] = pd.to_datetime(data_model_pred['Date'])
    data_true_price['Date'] = pd.to_datetime(data_true_price['Date'])

    merged_data = pd.merge(data_model_pred, data_true_price, on = "Date", how = "left")
    st.title("Gold Price: Model Validation")
    st.subheader("Comparing Actual Prices with Transformer and LSTM Predictions")
    
    # Remove the future prediction point with NaN Price for the visualization
    viz_data = merged_data.dropna(subset=['Price']).copy()
    
    # Store the future prediction separately
    future_pred = merged_data[merged_data['Price'].isna()].copy()
    
    # Convert dates to formatted strings for display
    dates_list = viz_data['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Create a date selector directly in the main panel with more prominence
    st.subheader("Select a Date to View")
    
    # Get min and max dates for better display
    min_date = viz_data['Date'].min().strftime('%Y-%m-%d')
    max_date = viz_data['Date'].max().strftime('%Y-%m-%d')
    
    # Create columns for date selector
    date_col1, date_col2 = st.columns([3, 1])
    
    with date_col1:
        # Create a date slider - make sure it's more visible in the main content
        default_idx = len(viz_data) - 1  # Default to the most recent date
        selected_idx = st.slider(
            f"Date Range: {min_date} to {max_date}",
            min_value=0,
            max_value=len(viz_data) - 1,
            value=default_idx,
        )
    
    with date_col2:
        selected_date = viz_data['Date'].iloc[selected_idx]
        st.info(f"Selected: {selected_date.strftime('%Y-%m-%d')}")
    
    # Filter data up to the selected date
    filtered_data = viz_data.iloc[:selected_idx + 1]
    
    st.subheader("Comparing Actual Prices with Transformer and LSTM Predictions")
    
    # Create visualization
    fig = go.Figure()
    
    # Add all the data up to selected date
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'], 
        y=filtered_data['Price'], 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='#22c55e', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'], 
        y=filtered_data['transformer_full'], 
        mode='lines', 
        name='Transformer Prediction', 
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'], 
        y=filtered_data['lstm_full'], 
        mode='lines', 
        name='LSTM Prediction', 
        line=dict(color='#f59e0b', width=3)
    ))
    
    # Add future predictions if we're at the last historical point
    if selected_idx == len(viz_data) - 1 and not future_pred.empty:
        # Get the last historical point
        last_historical_date = filtered_data['Date'].iloc[-1]
        last_historical_transformer = filtered_data['transformer_full'].iloc[-1]
        last_historical_lstm = filtered_data['lstm_full'].iloc[-1]
        
        # Add future transformer prediction
        transformer_future_x = [last_historical_date] + future_pred['Date'].tolist()
        transformer_future_y = [last_historical_transformer] + future_pred['transformer_full'].tolist()
        
        fig.add_trace(go.Scatter(
            x=transformer_future_x,
            y=transformer_future_y,
            mode='lines',
            name='Transformer (Future)',
            line=dict(color='#3b82f6', width=3, dash='dash')
        ))
        
        # Add future LSTM prediction
        lstm_future_x = [last_historical_date] + future_pred['Date'].tolist()
        lstm_future_y = [last_historical_lstm] + future_pred['lstm_full'].tolist()
        
        fig.add_trace(go.Scatter(
            x=lstm_future_x,
            y=lstm_future_y,
            mode='lines',
            name='LSTM (Future)',
            line=dict(color='#f59e0b', width=3, dash='dash')
        ))
        
        # Add vertical line for future boundary
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=last_historical_date,
            y0=0,
            x1=last_historical_date,
            y1=1,
            line=dict(
                color="gray",
                width=2,
                dash="dot",
            )
        )
    
    # Update layout with formatting
    y_values = pd.concat([filtered_data['Price'], filtered_data['transformer_full'], filtered_data['lstm_full']])
    
    if selected_idx == len(viz_data) - 1 and not future_pred.empty:
        y_values = pd.concat([y_values, future_pred['transformer_full'], future_pred['lstm_full']])
    
    y_min, y_max = y_values.min(), y_values.max()
    buffer = (y_max - y_min) * 0.1
    
    fig.update_layout(
        height=600,
        hovermode="x unified",
        xaxis=dict(
            title="Date",
            tickformat="%Y-%m-%d",
        ),
        yaxis=dict(
            title="Gold Price ($)",
            range=[y_min - buffer, y_max + buffer]
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=60, t=80, b=80),
        plot_bgcolor='rgba(240, 242, 246, 0.8)'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Get metrics for the selected date
    current_date = dates_list[selected_idx]
    current_actual = viz_data['Price'].iloc[selected_idx]
    current_transformer = viz_data['transformer_full'].iloc[selected_idx]
    current_lstm = viz_data['lstm_full'].iloc[selected_idx]
    
    # Display metrics for selected date
    st.subheader(f"Model Performance on {current_date}")
    curr_cols = st.columns(3)
    
    with curr_cols[0]:
        st.metric("Actual Gold Price", f"${current_actual:.2f}")
    with curr_cols[1]:
        transformer_diff = current_transformer - current_actual
        transformer_pct = (transformer_diff / current_actual) * 100
        st.metric(
            "Transformer Prediction", 
            f"${current_transformer:.2f}", 
            f"{transformer_diff:.2f} ({transformer_pct:.2f}%)",
            delta_color="inverse"
        )
    with curr_cols[2]:
        lstm_diff = current_lstm - current_actual
        lstm_pct = (lstm_diff / current_actual) * 100
        st.metric(
            "LSTM Prediction", 
            f"${current_lstm:.2f}", 
            f"{lstm_diff:.2f} ({lstm_pct:.2f}%)",
            delta_color="inverse"
        )
    
    # Calculate performance metrics up to selected date
    transformer_errors = filtered_data['transformer_full'] - filtered_data['Price']
    lstm_errors = filtered_data['lstm_full'] - filtered_data['Price']
    
    transformer_mse = np.mean(transformer_errors**2)
    lstm_mse = np.mean(lstm_errors**2)
    
    # Calculate RMSE correctly
    transformer_rmse = np.sqrt(transformer_mse)
    lstm_rmse = np.sqrt(lstm_mse)
    
    transformer_mae = np.mean(np.abs(transformer_errors))
    lstm_mae = np.mean(np.abs(lstm_errors))
    
    st.subheader("Model Performance Metrics (Up to Selected Date)")
    
    # Create comparison table for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'],
        'Transformer': [transformer_mse, transformer_rmse, transformer_mae],
        'LSTM': [lstm_mse, lstm_rmse, lstm_mae]
    })
    
    st.table(metrics_df.set_index('Metric').style.format('{:.4f}'))
    
    # Future prediction display (only show when at the latest date)
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
    
    # Add help text at the bottom
    st.markdown("---")
    st.caption("Use the slider above to navigate through different dates and see how model predictions change over time.")

    # ========================= #
    # Feature Importance Charts #
    # ========================= #
    st.title("Feature Importance")
    # Filter LSTM and Transformer data
    lstm_data = data_feature_importance[data_feature_importance['Model'] == 'lstm_full']
    transformer_data = data_feature_importance[data_feature_importance['Model'] == 'transformer_full']

    # Example feature importance columns to be plotted
    feature_columns = ['Price', 'DXY', 'DFII10', 'VIX', 'CPI', 'Sentiment_Score', 'Exponential_Weighted_Score', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']

    # Get feature importance for LSTM and Transformer
    lstm_mean_importance = lstm_data[feature_columns].values.flatten()
    transformer_mean_importance = transformer_data[feature_columns].values.flatten()
    
    # Sort LSTM feature importance
    lstm_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': lstm_mean_importance
    })
    lstm_df = lstm_df.sort_values('Importance', ascending=False)
    
    # Sort Transformer feature importance
    transformer_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': transformer_mean_importance
    })
    transformer_df = transformer_df.sort_values('Importance', ascending=False)
    
    # Create the side-by-side barplots using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Barplot for LSTM (sorted)
    axes[0].bar(lstm_df['Feature'], lstm_df['Importance'], color='blue')
    axes[0].set_title("LSTM Feature Importance")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Feature Importance")
    axes[0].tick_params(axis='x', rotation=45)

    # Barplot for Transformer (sorted)
    axes[1].bar(transformer_df['Feature'], transformer_df['Importance'], color='green')
    axes[1].set_title("Transformer Feature Importance")
    axes[1].set_xlabel("Features")
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.title("Actual vs Predicted Values Visualization")
    st.write("Upload CSV files containing 'actual' and 'predicted' values to visualize the comparison")

    # File uploaders - simplified to just one pair
    col1, col2 = st.columns(2)
    with col1:
        actual_file = st.file_uploader("Upload 'Actual Values' CSV file", type="csv", key="actual_file")
        st.info("CSV file should contain a single column of actual gold price values")
    with col2:
        pred_file = st.file_uploader("Upload 'Predicted Values' CSV file", type="csv", key="pred_file")
        st.info("CSV file should contain a single column of predicted gold price values")

    # Parse CSV files
    @st.cache_data
    def parse_csv(file):
        try:
            df = pd.read_csv(file)
            # If the CSV has multiple columns, we'll take the first column
            if len(df.columns) > 1:
                st.info(f"Multiple columns detected in file. Using the first column: {df.columns[0]}")
                return df.iloc[:, 0].values
            return df.iloc[:, 0].values
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            return None

    # Initialize session state for the current point
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
        st.session_state.current_actual = 0
        st.session_state.current_predicted = 0
        st.session_state.current_error = 0

    # Hide the number input with CSS
    st.markdown("""
    <style>
    [data-testid="stNumberInput"] {
        position: absolute;
        width: 0;
        height: 0;
        opacity: 0;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # Process data when files are uploaded
    if actual_file and pred_file:
        actual_values = parse_csv(actual_file)
        pred_values = parse_csv(pred_file)
        
        # Check if the arrays have the same length
        if len(actual_values) != len(pred_values):
            st.error("Error: The 'actual' and 'predicted' data must have the same number of points")
        else:
            st.success(f"Files loaded successfully! Found {len(actual_values)} data points.")
            
            # Calculate error metrics
            errors = np.abs(actual_values - pred_values)
            mse = np.mean(np.square(actual_values - pred_values))
            rmse = np.sqrt(mse)
            mae = np.mean(errors)
            
            # Create animation frames
            frames = []
            for i in range(len(actual_values)):
                # Data for each frame
                actual_trace = actual_values[:i+1]
                pred_trace = pred_values[:i+1]
                
                frame_data = [
                    go.Scatter(x=list(range(i+1)), y=actual_trace, mode='lines', line=dict(color='blue')),
                    go.Scatter(x=list(range(i+1)), y=pred_trace, mode='lines', line=dict(color='red'))
                ]
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            # Create base figure
            fig = go.Figure(
                data=[
                    go.Scatter(x=[0], y=[actual_values[0]], mode='lines', name='Actual', line=dict(color='blue')),
                    go.Scatter(x=[0], y=[pred_values[0]], mode='lines', name='Predicted', line=dict(color='red'))
                ]
            )
            
            # Update axes ranges
            y_min = min(min(actual_values), min(pred_values))
            y_max = max(max(actual_values), max(pred_values))
            buffer = (y_max - y_min) * 0.1
            fig.update_layout(
                xaxis=dict(title='Time', range=[0, len(actual_values)]),
                yaxis=dict(title='Value', range=[y_min - buffer, y_max + buffer])
            )
            
            # Add frames and buttons
            fig.frames = frames
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                            )
                        ],
                        direction="left",
                        pad={"r": 10, "t": 10},
                        showactive=False,
                        x=0.1,
                        xanchor="right",
                        y=1.1,
                        yanchor="top"
                    )
                ],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Frame: "
                    },
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate"
                                }
                            ],
                            "label": str(i),
                            "method": "animate"
                        }
                        for i in range(len(frames))
                    ]
                }],
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Add hidden number input to track current frame
            current_frame = st.number_input("Current Frame", min_value=0, max_value=len(actual_values)-1, 
                                          value=0, key="current_frame", label_visibility="hidden")
            
            # JavaScript to sync Plotly animation with Streamlit metrics
            js_code = """
            <script>
            const streamlitDoc = window.parent.document;
            
            function updateStreamlitFrame(frameNum) {
                // Find the hidden number input
                const numberInputs = streamlitDoc.querySelectorAll('input[type="number"]');
                let hiddenInput = null;
                
                for (const input of numberInputs) {
                    if (input.style.opacity === '0' || input.style.display === 'none' || 
                        input.parentElement.style.opacity === '0' || 
                        input.closest('[data-testid="stNumberInput"]')?.style.opacity === '0') {
                        hiddenInput = input;
                        break;
                    }
                }
                
                if (!hiddenInput) return;
                
                // Update value and dispatch events
                hiddenInput.value = frameNum;
                hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Force Streamlit to receive the update
                const changeEvent = new Event('change', { bubbles: true });
                hiddenInput.dispatchEvent(changeEvent);
            }
            
            // Listen for Plotly frame changes
            window.addEventListener('message', function(e) {
                const message = e.data;
                if (message && message.type === 'plotly_animate') {
                    const frameNumber = parseInt(message.frameNumber) || 0;
                    updateStreamlitFrame(frameNumber);
                }
            });
            </script>
            """
            
            # Display the figure and inject JS
            st.plotly_chart(fig, use_container_width=True)
            html(js_code)
            
            # Update session state values based on the current frame
            st.session_state.current_idx = current_frame
            st.session_state.current_actual = actual_values[current_frame]
            st.session_state.current_predicted = pred_values[current_frame]
            st.session_state.current_error = errors[current_frame]
            
            # Current point metrics
            st.subheader("Current Point Details")
            curr_cols = st.columns(3)
            with curr_cols[0]:
                st.metric("Actual Value", f"{st.session_state.current_actual:.2f}")
            with curr_cols[1]:
                st.metric("Predicted Value", f"{st.session_state.current_predicted:.2f}")
            with curr_cols[2]:
                st.metric("Absolute Error", f"{st.session_state.current_error:.2f}")
            
            # Overall metrics
            st.subheader("Overall Performance Metrics")
            overall_cols = st.columns(3)
            with overall_cols[0]:
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            with overall_cols[1]:
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
            with overall_cols[2]:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            
            # Add custom CSS for improved appearance
            st.markdown("""
            <style>
            /* Improve metric styling */
            [data-testid="stMetricValue"] {
                font-size: 1.8rem !important;
                font-weight: bold;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 1rem !important;
            }
            
            /* Make the plot container responsive */
            .js-plotly-plot, .plotly, .plot-container {
                width: 100%;
            }
            
            /* Better button styling */
            .stButton button {
                width: 100%;
                border-radius: 5px;
                height: 3em;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.info("Please upload both 'actual' and 'predicted' CSV files to visualize the data.")

if __name__ == "__main__":
    app()
