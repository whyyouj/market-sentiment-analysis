import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.cloud import bigquery
import utils

def plot_correlation_matrix(data, forward_filled_cols=None):
    """Function to plot correlation matrix of all columns."""
    # Drop Date column if it exists before calculating correlation
    numeric_data = data.select_dtypes(include=['number'])
    
    plt.figure(figsize=(12, 8))
    corr = numeric_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot the heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, fmt='.2f')
    
    # Mark forward-filled variables if provided
    if forward_filled_cols:
        for col in forward_filled_cols:
            if col in numeric_data.columns:
                idx = list(numeric_data.columns).index(col)
                plt.gca().add_patch(plt.Rectangle((0, idx), idx, 1, 
                                   fill=False, edgecolor='orange', lw=2, clip_on=False))
    
    plt.title('Correlation Matrix')
    if forward_filled_cols:
        plt.figtext(0.5, 0.01, "Variables with orange borders contain forward-filled values, which may affect correlation estimates",
                   ha="center", fontsize=9, bbox={"facecolor":"lightyellow", "alpha":0.8, "pad":5, "edgecolor":"orange"})
    
    plt.tight_layout()
    st.pyplot(plt)

def plot_feature_overview(data, column_name):
    """Plot a mini overview chart for a feature"""
    plt.figure(figsize=(8, 3))
    plt.plot(data['Date'], data[column_name], linewidth=1)
    plt.title(f'{column_name} Overview')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def app():
    """Exploratory Data Analysis overview page"""
    st.title("Exploratory Data Analysis (EDA)")
    
    # Get authenticated BigQuery client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            data = utils.load_data(client)
            
        # Check if data loaded successfully
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # Show overview of the data
        st.subheader("Data Overview")
        st.dataframe(data.head())
        
        # Summary statistics for all columns (excluding Date if present)
        st.subheader("Summary Statistics")
        numeric_data = data.select_dtypes(include=['number'])
        st.dataframe(numeric_data.describe())
        
        # Known forward-filled columns
        known_forward_filled = ['CPI']
        
        # Create a sidebar section for forward-fill info
        # with st.sidebar:
        #     st.subheader("About Forward-Filled Data")
        #     st.write("""
        #     Some variables like CPI are forward-filled, meaning the last known value is carried forward until a new value is available.
            
        #     This creates a 'step' pattern in time series and can affect:
        #     - Distribution shapes
        #     - Outlier detection
        #     - Correlation calculations
        #     """)
            
        #     # Allow user to mark other columns as forward-filled
        #     st.subheader("Forward-Filled Variables")
        #     forward_filled_cols = []
        #     for col in ['DXY', 'DFII10', 'VIX', 'CPI', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']:
        #         is_forward_filled = col in known_forward_filled
        #         if st.checkbox(f"{col} contains forward-filled values", value=is_forward_filled):
        #             forward_filled_cols.append(col)
        
        # Optional correlation matrix for all features
        st.subheader("Feature Correlation Matrix")
        # plot_correlation_matrix(data, forward_filled_cols)
        plot_correlation_matrix(data, known_forward_filled)
        
        # Detect forward filling in columns
        # st.subheader("Forward Fill Detection")
        # st.write("The following columns may contain forward-filled values (based on repeated values):")
        
        # detected_cols = []
        # forward_fill_stats = {}
        
        # for col in ['DXY', 'DFII10', 'VIX', 'CPI', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']:
        #     is_ff, pct = utils.detect_potential_forward_filling(data, col)
        #     forward_fill_stats[col] = pct
        #     if is_ff:
        #         detected_cols.append(col)
        
        # # Display forward fill statistics
        # stats_df = pd.DataFrame({
        #     'Column': forward_fill_stats.keys(),
        #     'Consecutive Identical Values (%)': [f"{v:.1f}%" for v in forward_fill_stats.values()]
        # })
        # st.dataframe(stats_df.sort_values('Column'))
        
        # if detected_cols:
        #     st.info(f"Detected potential forward filling in: {', '.join(detected_cols)}")
        # else:
        #     st.success("No additional columns with significant forward filling detected.")
        
        # Feature Overview Section
        st.header("Feature Overview")
        st.write("Click on any feature name in the sidebar to see a detailed analysis of that feature.")
        
        # Create grid for mini charts
        col1, col2 = st.columns(2)
        feature_index = 0
        
        for column in ['DXY', 'DFII10', 'VIX', 'CPI', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']:
            # Alternate between columns
            with col1 if feature_index % 2 == 0 else col2:
                st.subheader(column)
                if column in data.columns:
                    plot_feature_overview(data, column)
                    
                    # Add a button to navigate to the feature page
                    if st.button(f"Analyze {column} in Detail", key=f"analyze_{column}"):
                        st.session_state['current_page'] = column
                        st.rerun()
                else:
                    st.warning(f"{column} not found in dataset")
            
            feature_index += 1
    
    except Exception as e:
        st.error(f"An error occurred during the EDA process: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()