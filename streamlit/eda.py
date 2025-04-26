import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.cloud import bigquery
import utils
import traceback

def plot_correlation_matrix(data, forward_filled_cols=None):
    """Function that plots correlation matrix for all the features available."""
    
    # Dropping date
    numeric_data = data.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 8))
    
    corr = numeric_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plots heatmap
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, fmt='.2f')
    # Code to mark forward-filled variables here
    if forward_filled_cols:
        for col in forward_filled_cols:
            
            if col in numeric_data.columns:
                
                idx = list(numeric_data.columns).index(col)
                plt.gca().add_patch(plt.Rectangle((0, idx), idx, 1, fill=False, edgecolor='orange', lw=2, clip_on=False))
                
    plt.title('Correlation Matrix')
    # Makes the FF columns more obvious
    if forward_filled_cols:
        plt.figtext(0.5, 0.01, "Variables with orange borders contain forward-filled values, which may affect correlation estimates",ha="center", fontsize=9, bbox={"facecolor":"lightyellow", "alpha":0.8, "pad":5, "edgecolor":"orange"})
    
    plt.tight_layout()
    st.pyplot(plt)


def plot_feature_overview(data, column_name):
    """Plot time series with the code written in utils"""
    utils.plot_time_series(data, column_name)
    
    

def app():
    """Exploratory Data Analysis overview page"""
    
    st.title("Exploratory Data Analysis (EDA)")
    
    # Getting an authenticated BigQuery client from utils
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        with st.spinner("Loading data from BigQuery..."):
            data = utils.load_data(client)
            
        # Data load check
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # Data overview
        st.subheader("Data Overview")
        
        st.dataframe(data.head())
        
        # Summary statistics for all columns (remove date if present)
        st.subheader("Summary Statistics")
        numeric_data = data.select_dtypes(include=['number'])
        st.dataframe(numeric_data.describe())
        
        # Known forward-filled columns
        known_forward_filled = ['CPI']
        
        
        st.subheader("Feature Correlation Matrix")
        plot_correlation_matrix(data, known_forward_filled)
        
       
        st.header("Feature Overview")
        st.write("Click on any feature name in the sidebar to see a detailed analysis of that feature.")
        
        # Mini charts with navigation functionality
        col1, col2 = st.columns(2)
        feature_index = 0
        
        for column in ['DXY', 'DFII10', 'VIX', 'CPI', 'EMA30', 'EMA252', 'RSI', 'Band_Spread']:
            with col1 if feature_index % 2 == 0 else col2:
                st.subheader(column)
                
                if column in data.columns:
                    
                    plot_feature_overview(data, column)
                    if st.button(f"Analyze {column} in Detail", key=f"analyze_{column}"):
                        st.session_state['current_page'] = column
                        st.rerun()
                else:
                    st.warning(f"{column} not found in dataset")
            
            feature_index += 1
    
    except Exception as e:
        st.error(f"An error occurred during the EDA process: {str(e)}")
        
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()