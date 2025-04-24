import streamlit as st
import models
import bigquery
import eda
import dxy
import dfii10
import vix
import cpi
import ema30
import ema252
import rsi
import band_spread
import sentiment
import os
import importlib

# Configure page
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for the sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: rgba(211, 211, 211, 0.5);
        padding: 2rem 1rem;
    }
    .sidebar-header {
        font-size: 20px;
        margin-bottom: 10px;
    }
    .sidebar-subheader {
        font-size: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the session state if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Models'

# Dictionary of available pages
pages = {
    'Models': models,
    'BigQuery': bigquery,
    'EDA': eda,
    # New feature-specific pages
    'DXY': dxy,
    'DFII10': dfii10,
    'VIX': vix,
    'CPI': cpi,
    'EMA30': ema30,
    'EMA252': ema252,
    'RSI': rsi,
    'Band_Spread': band_spread,
    'Sentiment': sentiment
    # 'Forecast': forecast,
    # 'Corn Sales': corn_sales,
    # 'Weather': weather,
    # 'Ethanol': ethanol,
    # 'Relevant Indices': indices,
    # 'XGBoost Analysis': xgboost_analysis,
    # 'LSTM Analysis': lstm_analysis,
}

# Create the sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subheader">Overview</div>', unsafe_allow_html=True)
    
    # Create actual Streamlit buttons for each page
    for page_name in pages.keys():
        # Use Streamlit buttons that actually respond to clicks
        if st.button(page_name, key=f"btn_{page_name}", 
                     use_container_width=True,
                     type="primary" if st.session_state['current_page'] == page_name else "secondary"):
            st.session_state['current_page'] = page_name
            st.rerun()

# Display the selected page
try:
    pages[st.session_state['current_page']].app()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st.error("This page module might not be implemented yet. Please create the corresponding Python file.")
