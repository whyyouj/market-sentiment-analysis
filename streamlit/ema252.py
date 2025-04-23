import streamlit as st
import utils

def app():
    """EMA252 analysis page"""
    st.title("EMA252 Analysis")
    
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
        
        # Information about EMA252
        st.markdown("""
        ## What is EMA252?
        
        The 252-day Exponential Moving Average (EMA252) is a technical indicator that tracks the average price of gold over approximately one trading year
        (252 is typically the number of trading days in a year). Like all EMAs, it gives more weight to recent price data.
        
        The EMA252 is considered a long-term indicator that smooths out short-term fluctuations and helps identify the underlying trend in gold prices.
        It's often used as a major support or resistance level and can indicate long-term bullish or bearish sentiment in the gold market.
        
        When gold prices are trading above the EMA252, it generally indicates a long-term bullish trend, while prices below the EMA252 
        may suggest a long-term bearish trend.
        """)
        
        # Run the analysis for EMA252
        utils.run_feature_analysis(data, "EMA252")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()