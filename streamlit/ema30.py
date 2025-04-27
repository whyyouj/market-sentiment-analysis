import streamlit as st
import utils

def app():
    """EMA30 analysis page"""
    st.title("EMA30 Analysis")
    
    # Getting authenticated client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Loading data
        with st.spinner("Loading data from BigQuery..."):
            data = utils.load_data(client)
            
        # Date load check
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # info about EMA30
        st.markdown("""
        ## What is EMA30?
        
        The 30-day Exponential Moving Average (EMA30) is a technical indicator that gives more weight to recent price data. It's calculated
        by applying more weight to the most recent prices while still accounting for older prices with diminishing importance.
        
        The EMA30 responds more quickly to price changes than a simple moving average would, making it useful for identifying shorter-term trends and
        potential buy or sell signals. Traders often compare the current gold price to its EMA30 to gauge market momentum and potential trend changes.
        
        When gold prices cross above the EMA30, it may signal a bullish trend, while crossing below could indicate a bearish trend.
        """)
        
        # Run the standard analysis 
        utils.run_feature_analysis(data, "EMA30")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()