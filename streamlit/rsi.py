import streamlit as st
import matplotlib.pyplot as plt
import utils

def app():
    """RSI analysis page"""
    st.title("RSI Analysis")
    
    # Getting authenticated client 
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            data = utils.load_data(client)
            
        # Data load check
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # info about RSI
        st.markdown("""
        ## What is RSI?
        
        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in the gold market.
        
        Traditional interpretation and usage of the RSI:
        - RSI values of 70 or above indicate that gold may be overbought or overvalued, potentially signaling a price correction or reversal.
        - RSI values of 30 or below suggest that gold may be oversold or undervalued, possibly indicating a buying opportunity.
        
        Traders and analysts also look for divergences between the RSI and gold price to identify potential trend changes,
        as well as centerline crossovers (above or below 50) which may confirm trend direction.
        """)
        
        # Running the standard analysis for RSI
        utils.run_feature_analysis(data, "RSI")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()