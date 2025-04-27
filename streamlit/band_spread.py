import streamlit as st
import matplotlib.pyplot as plt
import utils

def app():
    """Band_Spread analysis page"""
    st.title("Bollinger Band Spread Analysis")
    
    # Get authenticated BigQuery client
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
        
        # info about Band_Spread
        st.markdown("""
        ## What is Band_Spread?
        
        The Bollinger Band Spread (Band_Spread) represents the width of Bollinger Bands, which is a volatility indicator. It is calculated as the 
        difference between the upper and lower Bollinger Bands, divided by the middle band (usually a 20-day moving average).
        
        The Band_Spread provides insight into market volatility:
        - A widening spread (increasing value) indicates increasing market volatility
        - A narrowing spread (decreasing value) indicates decreasing market volatility
        - Extremely low values often precede significant price movements, a phenomenon known as the "Bollinger Band Squeeze"
        
        In gold market analysis, Band_Spread can help identify periods of consolidation before major market moves and gauge overall market uncertainty.
        """)
        
      
        # Run standard analysis for Band_Spread
        utils.run_feature_analysis(data, "Band_Spread")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()
    