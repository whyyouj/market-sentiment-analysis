import streamlit as st
import matplotlib.pyplot as plt
import utils

def app():
    """RSI analysis page"""
    st.title("RSI Analysis")
    
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
        
        # Information about RSI
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
        
        # Add RSI specific visualization
        if "RSI" in data.columns:
            st.subheader("RSI Zones")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(data['Date'], data['RSI'], color='blue', linewidth=1.5)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Centerline (50)')
            
            # Fill overbought area (above 70)
            ax.fill_between(data['Date'], 70, 100, color='red', alpha=0.1)
            
            # Fill oversold area (below 30)
            ax.fill_between(data['Date'], 0, 30, color='green', alpha=0.1)
            
            ax.set_title('RSI with Overbought and Oversold Zones')
            ax.set_ylabel('RSI Value')
            ax.set_xlabel('Date')
            ax.legend()
            ax.set_ylim(0, 100)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Calculate percentage of time in each zone
            overbought_pct = (data['RSI'] > 70).mean() * 100
            oversold_pct = (data['RSI'] < 30).mean() * 100
            neutral_pct = 100 - overbought_pct - oversold_pct
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overbought (>70)", f"{overbought_pct:.2f}%")
            with col2:
                st.metric("Neutral (30-70)", f"{neutral_pct:.2f}%")
            with col3:
                st.metric("Oversold (<30)", f"{oversold_pct:.2f}%")
        
        # Run the standard analysis for RSI
        utils.run_feature_analysis(data, "RSI")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()