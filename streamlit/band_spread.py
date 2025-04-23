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
            
        # Check if data loaded successfully
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # Information about Band_Spread
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
        
        # Add Band_Spread specific visualization
        if "Band_Spread" in data.columns:
            st.subheader("Band Spread Volatility Analysis")
            
            # Calculate average band spread as reference
            avg_spread = data['Band_Spread'].mean()
            high_volatility_threshold = data['Band_Spread'].quantile(0.75)
            low_volatility_threshold = data['Band_Spread'].quantile(0.25)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the band spread
            ax.plot(data['Date'], data['Band_Spread'], color='blue', linewidth=1.5)
            
            # Add reference lines
            ax.axhline(y=avg_spread, color='black', linestyle='-', alpha=0.5, label=f'Average ({avg_spread:.4f})')
            ax.axhline(y=high_volatility_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'High Volatility Threshold ({high_volatility_threshold:.4f})')
            ax.axhline(y=low_volatility_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'Low Volatility Threshold ({low_volatility_threshold:.4f})')
            
            # Highlight high volatility periods
            high_volatility = data['Band_Spread'] > high_volatility_threshold
            if high_volatility.any():
                ax.fill_between(data['Date'], data['Band_Spread'], high_volatility_threshold, 
                               where=(data['Band_Spread'] > high_volatility_threshold),
                               color='red', alpha=0.2, label='High Volatility')
            
            # Highlight low volatility periods (potential squeezes)
            low_volatility = data['Band_Spread'] < low_volatility_threshold
            if low_volatility.any():
                ax.fill_between(data['Date'], data['Band_Spread'], low_volatility_threshold,
                               where=(data['Band_Spread'] < low_volatility_threshold),
                               color='green', alpha=0.2, label='Low Volatility (Potential Squeeze)')
            
            ax.set_title('Bollinger Band Spread (Volatility Indicator)')
            ax.set_ylabel('Band Spread Value')
            ax.set_xlabel('Date')
            ax.legend()
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Calculate percentage of time in high/low volatility
            high_vol_pct = high_volatility.mean() * 100
            low_vol_pct = low_volatility.mean() * 100
            normal_vol_pct = 100 - high_vol_pct - low_vol_pct
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Volatility", f"{high_vol_pct:.2f}%")
            with col2:
                st.metric("Normal Volatility", f"{normal_vol_pct:.2f}%")
            with col3:
                st.metric("Low Volatility", f"{low_vol_pct:.2f}%")
        
        # Run the standard analysis for Band_Spread
        utils.run_feature_analysis(data, "Band_Spread")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your BigQuery connection and try again.")

if __name__ == "__main__":
    app()