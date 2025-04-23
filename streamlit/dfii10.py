import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import utils
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

def app():
    """DFII10 analysis page"""
    st.title("DFII10 Analysis")
    
    # Get authenticated BigQuery client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            # Modified query to ensure we have both DFII10 and gold prices
            query = """
            SELECT Date, Price, DFII10
            FROM `IS3107_Project.gold_market_data`
            WHERE DFII10 IS NOT NULL
            ORDER BY Date
            """
            data = client.query(query).to_dataframe()
            
        # Check if data loaded successfully
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        # Ensure date is in datetime format
        if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
            data['Date'] = pd.to_datetime(data['Date'])
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # Information about DFII10
        with st.expander("What is DFII10?"):
            st.markdown("""
            ## What is DFII10?
            
            DFII10 refers to the 10-Year Treasury Inflation-Indexed Security, Constant Maturity rate. 
            This is essentially the real interest rate (adjusted for inflation) on 10-year US Treasury securities.
            
            ### Hypothesis
            
            Our hypothesis is that there is an inverse relationship between the 10-year real interest rates (DFII10) and gold prices.
            
            ### Economic Reasoning
            
            This hypothesis is based on economic theory that suggests:
            
            1. **Opportunity Cost**: When real interest rates rise, the opportunity cost of holding gold (which pays no interest) increases, 
               potentially putting downward pressure on gold prices.
            
            2. **Investment Alternatives**: Higher real rates make interest-bearing assets more attractive compared to non-yielding gold.
            
            3. **USD Strength**: Higher real rates often lead to a stronger US dollar, which typically has an inverse relationship with gold prices.
            """)
        utils.run_feature_analysis(data, "DFII10")
        # NEW SECTION: Clear Hypothesis Visualization
        st.header("Testing the Inverse Relationship Hypothesis")
        
        # Calculate correlation for annotation
        correlation = data['DFII10'].corr(data['Price'])
        correlation_text = f"Correlation: {correlation:.3f}"
        relationship_type = "inverse" if correlation < 0 else "direct"
        strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        st.markdown(f"""
        ### Visual Comparison: 10-Year Real Interest Rate vs Gold Prices
        
        Our hypothesis suggests an inverse relationship between 10-year real interest rates and gold prices.
        The actual correlation coefficient is **{correlation:.3f}**, indicating a **{strength} {relationship_type}** relationship.
        """)
        
        # Create the comparison chart with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Format the plot
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Gold Price (USD)', color='goldenrod')
        ax1.plot(data['Date'], data['Price'], color='goldenrod', linewidth=2, label='Gold Price')
        ax1.tick_params(axis='y', labelcolor='goldenrod')
        
        # Create second y-axis for DFII10
        ax2 = ax1.twinx()
        ax2.set_ylabel('10-Year Real Interest Rate (%)', color='navy')
        ax2.plot(data['Date'], data['DFII10'], color='navy', linewidth=2, label='DFII10')
        ax2.tick_params(axis='y', labelcolor='navy')
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        
        # Add title and annotation showing correlation
        plt.title('Gold Price vs 10-Year Real Interest Rate (DFII10)', fontsize=14)
        plt.figtext(0.5, 0.01, correlation_text, ha='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout(pad=3)
        st.pyplot(fig)
        
        # Add an inverse chart to better visualize the relationship
        st.subheader("Inverted Comparison for Clearer Visualization")
        st.markdown("""
        To better visualize whether the two metrics move in opposite directions,
        the chart below inverts the DFII10 values (multiplies by -1).
        If our hypothesis is correct, the two lines should generally move together.
        """)
        
        # Create the inverted comparison chart
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize both series for better visual comparison
        price_norm = (data['Price'] - data['Price'].min()) / (data['Price'].max() - data['Price'].min())
        dfii_norm = (data['DFII10'] - data['DFII10'].min()) / (data['DFII10'].max() - data['DFII10'].min())
        
        # Plot normalized price
        ax.plot(data['Date'], price_norm, color='goldenrod', linewidth=2, label='Gold Price (normalized)')
        
        # Plot inverted normalized DFII10
        ax.plot(data['Date'], -dfii_norm, color='navy', linewidth=2, linestyle='--', 
                label='Inverted DFII10 (normalized)')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        
        plt.title('Gold Price vs Inverted 10-Year Real Interest Rate', fontsize=14)
        fig2.tight_layout()
        st.pyplot(fig2)
        
        # Conclusion about the hypothesis
        st.markdown(f"""
        ### Conclusion
        
        Based on the visual comparison and correlation analysis:
        
        - The correlation coefficient between DFII10 and gold prices is **{correlation:.3f}**
        - This indicates a **{strength} {relationship_type}** relationship
        - The hypothesis of an inverse relationship is {"supported" if correlation < 0 else "not supported"} by the data
        
        This simple analysis {"confirms" if correlation < 0 else "contradicts"} the economic theory that suggests 
        rising real interest rates tend to put downward pressure on gold prices.
        """)
        
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
