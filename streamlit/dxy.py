import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import utils
import statsmodels.api as sm

def app():
    """DXY analysis page"""
    st.title("DXY and Gold Price Analysis")
    
    # Getting the  authenticated BigQuery client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        with st.spinner("Loading data from BigQuery..."):
            query = """
            SELECT Date, Price, DXY
            FROM `IS3107_Project.gold_market_data`
            WHERE DXY IS NOT NULL
            ORDER BY Date
            """
            data = client.query(query).to_dataframe()
            
        # Data load check
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        # Date format check
        if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Run the feature analysis function
        utils.run_feature_analysis(data, 'DXY')
        
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        # Information about DXY
        with st.expander("What is DXY and its relation to gold?"):
            st.markdown("""
            ## U.S. Dollar Index (DXY) and Gold
            
            The U.S. Dollar Index (DXY) is a measure of the value of the United States dollar relative to a basket of foreign currencies, often referred to as a basket of U.S. trade partners' currencies. The index rises when the U.S. dollar gains strength compared to other currencies.
            
            ### Hypothesis
            
            Our hypothesis is that there is an **inverse relationship** between DXY and gold prices.
            
            ### Economic Reasoning
            
            This hypothesis is based on economic theory that suggests:
            
            1. **Currency Valuation Effect**: Gold is priced in U.S. dollars internationally. When the dollar strengthens (DXY rises), it takes fewer dollars to buy the same amount of gold, which can put downward pressure on the dollar price of gold.
            
            2. **Alternative Investment**: When the U.S. dollar is strong, investors might prefer holding the currency itself or dollar-denominated assets rather than gold, reducing gold demand.
            
            3. **International Purchasing Power**: A stronger dollar makes gold more expensive for holders of other currencies, potentially reducing demand and prices.
            
            4. **Historical Pattern**: Historically, there has often been an observable inverse correlation between dollar strength and gold prices.
            """)
            

        st.header("Testing the Inverse Relationship Hypothesis")
        correlation = data['DXY'].corr(data['Price'])
        correlation_text = f"Correlation: {correlation:.3f}"
        relationship_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        # Plotting a dual time-series line chart
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # 1.  gold price
        color = 'goldenrod'
        ax1.set_xlabel('Date')
        
        ax1.set_ylabel('Gold Price (USD)', color=color)
        ax1.plot(data['Date'], data['Price'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 2.  DXY axis
        ax2 = ax1.twinx()
        color = 'navy'
        ax2.set_ylabel('DXY', color=color)
        
        ax2.plot(data['Date'], data['DXY'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Chart title and annotation
        plt.title('Gold Price and DXY Over Time', fontsize=14)
        
        plt.figtext(0.15, 0.85, correlation_text, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        fig1.tight_layout()
        st.pyplot(fig1)
        
        # Linear regression scatter plot to  test hypothesis
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['DXY'], data['Price'], alpha=0.6, c='goldenrod')
        X = data['DXY'].values.reshape(-1, 1)
        
        y = data['Price'].values
        
        # Basic LR model
        model = LinearRegression()
        
        model.fit(X, y)
        
        # Regression metrics 
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        
        
        
        # Generating predictions for the line
        x_range = np.linspace(data['DXY'].min(), data['DXY'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, color='red', linewidth=2)
        
        # Adding more information to the plot
        equation = f"Price = {intercept:.2f} + {slope:.2f} × DXY"
        r2_text = f"R² = {r_squared:.3f}"
        ax.annotate(equation + "\n" + r2_text, xy=(0.05, 0.95), xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),va='top')
        
        
        # Indicating the label and title
        ax.set_xlabel('U.S. Dollar Index (DXY)')
        ax.set_ylabel('Gold Price (USD)')
        ax.set_title('Gold Price vs. DXY: Testing Relationship', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Text to indicate whether Invverse relationship valid
        if slope < 0:
            st.success(f"The regression slope is negative ({slope:.2f}), supporting the inverse relationship hypothesis.")
        else:
            st.warning(f"The regression slope is positive ({slope:.2f}), which does not support the inverse relationship hypothesis.")
        
        # p-values
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        p_value = model_sm.pvalues[1]
        with st.expander("View Detailed Regression Statistics"):
            st.text(model_sm.summary().as_text())
        
        # Creation of tabs
        tab1, tab2 = st.tabs(["Rolling Correlation", "Advanced Analysis"])
        
        with tab1:
            # Rolling window correlation analysis
            st.subheader("Rolling Correlation Analysis")
            # Common windown sizes
            
            window_size = st.slider("Select rolling window size (days)", 30, 365, 90)
            data['rolling_corr'] = data['DXY'].rolling(window=window_size).corr(data['Price'])

            # Create the rolling correlation plot 
            fig4, ax = plt.subplots(figsize=(12, 6))



            # Ensures they have the same dimension
            ax.plot(data['Date'].iloc[window_size-1:], data['rolling_corr'].iloc[window_size-1:], 
                color='purple', linewidth=2)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Fill areas based on correlation sign for easier visibility
            ax.fill_between(data['Date'].iloc[window_size-1:], 
                        data['rolling_corr'].iloc[window_size-1:], 
                        0, where=(data['rolling_corr'].iloc[window_size-1:] > 0),color='green', alpha=0.3, label='Positive Correlation\n(Supports Safe-Haven Hypothesis)')

            ax.fill_between(data['Date'].iloc[window_size-1:], 
                        data['rolling_corr'].iloc[window_size-1:], 
                        0, where=(data['rolling_corr'].iloc[window_size-1:] <= 0),color='red', alpha=0.3, label='Negative Correlation\n(Contradicts Safe-Haven Hypothesis)')


            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Title will contain window_Size
            plt.title(f'{window_size}-day Rolling Correlation between Gold Price and DXY')



            # Display correlation statistics
            mean_corr = data['rolling_corr'].iloc[window_size-1:].mean()
            pos_pct = (data['rolling_corr'].iloc[window_size-1:] > 0).mean() * 100
            
            corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"

            # Add annotation for correlation stats
            
            plt.figtext(0.5, 0.01, corr_stats, ha='center', bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))

            fig4.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            st.pyplot(fig4)

            # Additional insights
            
            st.markdown(f"""
            **Rolling Correlation Insights:**
            - The mean correlation between DXY and gold prices over this period is **{mean_corr:.3f}**.
            - Gold prices show a positive correlation with VIX **{pos_pct:.1f}%** of the time.
            - {'This suggests gold generally acts as a safe-haven during market volatility.' if pos_pct > 60 else 
            'This suggests gold does not consistently act as a safe-haven during all market volatility periods.' if pos_pct < 40 else
            'This suggests gold sometimes acts as a safe-haven during market volatility, but the relationship is not consistent.'}
            """)
        
        with tab2:
            st.subheader("Hypothesis Testing Summary")
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is an 
            inverse relationship between DXY and gold prices:
            """)
            
            # Check if slope is negative 
            
            slope_supports = slope < 0
            # Check if correlation is negative
            
            correlation_supports = correlation < 0
            
            # Lets you see the results easily 
            results_table = pd.DataFrame({
                'Analysis Method': ['Correlation Analysis', 'Rolling Correlation', 'Linear Regression'],
                'Finding': [
                    f"Correlation coefficient: {correlation:.3f}",
                    f"Mean correlation: {mean_corr:.3f}, Positive: {pos_pct:.1f}% of time",
                    f"Slope: {slope:.2f}, R-squared: {r_squared:.3f}"
                ], 
                'Supports Inverse Hypothesis': [
                    "✓ Yes" if correlation_supports else "✗ No",
                    "✓ Yes" if mean_corr < 0 else "✗ No",
                    "✓ Yes" if slope_supports else "✗ No"
                ]
            })
            
            st.table(results_table)
            st.subheader("Conclusion")
            
            # Support count
            if 'results_table' in locals():
                support_count = results_table['Supports Inverse Hypothesis'].str.contains('Yes').sum()
                total_tests = len(results_table)
                
                if support_count / total_tests > 0.6:
                    st.success(f"The data largely supports the hypothesis of an inverse relationship between DXY and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
                elif support_count / total_tests > 0.4:
                    
                    st.info(f"The data shows mixed evidence regarding the hypothesis of an inverse relationship between DXY and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
                else:
                    st.warning(f"The data largely does not support the hypothesis of an inverse relationship between DXY and gold prices. Only {support_count} out of {total_tests} tests provide evidence for this relationship.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
    
if __name__ == "__main__":
    app()
