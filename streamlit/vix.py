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
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def app():
    """VIX analysis page"""
    st.title("VIX and Gold Price Analysis")
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            query = """
            SELECT Date, Price, VIX
            FROM `IS3107_Project.gold_market_data`
            WHERE VIX IS NOT NULL AND Price IS NOT NULL
            ORDER BY Date
            """
            data = client.query(query).to_dataframe()
            
        # Data load check
        if data is None or data.empty:
            st.error("No data was retrieved from BigQuery. Please check your query and connection.")
            return
            
        # Datetime format check
        if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
            data['Date'] = pd.to_datetime(data['Date'])
            
        # Remove any remaining nan values
        clean_data = data.dropna(subset=['Date', 'Price', 'VIX']).copy()
        
        # Feature analysis for VIX
        utils.run_feature_analysis(clean_data, 'VIX')
        
        st.success(f"Data loaded successfully! Total records: {len(clean_data)}")
        
        # Information about VIX 
        with st.expander("What is VIX and its relation to gold?"):
            st.markdown("""
            ## VIX and Gold
            
            The VIX (CBOE Volatility Index) is a real-time market index representing the market's expectations for volatility over the coming 30 days.
            Often referred to as the "fear index" or "fear gauge," the VIX is a measure of market risk and investors' sentiments.
            
            ### Hypothesis
            
            Our hypothesis is that there is a **linear relationship** between VIX and gold prices. Specifically, we expect that higher VIX values (indicating greater market uncertainty) will correspond to higher gold prices.
            
            ### Economic Reasoning
            
            This hypothesis is based on economic theory that suggests:
            
            1. **Safe Haven Effect**: Gold is traditionally viewed as a "safe haven" asset during times of market uncertainty and volatility.
            
            2. **Fear Factor**: When the VIX rises (indicating market fear), investors often shift capital toward perceived safe assets like gold.
            
            3. **Portfolio Diversification**: During volatile periods, institutional investors increase gold allocations to diversify risk.
            
            4. **Historical Patterns**: Historically, gold prices have often risen during periods of market stress and high volatility.
            """)
            

        st.header("Testing the Linear Relationship Hypothesis")
        
        
        
        
        # Calculate Pearson correlation coefficient
        correlation = clean_data['VIX'].corr(clean_data['Price'])
        correlation_text = f"Correlation: {correlation:.3f}"
        relationship_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        #Time series plot with dual axis
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot gold price
        color = 'goldenrod'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Gold Price (USD)', color=color)
        ax1.plot(clean_data['Date'], clean_data['Price'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for VIX
        ax2 = ax1.twinx()
        color = 'darkred'
        ax2.set_ylabel('VIX', color=color)
        ax2.plot(clean_data['Date'], clean_data['VIX'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Gold Price and VIX Over Time', fontsize=14)
        plt.figtext(0.15, 0.85, correlation_text, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        fig1.tight_layout()
        st.pyplot(fig1)
        
        
        st.subheader("Linear Relationship Analysis")        
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(clean_data['VIX'], clean_data['Price'], alpha=0.6, c='goldenrod')
        X = clean_data['VIX'].values.reshape(-1, 1)
        y = clean_data['Price'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Getting the regression metrics
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate p-value for the slope
        n = len(clean_data)
        y_pred = model.predict(X)
        residual = y - y_pred
        sse = np.sum(residual**2)
        variance = sse / (n - 2)
        se = np.sqrt(variance / np.sum((X - np.mean(X))**2))
        t_stat = slope / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Generating line predictions
        x_range = np.linspace(clean_data['VIX'].min(), clean_data['VIX'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        # Plotting the regression line
        ax.plot(x_range, y_pred, color='red', linewidth=2)
    
        equation = f"Price = {intercept:.2f} + {slope:.2f} × VIX"
        r2_text = f"R² = {r_squared:.3f}, p-value = {p_value:.4f}"
        ax.annotate(equation + "\n" + r2_text,xy=(0.05, 0.95), xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),va='top')
        
        ax.set_xlabel('VIX Index')
        ax.set_ylabel('Gold Price (USD)')
        ax.set_title('Gold Price vs. VIX: Linear Regression Analysis', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # Insight based on  the regression results
        if p_value < 0.05:
            significance = "statistically significant"
            evidence = "supports"
        else:
            significance = "not statistically significant"
            evidence = "does not support"
            
        st.markdown(f"""
        ### Regression Analysis Summary:
        - The slope coefficient is {slope:.2f}, indicating that for each one-point increase in VIX, gold price changes by ${slope:.2f} on average.
        - This relationship is {significance} (p-value: {p_value:.4f}), which {evidence} our hypothesis of a linear relationship.
        - The R² value of {r_squared:.3f} indicates that VIX explains approximately {r_squared*100:.1f}% of the variation in gold prices.
        - The relationship is {relationship_strength} based on the correlation coefficient of {correlation:.3f}.
        """)
        
        # Create a single tab for rolling correlation
        tab1, tab2 = st.tabs(["Rolling Correlation", "Hypothesis Testing Summary"])
        
        with tab1:
            st.subheader("Rolling Correlation Analysis")
            window_size = st.slider("Select rolling window size (days)", 30, 365, 90)
            clean_data['rolling_corr'] = clean_data['VIX'].rolling(window=window_size).corr(clean_data['Price'])
            valid_indices = clean_data['rolling_corr'].notna()
            fig4, ax = plt.subplots(figsize=(12, 6))
            valid_dates = clean_data.loc[valid_indices, 'Date']
            valid_corr = clean_data.loc[valid_indices, 'rolling_corr']
            
            # Plot the line
            ax.plot(valid_dates, valid_corr, color='purple', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            

            ax.fill_between(valid_dates, valid_corr, 0, where=(valid_corr > 0),color='green', alpha=0.3, label='Positive Correlation\n(Supports Safe-Haven Hypothesis)')
            
            ax.fill_between(valid_dates, valid_corr, 0, where=(valid_corr <= 0),color='red', alpha=0.3, label='Negative Correlation\n(Contradicts Safe-Haven Hypothesis)')
            

            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add title 
            plt.title(f'{window_size}-day Rolling Correlation between Gold Price and VIX')
            
            # Displaying statistics
            mean_corr = valid_corr.mean()
            pos_pct = (valid_corr > 0).mean() * 100
            corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"
            
            # Adding annotation for correlation stats
            plt.figtext(0.5, 0.01, corr_stats, ha='center', 
                        bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
            
            fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
            st.pyplot(fig4)
            
            # Text based on results
            st.markdown(f"""
            **Rolling Correlation Insights:**
            - The mean correlation between VIX and gold prices over this period is **{mean_corr:.3f}**.
            - Gold prices show a positive correlation with VIX **{pos_pct:.1f}%** of the time.
            - {'This suggests gold generally acts as a safe-haven during market volatility.' if pos_pct > 60 else 
              'This suggests gold does not consistently act as a safe-haven during all market volatility periods.' if pos_pct < 40 else
              'This suggests gold sometimes acts as a safe-haven during market volatility, but the relationship is not consistent.'}
            """)
            
        with tab2:
            st.subheader("Hypothesis Testing Summary")
            
            # Summarize findings
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is a 
            linear relationship between VIX and gold prices:
            """)
            
            # Check if correlation, regression support the hypothesis
            correlation_supports = correlation > 0.3
            regression_supports = p_value < 0.05 and slope > 0
            rolling_supports = mean_corr > 0.3
            
            # Create summary table based on results
            results_table = pd.DataFrame({
                'Analysis Method': ['Correlation Analysis', 'Linear Regression', 'Rolling Correlation'],
                'Finding': [
                    f"Correlation coefficient: {correlation:.3f} ({relationship_strength})",
                    f"Slope: {slope:.2f}, R-squared: {r_squared:.3f}, p-value: {p_value:.4f}",
                    f"Mean rolling correlation: {mean_corr:.3f}, Positive: {pos_pct:.1f}% of time"
                ],
                'Supports Linear Relationship': [
                    "✓ Yes" if correlation_supports else "✗ No",
                    "✓ Yes" if regression_supports else "✗ No",
                    "✓ Yes" if rolling_supports else "✗ No"
                ]
            })
            
            st.table(results_table)
            st.subheader("Conclusion")
            
            # Counting number of tests that supports
            support_count = results_table['Supports Linear Relationship'].str.contains('Yes').sum()
            total_tests = len(results_table)
            
            # Overall conclusion based on results
            if support_count / total_tests > 0.6:
                st.success(f"The data largely supports the hypothesis of a linear relationship between VIX and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            elif support_count / total_tests > 0.4:
                st.info(f"The data shows mixed evidence regarding the hypothesis of a linear relationship between VIX and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            else:
                st.warning(f"The data largely does not support the hypothesis of a linear relationship between VIX and gold prices. Only {support_count} out of {total_tests} tests provide evidence for this relationship.")
            
            # Direction of relationship if it exists, check results to determine
            if correlation > 0.1:
                st.success("The relationship appears to be positive, suggesting that higher VIX levels are associated with higher gold prices, consistent with gold's role as a safe-haven asset during periods of market uncertainty.")
            elif correlation < -0.1:
                st.warning("The relationship appears to be negative, contrary to the expected direction. This suggests that higher VIX levels are associated with lower gold prices, which is inconsistent with gold's traditional role as a safe-haven asset.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)  # This will display the full traceback for debugging

if __name__ == "__main__":
    app()
