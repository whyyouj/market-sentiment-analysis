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
    
    # Getting autheticated bigquery
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Loading data from BQ
        with st.spinner("Loading data from BigQuery..."):
            # Modified query to ensure we have both DFII10 and gold prices
            query = """
            SELECT Date, Price, DFII10
            FROM `IS3107_Project.gold_market_data`
            WHERE DFII10 IS NOT NULL
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
            
        st.success(f"Data loaded successfully! Total records: {len(data)}")
        
        #  DFII10 check
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
        
        # Hypothesis testing section
        st.header("Testing the Inverse Relationship Hypothesis")
        
        # Calculating correlation to put in chart
        correlation = data['DFII10'].corr(data['Price'])
        correlation_text = f"Correlation: {correlation:.3f}"
        relationship_type = "inverse" if correlation < 0 else "direct"
        strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        st.markdown(f"""
        ### Visual Comparison: 10-Year Real Interest Rate vs Gold Prices
        
        Our hypothesis suggests an inverse relationship between 10-year real interest rates and gold prices.
        The actual correlation coefficient is **{correlation:.3f}**, indicating a **{strength} {relationship_type}** relationship.
        """)
        
        # Creating dual axis chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        

        # Formatting the plots
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Gold Price (USD)', color='goldenrod')
        ax1.plot(data['Date'], data['Price'], color='goldenrod', linewidth=2, label='Gold Price')
        ax1.tick_params(axis='y', labelcolor='goldenrod')
        
        # second axis 
        ax2 = ax1.twinx()
        ax2.set_ylabel('10-Year Real Interest Rate (%)', color='navy')
        ax2.plot(data['Date'], data['DFII10'], color='navy', linewidth=2, label='DFII10')
        ax2.tick_params(axis='y', labelcolor='navy')
        
        # Adding grid to help with readability
        ax1.grid(True, alpha=0.3)
        plt.title('Gold Price vs 10-Year Real Interest Rate (DFII10)', fontsize=14)
        plt.figtext(0.15, 0.85, correlation_text, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adding the legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        fig.tight_layout(pad=3)
        st.pyplot(fig)
        
        # Linear Regression Analysis
        st.subheader("Linear Regression Analysis")
        
        # Scatter plot regression
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['DFII10'], data['Price'], alpha=0.6, c='goldenrod')
        
        # Including regression line
        X = data['DFII10'].values.reshape(-1, 1)
        y = data['Price'].values
        
        # Model fit + regression metrics
        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        x_range = np.linspace(data['DFII10'].min(), data['DFII10'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        ax.plot(x_range, y_pred, color='red', linewidth=2)
        
        # Add metrics to the plot
        equation = f"Price = {intercept:.2f} + {slope:.2f} × DFII10"
        r2_text = f"R² = {r_squared:.3f}"
        ax.annotate(equation + "\n" + r2_text, xy=(0.05, 0.95), xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), va='top')
        
        # Adding labels and title to the chart
        ax.set_xlabel('10-Year Real Interest Rate (DFII10)')
        ax.set_ylabel('Gold Price (USD)')
        ax.set_title('Gold Price vs. DFII10: Testing Relationship', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Text to show whether results support or don't support our hypothesis
        if slope < 0:
            st.success(f"The regression slope is negative ({slope:.2f}), supporting the inverse relationship hypothesis.")
        else:
            st.warning(f"The regression slope is positive ({slope:.2f}), which does not support the inverse relationship hypothesis.")

        # Regression with more detail
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        p_value = model_sm.pvalues[1]
        
        # Showing regression summary
        with st.expander("View Detailed Regression Statistics"):
            st.text(model_sm.summary().as_text())
        
        # Tab creation
        tab1, tab2 = st.tabs(["Rolling Correlation", "Hypothesis Testing Summary"])
        
        # First tab
        with tab1:
            # Calculate rolling correlation
            st.subheader("Rolling Correlation Analysis")
            window_size = st.slider("Select rolling window size (days)", 30, 365, 90)
            data['rolling_corr'] = data['DFII10'].rolling(window=window_size).corr(data['Price'])
            fig4, ax = plt.subplots(figsize=(12, 6))

            # start at window_size - 1 so same dimension
            ax.plot(data['Date'].iloc[window_size-1:], data['rolling_corr'].iloc[window_size-1:], 
                color='purple', linewidth=2)

            # Adding the  horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Fill areas based on correlation sign (support / no support)
            ax.fill_between(data['Date'].iloc[window_size-1:], data['rolling_corr'].iloc[window_size-1:], 
                        0, where=(data['rolling_corr'].iloc[window_size-1:] > 0), color='green', alpha=0.3, label='Positive Correlation\n(Does Not Support Hypothesis)')

            ax.fill_between(data['Date'].iloc[window_size-1:], data['rolling_corr'].iloc[window_size-1:], 
                        0, where=(data['rolling_corr'].iloc[window_size-1:] <= 0), color='red', alpha=0.3, label='Negative Correlation\n(Supports Hypothesis)')

            # Adding labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.legend()
            # Title
            plt.title(f'{window_size}-day Rolling Correlation between Gold Price and DFII10')

            # Display relevant statistics
            mean_corr = data['rolling_corr'].iloc[window_size-1:].mean()
            neg_pct = (data['rolling_corr'].iloc[window_size-1:] < 0).mean() * 100
            corr_stats = f"Mean correlation: {mean_corr:.3f} | Negative correlation: {neg_pct:.1f}% of time"

            # Add annotation for correlation stats
            plt.figtext(0.5, 0.01, corr_stats, ha='center', bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))

            fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
            st.pyplot(fig4)

            # Additional interpretation based on rolling correlation
            st.markdown(f"""
            **Rolling Correlation Insights:**
            - The mean correlation between DFII10 and gold prices over this period is **{mean_corr:.3f}**.
            - Gold prices show a negative correlation with DFII10 **{neg_pct:.1f}%** of the time.
            - {'This suggests real interest rates generally have an inverse relationship with gold prices.' if neg_pct > 60 else 
            'This suggests real interest rates do not consistently have an inverse relationship with gold prices.' if neg_pct < 40 else
            'This suggests real interest rates sometimes have an inverse relationship with gold prices, but the relationship is not consistent.'}
            """)
            
        # Second tab
        with tab2:
            st.subheader("Hypothesis Testing Summary")
            
            # Summarizing findings
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is an 
            inverse relationship between DFII10 and gold prices:
            """)
            
            # Inverse relationship support check
            slope_supports = slope < 0
            correlation_supports = correlation < 0
            
            # Creating summary table based on analysis
            results_table = pd.DataFrame({
                'Analysis Method': ['Correlation Analysis', 'Rolling Correlation', 'Linear Regression'],
                'Finding': [
                    f"Correlation coefficient: {correlation:.3f}",
                    f"Mean correlation: {mean_corr:.3f}, Negative: {neg_pct:.1f}% of time",
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
            
            # Count number of hypothesis support
            support_count = results_table['Supports Inverse Hypothesis'].str.contains('Yes').sum()
            total_tests = len(results_table)
            
            # Overall conclusino based on count
            if support_count / total_tests > 0.6:
                st.success(f"The data largely supports the hypothesis of an inverse relationship between DFII10 and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            elif support_count / total_tests > 0.4:
                st.info(f"The data shows mixed evidence regarding the hypothesis of an inverse relationship between DFII10 and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            else:
                st.warning(f"The data largely does not support the hypothesis of an inverse relationship between DFII10 and gold prices. Only {support_count} out of {total_tests} tests provide evidence for this relationship.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
