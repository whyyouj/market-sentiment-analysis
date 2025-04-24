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
    """CPI analysis page"""
    st.title("CPI and Gold Price Analysis")
    
    # Get authenticated BigQuery client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            query = """
            SELECT Date, Price, CPI
            FROM `IS3107_Project.gold_market_data`
            WHERE CPI IS NOT NULL
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
        
        # Information about CPI
        with st.expander("What is CPI and its relation to gold?"):
            st.markdown("""
            ## Consumer Price Index (CPI) and Gold
            
            The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. It is one of the most widely used indicators of inflation.
            
            ### Hypothesis
            
            Our hypothesis is that there is a linear relationship between CPI and gold prices.
            
            ### Economic Reasoning
            
            This hypothesis is based on economic theory that suggests:
            
            1. **Inflation Hedge**: Gold is often considered a hedge against inflation. As inflation (measured by CPI) rises, investors may turn to gold to preserve purchasing power.
            
            2. **Store of Value**: During periods of high inflation, gold's perceived role as a store of value can increase demand and potentially drive up prices.
            
            3. **Historical Correlation**: There is historical evidence suggesting a relationship between inflation metrics and gold prices, though the strength and consistency of this relationship have varied over time.
            """)
        
        # Run the feature analysis function first
        utils.run_feature_analysis(data,"CPI")
        
        # NEW SECTION: Clear Hypothesis Visualization
        st.header("Testing the Linear Relationship Hypothesis")
        
        # Calculate Pearson correlation coefficient
        correlation = data['CPI'].corr(data['Price'])
        correlation_text = f"Correlation: {correlation:.3f}"
        relationship_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        # Create a dual y-axis time series plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot gold price
        color = 'goldenrod'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Gold Price (USD)', color=color)
        ax1.plot(data['Date'], data['Price'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for CPI
        ax2 = ax1.twinx()
        color = 'firebrick'
        ax2.set_ylabel('CPI', color=color)
        ax2.plot(data['Date'], data['CPI'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add a title and annotation
        plt.title('Gold Price and CPI Over Time', fontsize=14)
        plt.figtext(0.15, 0.85, correlation_text, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        fig1.tight_layout()
        st.pyplot(fig1)
        
        # Linear regression scatter plot to explicitly test the hypothesis
        st.subheader("Linear Relationship Analysis")
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        ax.scatter(data['CPI'], data['Price'], alpha=0.6, c='goldenrod')
        
        # Add regression line
        X = data['CPI'].values.reshape(-1, 1)
        y = data['Price'].values
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get regression metrics
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Generate predictions for the line
        x_range = np.linspace(data['CPI'].min(), data['CPI'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        # Plot the regression line
        ax.plot(x_range, y_pred, color='red', linewidth=2)
        
        # Add equation and R² to the plot
        equation = f"Price = {intercept:.2f} + {slope:.2f} × CPI"
        r2_text = f"R² = {r_squared:.3f}"
        ax.annotate(equation + "\n" + r2_text,
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   va='top')
        
        # Set labels and title
        ax.set_xlabel('Consumer Price Index (CPI)')
        ax.set_ylabel('Gold Price (USD)')
        ax.set_title('Gold Price vs. CPI: Testing Linear Relationship', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # More detailed regression with statsmodels for p-values
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        # Extract p-value for CPI coefficient
        p_value = model_sm.pvalues[1]
        
        # Display regression summary in expandable section
        with st.expander("View Detailed Regression Statistics"):
            st.text(model_sm.summary().as_text())
        
        # Create tabs for additional analyses
        tab1, tab2 = st.tabs(["Rolling Correlation", "Hypothesis Testing Summary"])
        
        with tab1:
            st.subheader("Rolling Correlation Analysis")
            
            # Add rolling window selector
            window_size = st.slider("Select rolling window size (days)", 30, 365, 90)
            
            # Calculate rolling correlation
            data['rolling_corr'] = data['CPI'].rolling(window=window_size).corr(data['Price'])
            
            # Identify valid indices where we have correlation values
            valid_indices = data['rolling_corr'].notna()
            
            # Create the rolling correlation plot with properly aligned data
            fig4, ax = plt.subplots(figsize=(12, 6))
            
            # Use only non-NaN values for plotting
            valid_dates = data.loc[valid_indices, 'Date']
            valid_corr = data.loc[valid_indices, 'rolling_corr']
            
            # Plot the line
            ax.plot(valid_dates, valid_corr, color='purple', linewidth=2)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Fill areas based on correlation sign
            ax.fill_between(valid_dates, valid_corr, 0, 
                           where=(valid_corr > 0),
                           color='green', alpha=0.3, label='Positive Correlation\n(Supports Gold as Inflation Hedge)')
            
            ax.fill_between(valid_dates, valid_corr, 0, 
                           where=(valid_corr <= 0),
                           color='red', alpha=0.3, label='Negative Correlation\n(Contradicts Gold as Inflation Hedge)')
            
            # Add visualization enhancements
            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add title with window size information
            plt.title(f'{window_size}-day Rolling Correlation between Gold Price and CPI')
            
            # Display correlation statistics
            mean_corr = valid_corr.mean()
            pos_pct = (valid_corr > 0).mean() * 100
            corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"
            
            # Add annotation for correlation stats
            plt.figtext(0.5, 0.01, corr_stats, ha='center', 
                        bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
            
            fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
            st.pyplot(fig4)
            
            # Additional interpretation based on rolling correlation
            st.markdown(f"""
            **Rolling Correlation Insights:**
            - The mean correlation between CPI and gold prices over this period is **{mean_corr:.3f}**.
            - Gold prices show a positive correlation with CPI **{pos_pct:.1f}%** of the time.
            - {'This suggests gold generally acts as an inflation hedge.' if pos_pct > 60 else 
              'This suggests gold does not consistently act as an inflation hedge in all periods.' if pos_pct < 40 else
              'This suggests gold sometimes acts as an inflation hedge, but the relationship is not consistent.'}
            """)
        
        with tab2:
            # Final Summary
            st.subheader("Hypothesis Testing Summary")
            
            # Summarize all findings
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is a linear relationship between CPI and gold prices:
            """)
            
            # Check if correlation, regression support the hypothesis
            correlation_supports = correlation > 0.3
            regression_supports = p_value < 0.05 and slope > 0
            rolling_supports = mean_corr > 0.3
            
            # Create summary table
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
            
            # Final conclusion
            st.subheader("Conclusion")
            
            # Count how many tests support the hypothesis
            support_count = results_table['Supports Linear Relationship'].str.contains('Yes').sum()
            total_tests = len(results_table)
            
            # Provide overall conclusion
            if support_count / total_tests > 0.6:
                st.success(f"The data largely supports the hypothesis of a linear relationship between CPI and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            elif support_count / total_tests > 0.4:
                st.info(f"The data shows mixed evidence regarding the hypothesis of a linear relationship between CPI and gold prices. {support_count} out of {total_tests} tests provide evidence for this relationship.")
            else:
                st.warning(f"The data largely does not support the hypothesis of a linear relationship between CPI and gold prices. Only {support_count} out of {total_tests} tests provide evidence for this relationship.")
            
            # Direction of relationship if it exists
            if correlation > 0.1:
                st.success("The relationship appears to be positive, suggesting that higher CPI levels are associated with higher gold prices, consistent with gold's role as an inflation hedge.")
            elif correlation < -0.1:
                st.warning("The relationship appears to be negative, contrary to the expected direction. This suggests that higher CPI levels are associated with lower gold prices, which is inconsistent with gold's traditional role as an inflation hedge.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    app()
