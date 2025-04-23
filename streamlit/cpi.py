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
        
        # Add normalized comparison for clearer visual analysis
        st.subheader("Normalized Comparison")
        
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize both series for better comparison
        cpi_norm = (data['CPI'] - data['CPI'].min()) / (data['CPI'].max() - data['CPI'].min())
        price_norm = (data['Price'] - data['Price'].min()) / (data['Price'].max() - data['Price'].min())
        
        # Plot normalized values
        ax.plot(data['Date'], price_norm, color='goldenrod', linewidth=2, label='Gold Price (normalized)')
        ax.plot(data['Date'], cpi_norm, color='firebrick', linewidth=2, linestyle='--', label='CPI (normalized)')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        
        plt.title('Normalized Gold Price vs CPI', fontsize=14)
        fig3.tight_layout()
        st.pyplot(fig3)
        
        # More detailed regression with statsmodels for p-values
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        # Extract p-value for CPI coefficient
        p_value = model_sm.pvalues[1]
        
        # Display regression summary in expandable section
        with st.expander("View Detailed Regression Statistics"):
            st.text(model_sm.summary().as_text())
        
        # Conclusion about the hypothesis
        st.markdown(f"""
        ### Conclusion on Linear Relationship Hypothesis
        
        Based on our analysis of the relationship between CPI and gold prices:
        
        1. **Correlation Analysis**:
           - Pearson correlation coefficient: **{correlation:.3f}**
           - This indicates a **{relationship_strength} {"positive" if correlation > 0 else "negative"}** correlation
        
        2. **Linear Regression Analysis**:
           - R-squared value: **{r_squared:.3f}**
           - This means CPI explains approximately **{r_squared*100:.1f}%** of the variance in gold prices
           - Regression equation: **{equation}**
           - Statistical significance: p-value = **{p_value:.4f}** ({"significant" if p_value < 0.05 else "not significant"} at 0.05 level)
        
        3. **Interpretation**:
           - The data {"supports" if r_squared > 0.3 and p_value < 0.05 else "partially supports" if (r_squared > 0.1 and r_squared <= 0.3) and p_value < 0.05 else "does not support"} our hypothesis of a linear relationship between CPI and gold prices
           - For each unit increase in CPI, gold price {"increases" if slope > 0 else "decreases"} by approximately **${abs(slope):.2f}**
        
        This analysis {"confirms" if r_squared > 0.3 and p_value < 0.05 else "provides some evidence for" if (r_squared > 0.1 and r_squared <= 0.3) and p_value < 0.05 else "does not confirm"} the economic theory that gold prices tend to move in relation to inflation metrics like CPI.
        """)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "CPI Change Analysis", "Advanced Analysis"])
        
        with tab1:
            st.subheader("Time Series Analysis")
            
            # Calculate percentage changes
            data['CPI_pct_change'] = data['CPI'].pct_change().multiply(100)
            data['Price_pct_change'] = data['Price'].pct_change().multiply(100)
            
            # Plot percentage changes
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Date'], data['Price_pct_change'], label='Gold Price % Change', color='goldenrod')
            ax.plot(data['Date'], data['CPI_pct_change'], label='CPI % Change', color='firebrick')
            ax.set_xlabel('Date')
            ax.set_ylabel('Percentage Change (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.title('Gold Price vs CPI: Percentage Changes Over Time')
            
            st.pyplot(fig)
            
            # Calculate rolling correlation
            window_sizes = [30, 60, 90, 180, 365]
            selected_window = st.selectbox(
                "Select rolling window size (days):",
                options=window_sizes,
                index=2
            )
            
            # Calculate rolling correlation
            data['Rolling_Correlation'] = data['Price'].rolling(window=selected_window).corr(data['CPI'])
            
            # Plot rolling correlation
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Date'], data['Rolling_Correlation'])
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{selected_window}-day Rolling Correlation')
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            plt.title(f'{selected_window}-day Rolling Correlation between Gold Price and CPI')
            
            st.pyplot(fig)
        
        with tab2:
            st.subheader("CPI Change Analysis")
            
            # Create CPI change categories
            st.markdown("""
            Let's analyze how gold prices react to different levels of CPI changes:
            """)
            
            # Filter out rows with NaN in pct_change
            filtered_data = data.dropna(subset=['CPI_pct_change', 'Price_pct_change'])
            
            # Create CPI change categories
            q1, q3 = filtered_data['CPI_pct_change'].quantile([0.25, 0.75])
            # Ensure unique bin edges
            if q1 == q3:
                # Handle the case where q1 equals q3 (add a small offset)
                bins = [-float('inf'), q1, q1 + 0.00001, float('inf')]
            else:
                bins = [-float('inf'), q1, q3, float('inf')]
                
            filtered_data['CPI_Change_Category'] = pd.cut(
                filtered_data['CPI_pct_change'],
                bins=bins,
                labels=['Low', 'Medium', 'High']
            )

            
            # Calculate average gold price change by CPI category
            category_analysis = filtered_data.groupby('CPI_Change_Category')['Price_pct_change'].agg(['mean', 'std', 'count'])
            category_analysis.columns = ['Average Gold Price Change (%)', 'Standard Deviation', 'Count']
            category_analysis = category_analysis.reset_index()
            
            st.table(category_analysis)
            
            # Create boxplot of gold price changes by CPI category
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='CPI_Change_Category', y='Price_pct_change', data=filtered_data, ax=ax)
            ax.set_xlabel('CPI Change Category')
            ax.set_ylabel('Gold Price Change (%)')
            ax.set_title('Gold Price Changes by CPI Change Category')
            
            st.pyplot(fig)
            
            # ANOVA test to see if the differences are statistically significant
            if len(filtered_data['CPI_Change_Category'].unique()) > 1:
                try:
                    groups = [filtered_data[filtered_data['CPI_Change_Category'] == cat]['Price_pct_change'] 
                             for cat in filtered_data['CPI_Change_Category'].unique() if not filtered_data[filtered_data['CPI_Change_Category'] == cat].empty]
                    
                    anova_result = stats.f_oneway(*groups)
                    
                    st.markdown(f"""
                    **ANOVA Test Results:**
                    
                    F-statistic: {anova_result.statistic:.4f}
                    p-value: {anova_result.pvalue:.4f}
                    
                    The difference in gold price changes between CPI change categories is 
                    {"statistically significant" if anova_result.pvalue < 0.05 else "not statistically significant"}.
                    """)
                except Exception as e:
                    st.error(f"Could not perform ANOVA test: {str(e)}")
        
        with tab3:
            st.subheader("Advanced Analysis")
            
            # Granger Causality Test
            st.markdown("### Granger Causality Test")
            st.markdown("""
            The Granger Causality Test examines whether one time series is useful in forecasting another. 
            It helps determine if changes in CPI "cause" changes in gold prices or vice versa.
            """)
            
            max_lag = st.slider("Select maximum lag for Granger Causality Test (days)", 1, 30, 10)
            
            # Prepare data for Granger causality test (dropna)
            granger_data = data.dropna().copy()
            
            # Create lagged dataset
            granger_series = pd.DataFrame({'CPI': granger_data['CPI'], 'Gold_Price': granger_data['Price']})
            
            # Run the tests
            with st.spinner("Running Granger Causality Tests."):
                try:
                    # Test if CPI Granger-causes Gold Price
                    cpi_causes_gold = grangercausalitytests(granger_series[['Gold_Price', 'CPI']], maxlag=max_lag, verbose=False)
                    
                    # Test if Gold Price Granger-causes CPI
                    gold_causes_cpi = grangercausalitytests(granger_series[['CPI', 'Gold_Price']], maxlag=max_lag, verbose=False)
                    
                    # Extract p-values
                    cpi_to_gold_pvals = [cpi_causes_gold[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                    gold_to_cpi_pvals = [gold_causes_cpi[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                    
                    # Display results
                    granger_results = pd.DataFrame({
                        'Lag': list(range(1, max_lag+1)),
                        'CPI causes Gold Price (p-value)': cpi_to_gold_pvals,
                        'Gold Price causes CPI (p-value)': gold_to_cpi_pvals,
                    })
                    
                    # Add significance columns
                    granger_results['CPI→Gold Significant'] = granger_results['CPI causes Gold Price (p-value)'] < 0.05
                    granger_results['Gold→CPI Significant'] = granger_results['Gold Price causes CPI (p-value)'] < 0.05
                    
                    st.dataframe(granger_results)
                    
                    # Interpreting Granger causality results
                    cpi_causes_gold_sig = any(p < 0.05 for p in cpi_to_gold_pvals)
                    gold_causes_cpi_sig = any(p < 0.05 for p in gold_to_cpi_pvals)
                    
                    causality_conclusion = ""
                    if cpi_causes_gold_sig and gold_causes_cpi_sig:
                        causality_conclusion = "Bidirectional causality: CPI and gold prices Granger-cause each other."
                    elif cpi_causes_gold_sig:
                        causality_conclusion = "Unidirectional causality: CPI Granger-causes gold prices."
                    elif gold_causes_cpi_sig:
                        causality_conclusion = "Unidirectional causality: Gold prices Granger-cause CPI."
                    else:
                        causality_conclusion = "No Granger causality detected between CPI and gold prices."
                    
                    st.markdown(f"**Conclusion:** {causality_conclusion}")
                    
                except Exception as e:
                    st.error(f"Error in Granger causality test: {e}")
            
            # Final Summary
            st.subheader("Hypothesis Testing Summary")
            
            # Summarize all findings
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is a 
            linear relationship between CPI and gold prices:
            """)
            
            results_table = pd.DataFrame({
                'Analysis Method': ['Correlation Analysis', 'Linear Regression', 'Granger Causality', 'CPI Change Categories'],
                'Finding': [
                    f"Correlation coefficient: {correlation:.3f}",
                    f"R-squared: {r_squared:.3f}, Slope: {slope:.2f}",
                    causality_conclusion if 'causality_conclusion' in locals() else "Not computed",
                    "See CPI Change Analysis tab for details"
                ],
                'Supports Hypothesis': [
                    "✓ Yes" if abs(correlation) > 0.3 else "✗ No",
                    "✓ Yes" if r_squared > 0.3 and p_value < 0.05 else "✗ No",
                    "✓ Yes" if 'cpi_causes_gold_sig' in locals() and cpi_causes_gold_sig else "✗ No",
                    "✓ Yes" if 'anova_result' in locals() and anova_result.pvalue < 0.05 else "✗ No"
                ]
            })
            
            st.table(results_table)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
if __name__ == "__main__":
    app()