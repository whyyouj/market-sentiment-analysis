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
    """DXY analysis page"""
    st.title("DXY and Gold Price Analysis")
    
    # Get authenticated BigQuery client
    client = utils.get_bigquery_client()
    
    if client is None:
        st.error("Could not create BigQuery client. Please check your credentials.")
        return
    
    try:
        # Load data
        with st.spinner("Loading data from BigQuery..."):
            query = """
            SELECT Date, Price, DXY
            FROM `IS3107_Project.gold_market_data`
            WHERE DXY IS NOT NULL
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
            
        # Clear Hypothesis Visualization
        st.header("Testing the Inverse Relationship Hypothesis")
        
        # Calculate Pearson correlation coefficient
        correlation = data['DXY'].corr(data['Price'])
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
        
        # Create second y-axis for DXY
        ax2 = ax1.twinx()
        color = 'navy'
        ax2.set_ylabel('DXY', color=color)
        ax2.plot(data['Date'], data['DXY'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add a title and annotation
        plt.title('Gold Price and DXY Over Time', fontsize=14)
        plt.figtext(0.15, 0.85, correlation_text, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        fig1.tight_layout()
        st.pyplot(fig1)
        
        # Linear regression scatter plot to explicitly test the hypothesis
        st.subheader("Inverse Relationship Analysis")
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        ax.scatter(data['DXY'], data['Price'], alpha=0.6, c='goldenrod')
        
        # Add regression line
        X = data['DXY'].values.reshape(-1, 1)
        y = data['Price'].values
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get regression metrics
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Generate predictions for the line
        x_range = np.linspace(data['DXY'].min(), data['DXY'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        # Plot the regression line
        ax.plot(x_range, y_pred, color='red', linewidth=2)
        
        # Add equation and R² to the plot
        equation = f"Price = {intercept:.2f} + {slope:.2f} × DXY"
        r2_text = f"R² = {r_squared:.3f}"
        ax.annotate(equation + "\n" + r2_text,
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   va='top')
        
        # Set labels and title
        ax.set_xlabel('U.S. Dollar Index (DXY)')
        ax.set_ylabel('Gold Price (USD)')
        ax.set_title('Gold Price vs. DXY: Testing Inverse Relationship', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Add normalized comparison for clearer visual analysis
        st.subheader("Normalized Comparison")
        
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize both series for better comparison
        dxy_norm = (data['DXY'] - data['DXY'].min()) / (data['DXY'].max() - data['DXY'].min())
        price_norm = (data['Price'] - data['Price'].min()) / (data['Price'].max() - data['Price'].min())
        
        # Plot normalized values
        ax.plot(data['Date'], price_norm, color='goldenrod', linewidth=2, label='Gold Price (normalized)')
        ax.plot(data['Date'], dxy_norm, color='navy', linewidth=2, linestyle='--', label='DXY (normalized)')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        
        inverse_correlation = "The normalized chart shows " + (
            "a clear inverse relationship where DXY and gold prices move in opposite directions." 
            if correlation < -0.3 else 
            "a weak or inconsistent inverse relationship between DXY and gold prices."
        )
        
        plt.title('Normalized Gold Price vs DXY', fontsize=14)
        plt.figtext(0.5, 0.01, inverse_correlation, ha='center', 
                    bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
        
        fig3.tight_layout()
        st.pyplot(fig3)
        
        # Emphasize the concept of an inverse relationship
        if slope < 0:
            st.success(f"The regression slope is negative ({slope:.2f}), supporting the inverse relationship hypothesis.")
        else:
            st.warning(f"The regression slope is positive ({slope:.2f}), which does not support the inverse relationship hypothesis.")
        
        # More detailed regression with statsmodels for p-values
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        # Extract p-value for DXY coefficient
        p_value = model_sm.pvalues[1]
        
        # Display regression summary in expandable section
        with st.expander("View Detailed Regression Statistics"):
            st.text(model_sm.summary().as_text())
        
        # Create tabs for additional analyses
        tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "DXY Change Analysis", "Advanced Analysis"])
        
        with tab1:
            st.subheader("Time Series Analysis")
            
            # Calculate percentage changes
            data['DXY_pct_change'] = data['DXY'].pct_change() * 100
            data['Price_pct_change'] = data['Price'].pct_change() * 100
            
            # Plot percentage changes
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Date'].iloc[1:], data['Price_pct_change'].iloc[1:], label='Gold Price % Change', color='goldenrod', alpha=0.7)
            ax.plot(data['Date'].iloc[1:], data['DXY_pct_change'].iloc[1:], label='DXY % Change', color='navy', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Percentage Change')
            ax.legend()
            ax.set_title('Daily Percentage Changes: Gold Price vs DXY')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Rolling window correlation analysis
            st.subheader("Rolling Correlation Analysis")

            # Add rolling window selector
            window_size = st.slider("Select rolling window size (days)", 30, 365, 90)

            # Calculate rolling correlation
            data['rolling_corr'] = data['DXY'].rolling(window=window_size).corr(data['Price'])

            # Create the rolling correlation plot with properly aligned data
            fig4, ax = plt.subplots(figsize=(12, 6))

            # Use window_size-1 as the starting point for both arrays to ensure they have the same dimension
            ax.plot(data['Date'].iloc[window_size-1:], data['rolling_corr'].iloc[window_size-1:], 
                color='purple', linewidth=2)

            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Fill areas based on correlation sign - positive correlation supports the safe-haven hypothesis
            ax.fill_between(data['Date'].iloc[window_size-1:], 
                        data['rolling_corr'].iloc[window_size-1:], 
                        0, 
                        where=(data['rolling_corr'].iloc[window_size-1:] > 0),
                        color='green', alpha=0.3, label='Positive Correlation\n(Supports Safe-Haven Hypothesis)')

            ax.fill_between(data['Date'].iloc[window_size-1:], 
                        data['rolling_corr'].iloc[window_size-1:], 
                        0, 
                        where=(data['rolling_corr'].iloc[window_size-1:] <= 0),
                        color='red', alpha=0.3, label='Negative Correlation\n(Contradicts Safe-Haven Hypothesis)')

            # Add visualization enhancements
            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add title with window size information
            plt.title(f'{window_size}-day Rolling Correlation between Gold Price and DXY')

            # Display correlation statistics
            mean_corr = data['rolling_corr'].iloc[window_size-1:].mean()
            pos_pct = (data['rolling_corr'].iloc[window_size-1:] > 0).mean() * 100
            corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"

            # Add annotation for correlation stats
            plt.figtext(0.5, 0.01, corr_stats, ha='center', 
                        bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))

            fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
            st.pyplot(fig4)

            # Additional interpretation based on rolling correlation
            st.markdown(f"""
            **Rolling Correlation Insights:**
            - The mean correlation between DXY and gold prices over this period is **{mean_corr:.3f}**.
            - Gold prices show a positive correlation with VIX **{pos_pct:.1f}%** of the time.
            - {'This suggests gold generally acts as a safe-haven during market volatility.' if pos_pct > 60 else 
            'This suggests gold does not consistently act as a safe-haven during all market volatility periods.' if pos_pct < 40 else
            'This suggests gold sometimes acts as a safe-haven during market volatility, but the relationship is not consistent.'}
            """)
        
        with tab2:
            st.subheader("DXY Change Analysis")
            
            # Create DXY change categories
            st.markdown("""
            Let's analyze how gold prices react to different levels of DXY changes:
            """)
            
            # Filter out rows with NaN in pct_change
            filtered_data = data.dropna(subset=['DXY_pct_change', 'Price_pct_change'])
            
            # Create DXY change categories
            q1, q3 = filtered_data['DXY_pct_change'].quantile([0.25, 0.75])
            
            # Ensure bin edges are unique to prevent the error from before
            if q1 == q3:
                # If quartiles are equal, create slightly different bin edges
                bins = [-float('inf'), q1-0.00001, q1+0.00001, float('inf')]
            else:
                bins = [-float('inf'), q1, q3, float('inf')]
                
            filtered_data['DXY_Change_Category'] = pd.cut(
                filtered_data['DXY_pct_change'],
                bins=bins,
                labels=['Low', 'Medium', 'High']
            )
            
            # Calculate average gold price change by DXY category
            category_analysis = filtered_data.groupby('DXY_Change_Category')['Price_pct_change'].agg(['mean', 'std', 'count'])
            category_analysis.columns = ['Average Gold Price Change (%)', 'Standard Deviation', 'Count']
            category_analysis = category_analysis.reset_index()
            
            st.table(category_analysis)
            
            # Check for expected inverse relationship in the categories
            if category_analysis.loc[0, 'Average Gold Price Change (%)'] > category_analysis.loc[2, 'Average Gold Price Change (%)']:
                st.success("The data shows that when DXY changes are low, gold price changes tend to be higher, and when DXY changes are high, gold price changes tend to be lower. This pattern supports the inverse relationship hypothesis.")
            else:
                st.warning("The data does not show the expected pattern where low DXY changes correspond to high gold price changes and vice versa.")
            
            # Create boxplot of gold price changes by DXY category
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='DXY_Change_Category', y='Price_pct_change', data=filtered_data, ax=ax)
            ax.set_xlabel('DXY Change Category')
            ax.set_ylabel('Gold Price Change (%)')
            ax.set_title('Gold Price Changes by DXY Change Category')
            
            st.pyplot(fig)
            
            # ANOVA test to see if the differences are statistically significant
            if len(filtered_data['DXY_Change_Category'].unique()) > 1:
                try:
                    groups = [filtered_data[filtered_data['DXY_Change_Category'] == cat]['Price_pct_change'] 
                             for cat in filtered_data['DXY_Change_Category'].unique() if not filtered_data[filtered_data['DXY_Change_Category'] == cat].empty]
                    
                    anova_result = stats.f_oneway(*groups)
                    
                    st.markdown(f"""
                    **ANOVA Test Results:**
                    
                    F-statistic: {anova_result.statistic:.4f}
                    p-value: {anova_result.pvalue:.4f}
                    
                    The difference in gold price changes between DXY change categories is 
                    {"statistically significant" if anova_result.pvalue < 0.05 else "not statistically significant"}.
                    """)
                except Exception as e:
                    st.error(f"Could not perform ANOVA test: {str(e)}")
                    
            # Additional test specific to inverse relationship
            st.subheader("Inverse Correlation by DXY Change Magnitude")
            
            # Create a scatter plot with color coding by DXY change magnitude
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Creating a colormap based on DXY magnitude
            scatter = ax.scatter(filtered_data['DXY_pct_change'], 
                               filtered_data['Price_pct_change'],
                               c=np.abs(filtered_data['DXY_pct_change']), 
                               cmap='viridis', 
                               alpha=0.6)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Magnitude of DXY Change (%)')
            
            # Add a trend line
            z = np.polyfit(filtered_data['DXY_pct_change'], filtered_data['Price_pct_change'], 1)
            p = np.poly1d(z)
            ax.plot(np.array([filtered_data['DXY_pct_change'].min(), filtered_data['DXY_pct_change'].max()]), 
                   p(np.array([filtered_data['DXY_pct_change'].min(), filtered_data['DXY_pct_change'].max()])), 
                   "r--", lw=1)
            
            # Add axis labels and title
            ax.set_xlabel('DXY Percentage Change')
            ax.set_ylabel('Gold Price Percentage Change')
            ax.set_title('Gold Price Change vs. DXY Change\n(Inverse Relationship Test)')
            ax.grid(True, alpha=0.3)
            
            # Add equation of the line
            equation_text = f"y = {z[0]:.4f}x + {z[1]:.4f}"
            ax.annotate(equation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            st.pyplot(fig)
            
            # Calculate day-to-day inverse movement percentage
            filtered_data['inverse_movement'] = (
                (filtered_data['DXY_pct_change'] > 0) & (filtered_data['Price_pct_change'] < 0) |
                (filtered_data['DXY_pct_change'] < 0) & (filtered_data['Price_pct_change'] > 0)
            )
            
            inverse_pct = filtered_data['inverse_movement'].mean() * 100
            
            st.metric(
                label="Days with Inverse Movement",
                value=f"{inverse_pct:.1f}%",
                delta=f"{inverse_pct - 50:.1f}%" if inverse_pct != 50 else None,
                delta_color="normal" if inverse_pct > 50 else "inverse"
            )
            
            if inverse_pct > 55:
                st.success(f"Gold prices move in the opposite direction to DXY {inverse_pct:.1f}% of the time, supporting the inverse relationship hypothesis.")
            elif inverse_pct < 45:
                st.warning(f"Gold prices move in the opposite direction to DXY only {inverse_pct:.1f}% of the time, which does not support the inverse relationship hypothesis.")
            else:
                st.info(f"Gold prices move in the opposite direction to DXY {inverse_pct:.1f}% of the time, which suggests a weak or inconsistent inverse relationship.")
        
        with tab3:
            st.subheader("Advanced Analysis")
            
            # Granger Causality Test
            st.markdown("### Granger Causality Test")
            st.markdown("""
            The Granger Causality Test examines whether one time series is useful in forecasting another. 
            It helps determine if changes in DXY "cause" changes in gold prices or vice versa.
            """)
            
            max_lag = st.slider("Select maximum lag for Granger Causality Test (days)", 1, 30, 10)
            
            # Prepare data for Granger causality test (dropna)
            granger_data = data.dropna().copy()
            
            # Create lagged dataset
            granger_series = pd.DataFrame({'DXY': granger_data['DXY'], 'Gold_Price': granger_data['Price']})
            
            # Run the tests
            with st.spinner("Running Granger Causality Tests."):
                try:
                    # Test if DXY Granger-causes Gold Price
                    dxy_causes_gold = grangercausalitytests(granger_series[['Gold_Price', 'DXY']], maxlag=max_lag, verbose=False)
                    
                    # Test if Gold Price Granger-causes DXY
                    gold_causes_dxy = grangercausalitytests(granger_series[['DXY', 'Gold_Price']], maxlag=max_lag, verbose=False)
                    
                    # Extract p-values
                    dxy_to_gold_pvals = [dxy_causes_gold[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                    gold_to_dxy_pvals = [gold_causes_dxy[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                    
                    # Display results
                    granger_results = pd.DataFrame({
                        'Lag': list(range(1, max_lag+1)),
                        'DXY causes Gold Price (p-value)': dxy_to_gold_pvals,
                        'Gold Price causes DXY (p-value)': gold_to_dxy_pvals,
                    })
                    
                    # Add significance columns
                    granger_results['DXY→Gold Significant'] = granger_results['DXY causes Gold Price (p-value)'] < 0.05
                    granger_results['Gold→DXY Significant'] = granger_results['Gold Price causes DXY (p-value)'] < 0.05
                    
                    st.dataframe(granger_results)
                    
                    # Interpreting Granger causality results
                    dxy_causes_gold_sig = any(p < 0.05 for p in dxy_to_gold_pvals)
                    gold_causes_dxy_sig = any(p < 0.05 for p in gold_to_dxy_pvals)
                    
                    causality_conclusion = ""
                    if dxy_causes_gold_sig and gold_causes_dxy_sig:
                        causality_conclusion = "Bidirectional causality: DXY and gold prices Granger-cause each other."
                    elif dxy_causes_gold_sig:
                        causality_conclusion = "Unidirectional causality: DXY Granger-causes gold prices."
                    elif gold_causes_dxy_sig:
                        causality_conclusion = "Unidirectional causality: Gold prices Granger-cause DXY."
                    else:
                        causality_conclusion = "No Granger causality detected between DXY and gold prices."
                    
                    st.markdown(f"**Conclusion:** {causality_conclusion}")
                    
                except Exception as e:
                    st.error(f"Error in Granger causality test: {e}")
            
            # Final Summary
            st.subheader("Hypothesis Testing Summary")
            
            # Summarize all findings
            st.markdown("""
            ### Summary of Findings
            
            Based on our comprehensive analysis, we can draw the following conclusions about the hypothesis that there is an 
            inverse relationship between DXY and gold prices:
            """)
            
            # Check if slope is negative (supports inverse relationship)
            slope_supports = slope < 0
            
            # Check if correlation is negative (supports inverse relationship)
            correlation_supports = correlation < 0
            
            # Check if inverse movement percentage is above 50% (supports inverse relationship)
            inverse_movement_supports = inverse_pct > 50 if 'inverse_pct' in locals() else None
            
            # Create summary table
            results_table = pd.DataFrame({
                'Analysis Method': ['Correlation Analysis', 'Linear Regression', 'Granger Causality', 'Daily Inverse Movement', 'DXY Change Categories'],
                'Finding': [
                    f"Correlation coefficient: {correlation:.3f}",
                    f"Slope: {slope:.2f}, R-squared: {r_squared:.3f}",
                    causality_conclusion if 'causality_conclusion' in locals() else "Not computed",
                    f"{inverse_pct:.1f}% of days show inverse movement" if 'inverse_pct' in locals() else "Not computed",
                    "See DXY Change Analysis tab for details"
                ],
                'Supports Inverse Hypothesis': [
                    "✓ Yes" if correlation_supports else "✗ No",
                    "✓ Yes" if slope_supports else "✗ No",
                    "✓ Yes" if 'dxy_causes_gold_sig' in locals() and dxy_causes_gold_sig else "✗ No",
                    "✓ Yes" if inverse_movement_supports else "✗ No" if inverse_movement_supports is not None else "Not computed",
                    "✓ Yes" if 'anova_result' in locals() and anova_result.pvalue < 0.05 else "✗ No"
                ]
            })
            
            st.table(results_table)
            
            # Final conclusion
            st.subheader("Conclusion")
            
            # Count how many tests support the hypothesis
            if 'results_table' in locals():
                support_count = results_table['Supports Inverse Hypothesis'].str.contains('Yes').sum()
                total_tests = len(results_table)
                
                # Provide overall conclusion
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