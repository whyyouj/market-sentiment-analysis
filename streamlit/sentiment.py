import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import get_bigquery_client, load_data, plot_distribution, plot_time_series
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression

def app():
    """Main function to run the sentiment score analysis page"""
    st.title("Sentiment Score Analysis")
    
    # Add explanation of sentiment scores based on the information provided
    with st.expander("What are Sentiment Scores?", expanded=True):
        st.markdown("""
        ### Sentiment Score Methodology
        
        The sentiment scores in this dataset represent market sentiment derived from news headlines:
        
        1. **Raw Sentiment Score** (-1 to +1):
           - Raw news headlines are processed to generate sentiment scores
           - Negative scores (-1) indicate negative sentiment
           - Positive scores (+1) indicate positive sentiment
           - Multiple daily headlines are aggregated using simple averaging
           - Days without news are filled with zeros
        
        2. **Exponential Weighted Score**:
           - Computed with a 30-day decay factor to capture lingering market sentiment effects
           - Naturally carries forward past sentiment with decaying influence
           - Provides a smoothed view of sentiment that accounts for the persistence of market sentiment
        
        ### Hypothesis: Sentiment and Gold Prices
        
        We hypothesize that sentiment indicators derived from financial news have a significant relationship with gold prices.
        Specifically, we expect that:
        
        1. **Positive sentiment** will correlate with **higher gold prices** or price increases
        2. **Negative sentiment** will correlate with **lower gold prices** or price decreases
        3. **Sentiment changes** may **precede price movements**, suggesting predictive value
        
        This hypothesis is based on the efficient market hypothesis and behavioral finance theories suggesting
        that market sentiment impacts investor decision-making, which in turn affects asset prices.
        """)
    
    # Get data from BigQuery
    client = get_bigquery_client()
    if client:
        # Query to get sentiment-specific data
        try:
            query = """
                SELECT Date, Price, Sentiment_Score, Exponential_Weighted_Score
                FROM `IS3107_Project.gold_market_data`
                WHERE Sentiment_Score IS NOT NULL AND Price IS NOT NULL
                ORDER BY Date
            """
            data = client.query(query).to_dataframe()
            
            # Convert Date to datetime if it's not already
            if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Create a clean dataset without any NaN values
            clean_data = data.dropna(subset=['Date', 'Price', 'Sentiment_Score', 'Exponential_Weighted_Score']).copy()
            
            
                
            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Analysis", "Raw Sentiment Analysis", "Exponential Weighted Analysis", "Hypothesis Testing"])
            
            with tab1:
                st.subheader("Summary Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Raw Sentiment Score")
                    st.dataframe(clean_data['Sentiment_Score'].describe())
                    
                    # Distribution plot
                    st.subheader("Distribution of Raw Sentiment Score")
                    plot_distribution(clean_data, 'Sentiment_Score')
                    
                with col2:
                    st.markdown("#### Exponential Weighted Score")
                    st.dataframe(clean_data['Exponential_Weighted_Score'].describe())
                    
                    # Distribution plot
                    st.subheader("Distribution of Exponential Weighted Score")
                    plot_distribution(clean_data, 'Exponential_Weighted_Score')
                
                # Additional insights
                st.subheader("Sentiment Analysis Insights")
                
                # Calculate days with neutral, positive, negative sentiment
                positive_days = (clean_data['Sentiment_Score'] > 0).sum()
                negative_days = (clean_data['Sentiment_Score'] < 0).sum()
                neutral_days = (clean_data['Sentiment_Score'] == 0).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Sentiment Days", f"{positive_days} ({positive_days/len(clean_data)*100:.1f}%)")
                with col2:
                    st.metric("Negative Sentiment Days", f"{negative_days} ({negative_days/len(clean_data)*100:.1f}%)")
                with col3:
                    st.metric("Neutral Sentiment Days", f"{neutral_days} ({neutral_days/len(clean_data)*100:.1f}%)")
            
            with tab2:
                st.header("Testing Raw Sentiment Score Relationship with Gold")
                
                # Calculate Pearson correlation coefficient
                correlation = clean_data['Sentiment_Score'].corr(clean_data['Price'])
                correlation_text = f"Correlation: {correlation:.3f}"
                relationship_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                
                # Create a dual y-axis time series plot
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot gold price
                color = 'goldenrod'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Gold Price (USD)', color=color)
                ax1.plot(clean_data['Date'], clean_data['Price'], color=color, linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color)
                
                # Create second y-axis for Sentiment Score
                ax2 = ax1.twinx()
                color = 'blue'
                ax2.set_ylabel('Raw Sentiment Score', color=color)
                ax2.plot(clean_data['Date'], clean_data['Sentiment_Score'], color=color, linewidth=2, alpha = 0.5)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add a title and annotation
                plt.title('Gold Price and Raw Sentiment Score Over Time', fontsize=14)
                plt.figtext(0.15, 0.85, correlation_text, 
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig1.tight_layout()
                st.pyplot(fig1)
                
                # Linear regression scatter plot to explicitly test the hypothesis
                fig2, ax = plt.subplots(figsize=(10, 6))
                
                # Create scatter plot
                ax.scatter(clean_data['Sentiment_Score'], clean_data['Price'], alpha=0.6, c='blue')
                
                # Add regression line
                X = clean_data['Sentiment_Score'].values.reshape(-1, 1)
                y = clean_data['Price'].values
                
                # Fit the model
                model = LinearRegression()
                model.fit(X, y)
                
                # Get regression metrics
                r_squared = model.score(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                
                # Generate predictions for the line
                x_range = np.linspace(clean_data['Sentiment_Score'].min(), clean_data['Sentiment_Score'].max(), 100)
                y_pred = model.predict(x_range.reshape(-1, 1))
                
                # Plot the regression line
                ax.plot(x_range, y_pred, color='red', linewidth=2)
                
                # Add equation and R² to the plot
                equation = f"Price = {intercept:.2f} + {slope:.2f} × Sentiment"
                r2_text = f"R² = {r_squared:.3f}"
                ax.annotate(equation + "\n" + r2_text,
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                            va='top')
                
                # Set labels and title
                ax.set_xlabel('Raw Sentiment Score')
                ax.set_ylabel('Gold Price (USD)')
                ax.set_title('Gold Price vs. Raw Sentiment Score: Testing Relationship', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                # Emphasize the concept of relationship
                if slope > 0:
                    st.success(f"The regression slope is positive ({slope:.2f}), supporting the hypothesis that positive sentiment correlates with higher gold prices.")
                else:
                    st.warning(f"The regression slope is negative ({slope:.2f}), which does not support the hypothesis that positive sentiment correlates with higher gold prices.")
                
                # More detailed regression with statsmodels for p-values
                X_sm = sm.add_constant(X)
                model_sm = sm.OLS(y, X_sm).fit()
                
                # Extract p-value for Sentiment Score coefficient
                p_value = model_sm.pvalues[1]
                
                # Display regression summary in expandable section
                with st.expander("View Detailed Regression Statistics"):
                    st.text(model_sm.summary().as_text())
                
                # Add rolling correlation analysis
                st.subheader("Rolling Correlation Analysis")
                
                # Add rolling window selector
                window_size = st.slider("Select rolling window size (days) for Raw Sentiment", 30, 365, 90)
                
                # Calculate rolling correlation
                clean_data['rolling_corr_raw'] = clean_data['Sentiment_Score'].rolling(window=window_size).corr(clean_data['Price'])
                
                # Create the rolling correlation plot with properly aligned data
                fig4, ax = plt.subplots(figsize=(12, 6))
                
                # Filter out NaN values for plotting
                valid_data = clean_data.dropna(subset=['rolling_corr_raw'])
                
                # Plot the correlation line
                ax.plot(valid_data['Date'], valid_data['rolling_corr_raw'], 
                        color='purple', linewidth=2)
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Fill areas based on correlation sign
                ax.fill_between(valid_data['Date'], 
                                valid_data['rolling_corr_raw'], 
                                0, 
                                where=(valid_data['rolling_corr_raw'] > 0),
                                color='green', alpha=0.3, label='Positive Correlation\n(Higher sentiment → Higher prices)')
                
                ax.fill_between(valid_data['Date'], 
                                valid_data['rolling_corr_raw'], 
                                0, 
                                where=(valid_data['rolling_corr_raw'] <= 0),
                                color='red', alpha=0.3, label='Negative Correlation\n(Higher sentiment → Lower prices)')
                
                # Add visualization enhancements
                ax.set_xlabel('Date')
                ax.set_ylabel('Correlation Coefficient')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add title with window size information
                plt.title(f'{window_size}-day Rolling Correlation between Raw Sentiment and Gold Price')
                
                # Display correlation statistics
                mean_corr = valid_data['rolling_corr_raw'].mean()
                pos_pct = (valid_data['rolling_corr_raw'] > 0).mean() * 100
                corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"
                
                # Add annotation for correlation stats
                plt.figtext(0.5, 0.01, corr_stats, ha='center', 
                            bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
                st.pyplot(fig4)
                
                # Additional interpretation based on rolling correlation
                st.markdown(f"""
                **Rolling Correlation Insights:**
                - The mean correlation between raw sentiment and gold prices over this period is **{mean_corr:.3f}**.
                - Sentiment shows a positive correlation with gold prices **{pos_pct:.1f}%** of the time.
                - {'This suggests that market sentiment generally drives gold prices in the same direction.' if pos_pct > 60 else 
                'This suggests that market sentiment generally drives gold prices in the opposite direction.' if pos_pct < 40 else
                'This suggests that the relationship between sentiment and gold prices is inconsistent or context-dependent.'}
                """)

            with tab3:
                st.header("Testing Exponential Weighted Score Relationship with Gold")
                
                # Calculate Pearson correlation coefficient
                exp_correlation = clean_data['Exponential_Weighted_Score'].corr(clean_data['Price'])
                exp_correlation_text = f"Correlation: {exp_correlation:.3f}"
                exp_relationship_strength = "strong" if abs(exp_correlation) > 0.7 else "moderate" if abs(exp_correlation) > 0.3 else "weak"
                
                # Create a dual y-axis time series plot
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot gold price
                color = 'goldenrod'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Gold Price (USD)', color=color)
                ax1.plot(clean_data['Date'], clean_data['Price'], color=color, linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color)
                
                # Create second y-axis for Exponential Weighted Score
                ax2 = ax1.twinx()
                color = 'red'
                ax2.set_ylabel('Exponential Weighted Score', color=color)
                ax2.plot(clean_data['Date'], clean_data['Exponential_Weighted_Score'], color=color, linewidth=2)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add a title and annotation
                plt.title('Gold Price and Exponential Weighted Score Over Time', fontsize=14)
                plt.figtext(0.15, 0.85, exp_correlation_text, 
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig1.tight_layout()
                st.pyplot(fig1)
                
                # Linear regression scatter plot to explicitly test the hypothesis
                fig2, ax = plt.subplots(figsize=(10, 6))
                
                # Create scatter plot
                ax.scatter(clean_data['Exponential_Weighted_Score'], clean_data['Price'], alpha=0.6, c='red')
                
                # Add regression line
                X_exp = clean_data['Exponential_Weighted_Score'].values.reshape(-1, 1)
                y = clean_data['Price'].values
                
                # Fit the model
                model_exp = LinearRegression()
                model_exp.fit(X_exp, y)
                
                # Get regression metrics
                r_squared_exp = model_exp.score(X_exp, y)
                slope_exp = model_exp.coef_[0]
                intercept_exp = model_exp.intercept_
                
                # Generate predictions for the line
                x_range_exp = np.linspace(clean_data['Exponential_Weighted_Score'].min(), 
                                            clean_data['Exponential_Weighted_Score'].max(), 100)
                y_pred_exp = model_exp.predict(x_range_exp.reshape(-1, 1))
                
                # Plot the regression line
                ax.plot(x_range_exp, y_pred_exp, color='red', linewidth=2)
                
                # Add equation and R² to the plot
                equation_exp = f"Price = {intercept_exp:.2f} + {slope_exp:.2f} × ExpWeightedScore"
                r2_text_exp = f"R² = {r_squared_exp:.3f}"
                ax.annotate(equation_exp + "\n" + r2_text_exp,
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                            va='top')
                
                # Set labels and title
                ax.set_xlabel('Exponential Weighted Score')
                ax.set_ylabel('Gold Price (USD)')
                ax.set_title('Gold Price vs. Exponential Weighted Score: Testing Relationship', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                # Emphasize the concept of relationship
                if slope_exp > 0:
                    st.success(f"The regression slope is positive ({slope_exp:.2f}), supporting the hypothesis that positive exponential weighted sentiment correlates with higher gold prices.")
                else:
                    st.warning(f"The regression slope is negative ({slope_exp:.2f}), which does not support the hypothesis that positive exponential weighted sentiment correlates with higher gold prices.")
                
                # More detailed regression with statsmodels for p-values
                X_exp_sm = sm.add_constant(X_exp)
                model_exp_sm = sm.OLS(y, X_exp_sm).fit()
                
                # Extract p-value for Exponential Score coefficient
                p_value_exp = model_exp_sm.pvalues[1]
                
                # Display regression summary in expandable section
                with st.expander("View Detailed Regression Statistics"):
                    st.text(model_exp_sm.summary().as_text())
                
                # Add rolling correlation analysis
                st.subheader("Rolling Correlation Analysis")
                
                # Add rolling window selector
                window_size_exp = st.slider("Select rolling window size (days) for Exponential Weighted Score", 30, 365, 90)
                
                # Calculate rolling correlation
                clean_data['rolling_corr_exp'] = clean_data['Exponential_Weighted_Score'].rolling(window=window_size_exp).corr(clean_data['Price'])
                
                # Create the rolling correlation plot with properly aligned data
                fig5, ax = plt.subplots(figsize=(12, 6))
                
                # Filter out NaN values for plotting
                valid_data_exp = clean_data.dropna(subset=['rolling_corr_exp'])
                
                # Plot the correlation line
                ax.plot(valid_data_exp['Date'], valid_data_exp['rolling_corr_exp'], 
                        color='purple', linewidth=2)
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Fill areas based on correlation sign
                ax.fill_between(valid_data_exp['Date'], 
                                valid_data_exp['rolling_corr_exp'], 
                                0, 
                                where=(valid_data_exp['rolling_corr_exp'] > 0),
                                color='green', alpha=0.3, label='Positive Correlation\n(Higher sentiment → Higher prices)')
                
                ax.fill_between(valid_data_exp['Date'], 
                                valid_data_exp['rolling_corr_exp'], 
                                0, 
                                where=(valid_data_exp['rolling_corr_exp'] <= 0),
                                color='red', alpha=0.3, label='Negative Correlation\n(Higher sentiment → Lower prices)')
                
                # Add visualization enhancements
                ax.set_xlabel('Date')
                ax.set_ylabel('Correlation Coefficient')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add title with window size information
                plt.title(f'{window_size_exp}-day Rolling Correlation between Exponential Weighted Score and Gold Price')
                
                # Display correlation statistics
                mean_corr_exp = valid_data_exp['rolling_corr_exp'].mean()
                pos_pct_exp = (valid_data_exp['rolling_corr_exp'] > 0).mean() * 100
                corr_stats_exp = f"Mean correlation: {mean_corr_exp:.3f} | Positive correlation: {pos_pct_exp:.1f}% of time"
                
                # Add annotation for correlation stats
                plt.figtext(0.5, 0.01, corr_stats_exp, ha='center', 
                            bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig5.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
                st.pyplot(fig5)
                
                # Additional interpretation based on rolling correlation
                st.markdown(f"""
                **Rolling Correlation Insights:**
                - The mean correlation between exponential weighted sentiment and gold prices over this period is **{mean_corr_exp:.3f}**.
                - Exponential weighted sentiment shows a positive correlation with gold prices **{pos_pct_exp:.1f}%** of the time.
                - {'This suggests that market sentiment generally drives gold prices in the same direction.' if pos_pct_exp > 60 else 
                'This suggests that market sentiment generally drives gold prices in the opposite direction.' if pos_pct_exp < 40 else
                'This suggests that the relationship between sentiment and gold prices is inconsistent or context-dependent.'}
                """)
            
            with tab4:
                # st.subheader("Sentiment Score Hypothesis Testing")
                
                # # Granger causality test
                # st.markdown("#### Granger Causality Test: Does Sentiment Help Predict Gold Prices?")
                
                # # Prepare data for Granger causality test
                # granger_data = clean_data[['Date', 'Price', 'Sentiment_Score', 'Exponential_Weighted_Score']].set_index('Date')
                # granger_data = granger_data.sort_index()
                
                # # Calculate daily returns for stationarity
                # granger_data['Price_Change'] = granger_data['Price'].pct_change() * 100
                # granger_data = granger_data.dropna()
                
                # # Test different maximum lags
                # max_lag = min(8, len(granger_data) // 10)  # Don't use too many lags with limited data
                
                # try:
                #     # Run Granger test (Raw Sentiment -> Price)
                #     gc_results_raw = grangercausalitytests(
                #         granger_data[['Sentiment_Score', 'Price_Change']], 
                #         max_lag, 
                #         verbose=False
                #     )
                    
                #     # Extract p-values for each lag
                #     p_values_raw = [gc_results_raw[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                    
                #     # Create dataframe for display
                #     granger_results_raw = pd.DataFrame({
                #         'Lag': range(1, max_lag+1),
                #         'P-Value': p_values_raw,
                #         'Significant at 5%': [p < 0.05 for p in p_values_raw]
                #     })
                    
                #     st.markdown("##### Raw Sentiment Score → Gold Price")
                #     st.dataframe(granger_results_raw)
                    
                #     # Check if any lag is significant
                #     any_significant_raw = any(granger_results_raw['Significant at 5%'])
                    
                #     if any_significant_raw:
                #         st.success("**Granger causality test indicates that raw sentiment helps predict future gold price changes**")
                #         significant_lags_raw = granger_results_raw[granger_results_raw['Significant at 5%']]['Lag'].tolist()
                #         st.markdown(f"Significant lags: {', '.join(map(str, significant_lags_raw))} days")
                #     else:
                #         st.warning("**Granger causality test does not show significant predictive power of raw sentiment on gold prices**")
                    
                #     # Run Granger test (Exponential Weighted Score -> Price)
                #     gc_results_exp = grangercausalitytests(
                #         granger_data[['Exponential_Weighted_Score', 'Price_Change']], 
                #         max_lag, 
                #         verbose=False
                #     )
                    
                #     # Extract p-values for each lag
                #     p_values_exp = [gc_results_exp[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                    
                #     # Create dataframe for display
                #     granger_results_exp = pd.DataFrame({
                #         'Lag': range(1, max_lag+1),
                #         'P-Value': p_values_exp,
                #         'Significant at 5%': [p < 0.05 for p in p_values_exp]
                #     })
                    
                #     st.markdown("##### Exponential Weighted Score → Gold Price")
                #     st.dataframe(granger_results_exp)
                    
                #     # Check if any lag is significant
                #     any_significant_exp = any(granger_results_exp['Significant at 5%'])
                    
                #     if any_significant_exp:
                #         st.success("**Granger causality test indicates that exponential weighted sentiment helps predict future gold price changes**")
                #         significant_lags_exp = granger_results_exp[granger_results_exp['Significant at 5%']]['Lag'].tolist()
                #         st.markdown(f"Significant lags: {', '.join(map(str, significant_lags_exp))} days")
                #     else:
                #         st.warning("**Granger causality test does not show significant predictive power of exponential weighted sentiment on gold prices**")
                
                # except Exception as e:
                #     st.error(f"Granger causality test failed: {str(e)}")
                #     st.info("This may be due to insufficient data or stationarity issues in the time series.")
                
                # Summary of findings
                st.subheader("Summary of Hypothesis Testing")
                
                # Check all test results for Raw Sentiment
                # granger_supports_raw = any_significant_raw if 'any_significant_raw' in locals() else False
                corr_supports_raw = abs(correlation) > 0.1 and correlation > 0 if 'correlation' in locals() else False
                regression_supports_raw = slope > 0 and p_value < 0.05 if 'slope' in locals() and 'p_value' in locals() else False
                rolling_supports_raw = mean_corr > 0 and pos_pct > 50 if 'mean_corr' in locals() and 'pos_pct' in locals() else False
                
                # Check all test results for Exponential Weighted Score
                # granger_supports_exp = any_significant_exp if 'any_significant_exp' in locals() else False
                corr_supports_exp = abs(exp_correlation) > 0.1 and exp_correlation > 0 if 'exp_correlation' in locals() else False
                regression_supports_exp = slope_exp > 0 and p_value_exp < 0.05 if 'slope_exp' in locals() and 'p_value_exp' in locals() else False
                rolling_supports_exp = mean_corr_exp > 0 and pos_pct_exp > 50 if 'mean_corr_exp' in locals() and 'pos_pct_exp' in locals() else False
                
                # Create summary table for Raw Sentiment
                summary_df_raw = pd.DataFrame({
                    'Test': [
                        'Correlation Analysis', 
                        'Linear Regression', 
                        'Rolling Correlation'
                        # 'Granger Causality'
                    ],
                    'Finding (Raw Sentiment)': [
                        f"Correlation = {correlation:.3f}" if 'correlation' in locals() else "Not computed",
                        f"Slope = {slope:.3f}, p-value = {p_value:.4f}" if 'slope' in locals() and 'p_value' in locals() else "Not computed",
                        f"Mean = {mean_corr:.3f}, Positive: {pos_pct:.1f}%" if 'mean_corr' in locals() and 'pos_pct' in locals() else "Not computed"
                        # "Significant at some lags" if granger_supports_raw else "Not significant"
                    ],
                    'Supports Hypothesis': [
                        "✓ Yes" if corr_supports_raw else "✗ No",
                        "✓ Yes" if regression_supports_raw else "✗ No",
                        "✓ Yes" if rolling_supports_raw else "✗ No"
                        # "✓ Yes" if granger_supports_raw else "✗ No"
                    ]
                })
                
                    # Create summary table for Exponential Weighted Score
                summary_df_exp = pd.DataFrame({
                    'Test': [
                        'Correlation Analysis', 
                        'Linear Regression', 
                        'Rolling Correlation'
                        # 'Granger Causality'
                    ],
                    'Finding (Exponential Weighted)': [
                        f"Correlation = {exp_correlation:.3f}" if 'exp_correlation' in locals() else "Not computed",
                        f"Slope = {slope_exp:.3f}, p-value = {p_value_exp:.4f}" if 'slope_exp' in locals() and 'p_value_exp' in locals() else "Not computed",
                        f"Mean = {mean_corr_exp:.3f}, Positive: {pos_pct_exp:.1f}%" if 'mean_corr_exp' in locals() and 'pos_pct_exp' in locals() else "Not computed"
                        # "Significant at some lags" if granger_supports_exp else "Not significant"
                    ],
                    'Supports Hypothesis': [
                        "✓ Yes" if corr_supports_exp else "✗ No",
                        "✓ Yes" if regression_supports_exp else "✗ No",
                        "✓ Yes" if rolling_supports_exp else "✗ No"
                        # "✓ Yes" if granger_supports_exp else "✗ No"
                    ]
                })
                
                                    # Display the summary tables
                st.markdown("##### Raw Sentiment Score Summary")
                st.dataframe(summary_df_raw)
                
                st.markdown("##### Exponential Weighted Score Summary")
                st.dataframe(summary_df_exp)
                
                # Calculate overall support counts
                raw_supports_count = sum([
                    corr_supports_raw, 
                    regression_supports_raw, 
                    rolling_supports_raw
                    # granger_supports_raw
                ])
                
                exp_supports_count = sum([
                    corr_supports_exp, 
                    regression_supports_exp, 
                    rolling_supports_exp 
                    # granger_supports_exp
                ])
                
                # Overall conclusion section
                st.subheader("Overall Conclusion")
                
                # Compare methods
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Raw Sentiment Tests Supporting Hypothesis", f"{raw_supports_count}/3")
                with col2:
                    st.metric("Exponential Weighted Tests Supporting Hypothesis", f"{exp_supports_count}/3")
                
                # Final conclusion based on test counts
                if raw_supports_count > exp_supports_count:
                    st.success(f"**The Raw Sentiment Score shows stronger support ({raw_supports_count}/3 tests) for our hypothesis than the Exponential Weighted Score ({exp_supports_count}/3 tests).**")
                    better_method = "Raw Sentiment"
                elif exp_supports_count > raw_supports_count:
                    st.success(f"**The Exponential Weighted Score shows stronger support ({exp_supports_count}/3 tests) for our hypothesis than the Raw Sentiment Score ({raw_supports_count}/3 tests).**")
                    better_method = "Exponential Weighted"
                else:
                    st.info(f"**Both sentiment methods show equal support ({raw_supports_count}/3 tests) for our hypothesis.**")
                    better_method = "Both methods equally"
                
                # Visual comparison
                st.subheader("Visual Comparison")
                
                # Create a bar chart comparing support counts
                fig_comparison = plt.figure(figsize=(10, 6))
                
                tests = ['Correlation', 'Regression', 'Rolling Corr']
                raw_results = [corr_supports_raw, regression_supports_raw, rolling_supports_raw]
                exp_results = [corr_supports_exp, regression_supports_exp, rolling_supports_exp]
                
                x = np.arange(len(tests))
                width = 0.35
                
                plt.bar(x - width/2, [int(res) for res in raw_results], width, label='Raw Sentiment', color='blue', alpha=0.4)
                plt.bar(x + width/2, [int(res) for res in exp_results], width, label='Exponential Weighted', color='red', alpha=0.7)
                
                plt.ylabel('Supports Hypothesis')
                plt.title('Comparison of Hypothesis Test Results')
                plt.xticks(x, tests)
                plt.yticks([0, 1], ['No', 'Yes'])
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig_comparison)
                
                # Recommendations section
                st.subheader("Recommendations")
                st.markdown(f"""
                Based on our analysis, we recommend:
                
                1. **Primary Sentiment Metric:** Use the {better_method} Score as the primary sentiment indicator for gold price analysis.
                
                2. **Trading Strategy Development:** Focus on developing strategies that incorporate sentiment metrics with{
                    ' a ' + str(window_size) + '-day window' if 'window_size' in locals() else ''
                } for optimal signal generation.
                
                3. **Combined Approach:** Consider using both raw and exponential weighted sentiment in a combined model to capture both immediate sentiment shifts and longer-term sentiment trends.
                
                4. **Further Research:** Explore additional transformations of the sentiment data, such as volatility-adjusted sentiment or sentiment momentum indicators.
                
                
                """)
                
                # Add export options
                st.subheader("Export Analysis Results")
                
                # Create a dictionary with key results
                analysis_results = {
                    "Raw Sentiment": {
                        "Correlation": correlation if 'correlation' in locals() else None,
                        "Regression_Slope": slope if 'slope' in locals() else None,
                        "Regression_P_Value": p_value if 'p_value' in locals() else None,
                        "Mean_Rolling_Correlation": mean_corr if 'mean_corr' in locals() else None,
                        "Percent_Positive_Correlation": pos_pct if 'pos_pct' in locals() else None,
                        # "Significant_Granger_Lags": significant_lags_raw if 'significant_lags_raw' in locals() and any_significant_raw else [],
                        "Tests_Supporting_Hypothesis": raw_supports_count
                    },
                    "Exponential_Weighted": {
                        "Correlation": exp_correlation if 'exp_correlation' in locals() else None,
                        "Regression_Slope": slope_exp if 'slope_exp' in locals() else None,
                        "Regression_P_Value": p_value_exp if 'p_value_exp' in locals() else None,
                        "Mean_Rolling_Correlation": mean_corr_exp if 'mean_corr_exp' in locals() else None,
                        "Percent_Positive_Correlation": pos_pct_exp if 'pos_pct_exp' in locals() else None,
                        # "Significant_Granger_Lags": significant_lags_exp if 'significant_lags_exp' in locals() and any_significant_exp else [],
                        "Tests_Supporting_Hypothesis": exp_supports_count
                    }
                }
                
                # Convert to DataFrame for CSV export
                export_df = pd.DataFrame({
                    "Metric": ["Correlation", "Regression Slope", "Regression P-Value", 
                                "Mean Rolling Correlation", "% Positive Correlation", 
                                "Tests Supporting Hypothesis"],
                    "Raw_Sentiment": [
                        f"{correlation:.4f}" if 'correlation' in locals() else "N/A",
                        f"{slope:.4f}" if 'slope' in locals() else "N/A",
                        f"{p_value:.4f}" if 'p_value' in locals() else "N/A",
                        f"{mean_corr:.4f}" if 'mean_corr' in locals() else "N/A",
                        f"{pos_pct:.1f}%" if 'pos_pct' in locals() else "N/A",
                        f"{raw_supports_count}/3"
                    ],
                    "Exponential_Weighted": [
                        f"{exp_correlation:.4f}" if 'exp_correlation' in locals() else "N/A",
                        f"{slope_exp:.4f}" if 'slope_exp' in locals() else "N/A",
                        f"{p_value_exp:.4f}" if 'p_value_exp' in locals() else "N/A",
                        f"{mean_corr_exp:.4f}" if 'mean_corr_exp' in locals() else "N/A",
                        f"{pos_pct_exp:.1f}%" if 'pos_pct_exp' in locals() else "N/A",
                        f"{exp_supports_count}/3"
                    ]
                })
                
                # Create a CSV for download
                csv = export_df.to_csv(index=False).encode('utf-8')
                
                # Create the download button
                st.download_button(
                    label="Download Analysis as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure your data is properly formatted and try again.")
    else:
        st.error("Could not connect to BigQuery. Please check your credentials.")

