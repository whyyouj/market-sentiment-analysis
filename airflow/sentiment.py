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
    
    # Sentiment information
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
    
    # Getting data from BQ
    client = get_bigquery_client()
    if client:
        # Getting sentiment score from BQ
        try:
            query = """
                SELECT Date, Price, Sentiment_Score, Exponential_Weighted_Score
                FROM `IS3107_Project.gold_market_data`
                WHERE Sentiment_Score IS NOT NULL AND Price IS NOT NULL
                ORDER BY Date
            """
            data = client.query(query).to_dataframe()
            
            # Datetime check
            if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Dataset that has no Nan values
            clean_data = data.dropna(subset=['Date', 'Price', 'Sentiment_Score', 'Exponential_Weighted_Score']).copy()
            
            
                
            # Tab creation
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Analysis", "Raw Sentiment Analysis", "Exponential Weighted Analysis", "Hypothesis Testing"])
            
            with tab1:
                st.subheader("Summary Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Raw Sentiment Score")
                    st.dataframe(clean_data['Sentiment_Score'].describe())
                    st.subheader("Distribution of Raw Sentiment Score")
                    plot_distribution(clean_data, 'Sentiment_Score')
                    
                with col2:
                    st.markdown("#### Exponential Weighted Score")
                    st.dataframe(clean_data['Exponential_Weighted_Score'].describe())
                    st.subheader("Distribution of Exponential Weighted Score")
                    plot_distribution(clean_data, 'Exponential_Weighted_Score')
                

                st.subheader("Sentiment Analysis Insights")
                
                # Number of days with neutral, positive, negative sentiment
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
                
                # Creating dual axis plot
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot gold price
                color = 'goldenrod'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Gold Price (USD)', color=color)
                ax1.plot(clean_data['Date'], clean_data['Price'], color=color, linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color)
                
                # Second y-axis for Sentiment Score
                ax2 = ax1.twinx()
                color = 'blue'
                ax2.set_ylabel('Raw Sentiment Score', color=color)
                ax2.plot(clean_data['Date'], clean_data['Sentiment_Score'], color=color, linewidth=2, alpha = 0.5)
                ax2.tick_params(axis='y', labelcolor=color)
                

                plt.title('Gold Price and Raw Sentiment Score Over Time', fontsize=14)
                plt.figtext(0.15, 0.85, correlation_text, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig1.tight_layout()
                st.pyplot(fig1)
                
                # Linear regression scatter plot
                fig2, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(clean_data['Sentiment_Score'], clean_data['Price'], alpha=0.6, c='blue')
                X = clean_data['Sentiment_Score'].values.reshape(-1, 1)
                y = clean_data['Price'].values
                

                model = LinearRegression()
                model.fit(X, y)
                
                # Getting regression metrics
                r_squared = model.score(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                
                # Generate line predictions
                x_range = np.linspace(clean_data['Sentiment_Score'].min(), clean_data['Sentiment_Score'].max(), 100)
                y_pred = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_pred, color='red', linewidth=2)
                
                # Adding more information to the plot
                equation = f"Price = {intercept:.2f} + {slope:.2f} × Sentiment"
                r2_text = f"R² = {r_squared:.3f}"
                ax.annotate(equation + "\n" + r2_text, xy=(0.05, 0.95), xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),va='top')
                

                ax.set_xlabel('Raw Sentiment Score')
                ax.set_ylabel('Gold Price (USD)')
                ax.set_title('Gold Price vs. Raw Sentiment Score: Testing Relationship', fontsize=14)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Relationship text dependent on results
                if slope > 0:
                    st.success(f"The regression slope is positive ({slope:.2f}), supporting the hypothesis that positive sentiment correlates with higher gold prices.")
                else:
                    st.warning(f"The regression slope is negative ({slope:.2f}), which does not support the hypothesis that positive sentiment correlates with higher gold prices.")

                
                X_sm = sm.add_constant(X)
                model_sm = sm.OLS(y, X_sm).fit()
                p_value = model_sm.pvalues[1]
                
                # Display regression summary 
                with st.expander("View Detailed Regression Statistics"):
                    st.text(model_sm.summary().as_text())
                

                st.subheader("Rolling Correlation Analysis")
                window_size = st.slider("Select rolling window size (days) for Raw Sentiment", 30, 365, 90)
                clean_data['rolling_corr_raw'] = clean_data['Sentiment_Score'].rolling(window=window_size).corr(clean_data['Price'])
                fig4, ax = plt.subplots(figsize=(12, 6))
                
                # Filtering Nan values
                valid_data = clean_data.dropna(subset=['rolling_corr_raw'])
                ax.plot(valid_data['Date'], valid_data['rolling_corr_raw'], color='purple', linewidth=2)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                
                ax.fill_between(valid_data['Date'], valid_data['rolling_corr_raw'], 0, where=(valid_data['rolling_corr_raw'] > 0),color='green', alpha=0.3, label='Positive Correlation\n(Higher sentiment → Higher prices)')
                ax.fill_between(valid_data['Date'], valid_data['rolling_corr_raw'], 0, where=(valid_data['rolling_corr_raw'] <= 0),color='red', alpha=0.3, label='Negative Correlation\n(Higher sentiment → Lower prices)')
                
                # Adding more visualisation enhancements
                ax.set_xlabel('Date')
                ax.set_ylabel('Correlation Coefficient')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add title
                plt.title(f'{window_size}-day Rolling Correlation between Raw Sentiment and Gold Price')
                
                # Display correlation statistics
                mean_corr = valid_data['rolling_corr_raw'].mean()
                pos_pct = (valid_data['rolling_corr_raw'] > 0).mean() * 100
                corr_stats = f"Mean correlation: {mean_corr:.3f} | Positive correlation: {pos_pct:.1f}% of time"
                
                # Add annotatio
                plt.figtext(0.5, 0.01, corr_stats, ha='center', 
                            bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig4.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the annotation
                st.pyplot(fig4)
                
                # Additional interpretation based on results
                st.markdown(f"""
                **Rolling Correlation Insights:**
                - The mean correlation between raw sentiment and gold prices over this period is **{mean_corr:.3f}**.
                - Sentiment shows a positive correlation with gold prices **{pos_pct:.1f}%** of the time.
                - {'This suggests that market sentiment generally drives gold prices in the same direction.' if pos_pct > 60 else 
                'This suggests that market sentiment generally drives gold prices in the opposite direction.' if pos_pct < 40 else
                'This suggests that the relationship between sentiment and gold prices is inconsistent or context-dependent.'}
                """)

            with tab3:
                
                # Same steps as above, just for exponential weighted score now
                st.header("Testing Exponential Weighted Score Relationship with Gold")
                
                # Calculate Pearson correlation coefficient
                exp_correlation = clean_data['Exponential_Weighted_Score'].corr(clean_data['Price'])
                exp_correlation_text = f"Correlation: {exp_correlation:.3f}"
                exp_relationship_strength = "strong" if abs(exp_correlation) > 0.7 else "moderate" if abs(exp_correlation) > 0.3 else "weak"
                

                fig1, ax1 = plt.subplots(figsize=(12, 6))

                color = 'goldenrod'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Gold Price (USD)', color=color)
                ax1.plot(clean_data['Date'], clean_data['Price'], color=color, linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'red'
                ax2.set_ylabel('Exponential Weighted Score', color=color)
                ax2.plot(clean_data['Date'], clean_data['Exponential_Weighted_Score'], color=color, linewidth=2)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add a title and annotation
                plt.title('Gold Price and Exponential Weighted Score Over Time', fontsize=14)
                plt.figtext(0.15, 0.85, exp_correlation_text, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig1.tight_layout()
                st.pyplot(fig1)
                
                fig2, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(clean_data['Exponential_Weighted_Score'], clean_data['Price'], alpha=0.6, c='red')
                X_exp = clean_data['Exponential_Weighted_Score'].values.reshape(-1, 1)
                y = clean_data['Price'].values

                model_exp = LinearRegression()
                model_exp.fit(X_exp, y)

                r_squared_exp = model_exp.score(X_exp, y)
                slope_exp = model_exp.coef_[0]
                intercept_exp = model_exp.intercept_
                x_range_exp = np.linspace(clean_data['Exponential_Weighted_Score'].min(), clean_data['Exponential_Weighted_Score'].max(), 100)
                y_pred_exp = model_exp.predict(x_range_exp.reshape(-1, 1))
                

                ax.plot(x_range_exp, y_pred_exp, color='red', linewidth=2)
                
                # Add more information to the plot
                equation_exp = f"Price = {intercept_exp:.2f} + {slope_exp:.2f} × ExpWeightedScore"
                r2_text_exp = f"R² = {r_squared_exp:.3f}"
                ax.annotate(equation_exp + "\n" + r2_text_exp,
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                            va='top')

                ax.set_xlabel('Exponential Weighted Score')
                ax.set_ylabel('Gold Price (USD)')
                ax.set_title('Gold Price vs. Exponential Weighted Score: Testing Relationship', fontsize=14)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Relationship text changes based on results
                if slope_exp > 0:
                    st.success(f"The regression slope is positive ({slope_exp:.2f}), supporting the hypothesis that positive exponential weighted sentiment correlates with higher gold prices.")
                else:
                    st.warning(f"The regression slope is negative ({slope_exp:.2f}), which does not support the hypothesis that positive exponential weighted sentiment correlates with higher gold prices.")

                X_exp_sm = sm.add_constant(X_exp)
                model_exp_sm = sm.OLS(y, X_exp_sm).fit()
                

                p_value_exp = model_exp_sm.pvalues[1]
                with st.expander("View Detailed Regression Statistics"):
                    st.text(model_exp_sm.summary().as_text())
                st.subheader("Rolling Correlation Analysis")
                window_size_exp = st.slider("Select rolling window size (days) for Exponential Weighted Score", 30, 365, 90)
                clean_data['rolling_corr_exp'] = clean_data['Exponential_Weighted_Score'].rolling(window=window_size_exp).corr(clean_data['Price'])
                
                fig5, ax = plt.subplots(figsize=(12, 6))
                
                # Filtering out nan values
                valid_data_exp = clean_data.dropna(subset=['rolling_corr_exp'])
                ax.plot(valid_data_exp['Date'], valid_data_exp['rolling_corr_exp'], 
                        color='purple', linewidth=2)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                ax.fill_between(valid_data_exp['Date'], valid_data_exp['rolling_corr_exp'], 0, where=(valid_data_exp['rolling_corr_exp'] > 0),color='green', alpha=0.3, label='Positive Correlation\n(Higher sentiment → Higher prices)')
                
                ax.fill_between(valid_data_exp['Date'], valid_data_exp['rolling_corr_exp'], 0, where=(valid_data_exp['rolling_corr_exp'] <= 0),color='red', alpha=0.3, label='Negative Correlation\n(Higher sentiment → Lower prices)')
                
                # Add enhancements to visualisation
                ax.set_xlabel('Date')
                ax.set_ylabel('Correlation Coefficient')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add title
                plt.title(f'{window_size_exp}-day Rolling Correlation between Exponential Weighted Score and Gold Price')
                
                # Display  statistics
                mean_corr_exp = valid_data_exp['rolling_corr_exp'].mean()
                pos_pct_exp = (valid_data_exp['rolling_corr_exp'] > 0).mean() * 100
                corr_stats_exp = f"Mean correlation: {mean_corr_exp:.3f} | Positive correlation: {pos_pct_exp:.1f}% of time"
                
                # Add annotation
                plt.figtext(0.5, 0.01, corr_stats_exp, ha='center', bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
                
                fig5.tight_layout(rect=[0, 0.03, 1, 0.95])  
                st.pyplot(fig5)
                
                # interpretation based on rolling correlation results
                st.markdown(f"""
                **Rolling Correlation Insights:**
                - The mean correlation between exponential weighted sentiment and gold prices over this period is **{mean_corr_exp:.3f}**.
                - Exponential weighted sentiment shows a positive correlation with gold prices **{pos_pct_exp:.1f}%** of the time.
                - {'This suggests that market sentiment generally drives gold prices in the same direction.' if pos_pct_exp > 60 else 
                'This suggests that market sentiment generally drives gold prices in the opposite direction.' if pos_pct_exp < 40 else
                'This suggests that the relationship between sentiment and gold prices is inconsistent or context-dependent.'}
                """)
            
            with tab4:
               
                st.subheader("Summary of Hypothesis Testing")
                
                # Check all test results for Raw Sentiment
                corr_supports_raw = abs(correlation) > 0.1 and correlation > 0 if 'correlation' in locals() else False
                regression_supports_raw = slope > 0 and p_value < 0.05 if 'slope' in locals() and 'p_value' in locals() else False
                rolling_supports_raw = mean_corr > 0 and pos_pct > 50 if 'mean_corr' in locals() and 'pos_pct' in locals() else False
                
                # Check all test results for Exponential Weighted Score
                corr_supports_exp = abs(exp_correlation) > 0.1 and exp_correlation > 0 if 'exp_correlation' in locals() else False
                regression_supports_exp = slope_exp > 0 and p_value_exp < 0.05 if 'slope_exp' in locals() and 'p_value_exp' in locals() else False
                rolling_supports_exp = mean_corr_exp > 0 and pos_pct_exp > 50 if 'mean_corr_exp' in locals() and 'pos_pct_exp' in locals() else False
                
                # Create summary table for Raw Sentiment based on results
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
                
                # Create summary table for Exponential Weighted Score based on the results
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
                
                # Overall counts
                raw_supports_count = sum([corr_supports_raw, regression_supports_raw, rolling_supports_raw])
                exp_supports_count = sum([corr_supports_exp, regression_supports_exp, rolling_supports_exp])



                st.subheader("Overall Conclusion")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Raw Sentiment Tests Supporting Hypothesis", f"{raw_supports_count}/3")
                with col2:
                    st.metric("Exponential Weighted Tests Supporting Hypothesis", f"{exp_supports_count}/3")
                
                # conclusion based on counts (results)
                if raw_supports_count > exp_supports_count:
                    st.success(f"**The Raw Sentiment Score shows stronger support ({raw_supports_count}/3 tests) for our hypothesis than the Exponential Weighted Score ({exp_supports_count}/3 tests).**")
                    better_method = "Raw Sentiment"
                elif exp_supports_count > raw_supports_count:
                    st.success(f"**The Exponential Weighted Score shows stronger support ({exp_supports_count}/3 tests) for our hypothesis than the Raw Sentiment Score ({raw_supports_count}/3 tests).**")
                    better_method = "Exponential Weighted"
                else:
                    st.info(f"**Both sentiment methods show equal support ({raw_supports_count}/3 tests) for our hypothesis.**")
                    better_method = "Both methods equally"
                
                
                st.subheader("Visual Comparison")
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
                
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure your data is properly formatted and try again.")
    else:
        st.error("Could not connect to BigQuery. Please check your credentials.")

