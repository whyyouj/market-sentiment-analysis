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
            
            # Display basic information
            if not clean_data.empty:
                st.subheader("Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(clean_data))
                with col2:
                    st.metric("Date Range", f"{clean_data['Date'].min().date()} to {clean_data['Date'].max().date()}")
                with col3:
                    st.metric("Data Coverage", f"{len(clean_data)}/{len(data)} ({len(clean_data)/len(data)*100:.1f}%)")
                
                # Show data sample
                with st.expander("View data sample"):
                    st.dataframe(clean_data.head(10))
                
                # Analysis tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Analysis", "Time Series Analysis", "Correlation Analysis", "Impact on Price", "Hypothesis Testing"])
                
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
                    st.subheader("Time Series Analysis")
                    
                    # Create a figure for the sentiment time series
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot raw sentiment scores
                    ax.plot(clean_data['Date'], clean_data['Sentiment_Score'], 
                           label='Raw Sentiment Score', color='blue', alpha=0.5)
                    
                    # Plot exponential weighted scores
                    ax.plot(clean_data['Date'], clean_data['Exponential_Weighted_Score'], 
                           label='Exponential Weighted Score', color='red', 
                           linestyle='-', alpha=0.3)
                    
                    # Customize the plot
                    ax.set_title('Sentiment Score Over Time')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sentiment Score')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-tick labels for better readability
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Plot sentiment and price together
                    st.subheader("Sentiment Score vs Gold Price")
                    
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Plot sentiment on primary y-axis
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Sentiment Score', color='tab:blue')
                    ax1.plot(clean_data['Date'], clean_data['Exponential_Weighted_Score'], color='tab:blue', alpha=0.7, label='Sentiment')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    
                    # Create secondary y-axis for price
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Gold Price', color='tab:orange')
                    ax2.plot(clean_data['Date'], clean_data['Price'], color='tab:orange', alpha=0.7, label='Gold Price')
                    ax2.tick_params(axis='y', labelcolor='tab:orange')
                    
                    # Add legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    plt.title('Sentiment Score and Gold Price Over Time')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab3:
                    st.subheader("Correlation Analysis")
                    
                    # Create correlation matrix
                    corr_data = clean_data[['Price', 'Sentiment_Score', 'Exponential_Weighted_Score']].copy()
                    corr_matrix = corr_data.corr()
                    
                    # Plot correlation heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Explanation of findings
                    st.markdown("### Key Observations")
                    
                    # Calculate correlations explicitly for better readability
                    corr_sentiment_price = corr_data['Sentiment_Score'].corr(corr_data['Price'])
                    corr_exp_price = corr_data['Exponential_Weighted_Score'].corr(corr_data['Price'])
                    corr_sentiment_exp = corr_data['Sentiment_Score'].corr(corr_data['Exponential_Weighted_Score'])
                    
                    st.markdown(f"- Correlation between Raw Sentiment and Price: **{corr_sentiment_price:.3f}**")
                    st.markdown(f"- Correlation between Exponential Weighted Sentiment and Price: **{corr_exp_price:.3f}**")
                    st.markdown(f"- Correlation between Raw and Exponential Weighted Sentiment: **{corr_sentiment_exp:.3f}**")
                    
                    # Rolling correlation
                    st.subheader("Rolling Correlation Analysis")
                    
                    # Calculate rolling correlation
                    window_size = st.slider("Select window size (days)", min_value=10, max_value=100, value=30, step=5)
                    
                    # Compute rolling correlation
                    rolling_df = clean_data.copy()
                    rolling_df['rolling_corr'] = clean_data['Sentiment_Score'].rolling(window=window_size).corr(clean_data['Price'])
                    
                    # Filter to only valid correlation values
                    valid_indices = rolling_df['rolling_corr'].notna()
                    valid_dates = rolling_df.loc[valid_indices, 'Date']
                    valid_corr = rolling_df.loc[valid_indices, 'rolling_corr']
                    
                    # Plot rolling correlation
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(valid_dates, valid_corr, color='purple')
                    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax.set_xlabel('Date')
                    ax.set_ylabel(f'{window_size}-day Rolling Correlation')
                    ax.set_title(f'{window_size}-day Rolling Correlation between Sentiment and Gold Price')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Interpretation
                    mean_corr = valid_corr.mean()
                    st.markdown(f"**Average Rolling Correlation:** {mean_corr:.3f}")
                    
                    if abs(mean_corr) < 0.1:
                        st.info("The rolling correlation analysis shows a very weak relationship between sentiment and gold prices over time.")
                    elif mean_corr > 0:
                        st.success(f"The rolling correlation is generally positive ({mean_corr:.3f}), suggesting that positive sentiment tends to correlate with higher gold prices.")
                    else:
                        st.warning(f"The rolling correlation is generally negative ({mean_corr:.3f}), suggesting that positive sentiment tends to correlate with lower gold prices.")
                
                with tab4:
                    st.subheader("Impact on Price")
                    
                    # Calculate price changes
                    price_data = clean_data.copy()
                    price_data['Price_Change'] = price_data['Price'].diff()
                    price_data['Price_Change_Pct'] = price_data['Price'].pct_change() * 100
                    price_data = price_data.dropna()
                    
                    # Group by sentiment ranges and analyze price impact
                    st.markdown("### Price Changes by Sentiment Range")
                    
                    # Create sentiment bins
                    price_data['Sentiment_Bin'] = pd.cut(price_data['Sentiment_Score'], 
                                                        bins=[-1.01, -0.5, -0.1, 0.1, 0.5, 1.01],
                                                        labels=['Very Negative', 'Negative', 'Neutral', 
                                                            'Positive', 'Very Positive'])
                    
                    # Group by sentiment bins
                    sentiment_impact = price_data.groupby('Sentiment_Bin').agg({
                        'Price_Change': ['mean', 'std', 'count'],
                        'Price_Change_Pct': ['mean', 'std']
                    }).reset_index()
                    
                    # Format the table for display
                    formatted_impact = pd.DataFrame({
                        'Sentiment Range': sentiment_impact['Sentiment_Bin'],
                        'Count': sentiment_impact[('Price_Change', 'count')],
                        'Avg Price Change': sentiment_impact[('Price_Change', 'mean')].round(2),
                        'Avg Pct Change': sentiment_impact[('Price_Change_Pct', 'mean')].round(2),
                        'Std Dev': sentiment_impact[('Price_Change_Pct', 'std')].round(2)
                    })
                    
                    st.dataframe(formatted_impact)
                    
                    # Visualize average price change by sentiment bin
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Use the formatted impact dataframe for plotting
                    sns.barplot(x='Sentiment Range', y='Avg Pct Change', 
                               data=formatted_impact, ax=ax, palette='RdYlGn')
                    plt.title('Average Price Change % by Sentiment Range')
                    plt.ylabel('Average Price Change %')
                    plt.xlabel('Sentiment Range')
                    
                    # Add count labels on top of bars
                    for i, v in enumerate(formatted_impact['Count']):
                        ax.text(i, formatted_impact['Avg Pct Change'].iloc[i] + 0.1, 
                               f"n={v}", ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Regression analysis
                    st.subheader("Regression Analysis: Sentiment → Price")
                    
                    # Prepare data for regression
                    X = sm.add_constant(price_data['Sentiment_Score'])
                    y = price_data['Price_Change_Pct']
                    
                    # Run regression
                    model = sm.OLS(y, X).fit()
                    
                    # Display regression results
                    st.markdown("#### Regression Results")
                    st.text(model.summary().tables[1].as_text())
                    
                    # Plot regression line
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.regplot(x='Sentiment_Score', y='Price_Change_Pct', data=price_data, ax=ax, 
                               scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    plt.title('Regression: Sentiment Score vs Price Change %')
                    plt.xlabel('Sentiment Score')
                    plt.ylabel('Price Change %')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab5:
                    st.subheader("Gold Price Sentiment Hypothesis Testing")
                    
                    st.markdown("""
                    ### Hypothesis:
                    
                    **Sentiment indicators have a significant relationship with gold prices.**
                    
                    We'll test this hypothesis through multiple statistical approaches, examining:
                    
                    1. Immediate effects (same-day relationship)
                    2. Lagged effects (does sentiment predict future price changes?)
                    3. Granger causality (does sentiment help predict gold prices beyond what past gold prices predict?)
                    """)
                    
                    # Create a new DataFrame for hypothesis testing
                    hypo_data = clean_data.copy()
                    
                    # Calculate next day's price and price changes
                    hypo_data['Next_Day_Price'] = hypo_data['Price'].shift(-1)
                    hypo_data['Price_Change'] = hypo_data['Next_Day_Price'] - hypo_data['Price']
                    hypo_data['Price_Change_Pct'] = (hypo_data['Next_Day_Price'] / hypo_data['Price'] - 1) * 100
                    hypo_data['Price_Direction'] = np.where(hypo_data['Price_Change'] > 0, 1, 0)
                    hypo_data = hypo_data.dropna()
                    
                    # Create sentiment categories
                    hypo_data['Sentiment_Category'] = np.where(hypo_data['Sentiment_Score'] > 0.1, 'Positive',
                                                      np.where(hypo_data['Sentiment_Score'] < -0.1, 'Negative', 'Neutral'))
                    
                    # 1. Testing the main hypothesis: Do positive sentiment days lead to price increases?
                    st.markdown("#### 1. Do sentiment scores predict gold price movements?")
                    
                    # Count price directions by sentiment category
                    cross_tab = pd.crosstab(
                        hypo_data['Sentiment_Category'], 
                        hypo_data['Price_Direction'].map({1: 'Increase', 0: 'Decrease'}),
                        normalize='index'
                    ) * 100
                    
                    # Add raw counts to the table
                    raw_counts = pd.crosstab(hypo_data['Sentiment_Category'], 
                                           hypo_data['Price_Direction'].map({1: 'Increase', 0: 'Decrease'}))
                    
                    # Display formatted table with percentages and counts
                    formatted_table = pd.DataFrame({
                        'Sentiment': cross_tab.index,
                        'Price Increase %': cross_tab['Increase'].round(1),
                        'Price Decrease %': cross_tab['Decrease'].round(1),
                        'Increase Count': raw_counts['Increase'],
                        'Decrease Count': raw_counts['Decrease'],
                        'Total Days': raw_counts.sum(axis=1)
                    })
                    
                    st.dataframe(formatted_table)
                    
                    # Run chi-square test to check for statistical significance
                    chi2, p, dof, expected = stats.chi2_contingency(raw_counts)
                    
                    # Display chi-square test results
                    st.markdown(f"**Chi-square test results:** χ² = {chi2:.2f}, p-value = {p:.4f}")
                    
                    if p < 0.05:
                        st.success("**The relationship between sentiment and price direction is statistically significant**")
                    else:
                        st.warning("**The relationship between sentiment and price direction is not statistically significant**")
                    
                    # Visualize the relationship with a better chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Stacked percentage bar chart
                    cross_tab.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'])
                    
                    for i, category in enumerate(cross_tab.index):
                        ax.text(i, 50, f"n={raw_counts.loc[category].sum()}", 
                               ha='center', va='center', color='white', fontweight='bold')
                    
                    plt.title('Percentage of Price Increases vs Decreases Following Different Sentiment Days')
                    plt.ylabel('Percentage')
                    plt.xlabel('Sentiment Category')
                    plt.legend(title='Next Day Price')
                    plt.ylim(0, 100)
                    
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.1f%%', label_type='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 2. Examine average price changes following different sentiment categories
                    st.markdown("#### 2. Average Gold Price Changes Following Different Sentiment Categories")
                    
                    # Calculate average price changes by sentiment category
                    avg_changes = hypo_data.groupby('Sentiment_Category')['Price_Change_Pct'].agg(['mean', 'std', 'count']).reset_index()
                    avg_changes.columns = ['Sentiment Category', 'Avg % Change', 'Std Dev', 'Count']
                    
                    # Calculate confidence intervals
                    confidence = 0.95
                    avg_changes['Error Margin'] = stats.t.ppf((1 + confidence) / 2, avg_changes['Count'] - 1) * \
                                                  avg_changes['Std Dev'] / np.sqrt(avg_changes['Count'])
                    
                    # Round for display
                    avg_changes['Avg % Change'] = avg_changes['Avg % Change'].round(3)
                    avg_changes['Std Dev'] = avg_changes['Std Dev'].round(3)
                    avg_changes['Error Margin'] = avg_changes['Error Margin'].round(3)
                    
                    st.dataframe(avg_changes)
                    
                    # Run ANOVA test to check for statistical significance
                    groups = [hypo_data[hypo_data['Sentiment_Category'] == cat]['Price_Change_Pct'] 
                             for cat in ['Positive', 'Neutral', 'Negative']]
                    
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        # Display ANOVA results
                        st.markdown(f"**ANOVA test results:** F = {f_stat:.2f}, p-value = {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("**The differences in price changes across sentiment categories are statistically significant**")
                        else:
                            st.warning("**The differences in price changes across sentiment categories are not statistically significant**")
                    else:
                        st.warning("Not enough data to perform ANOVA test")
                    
                    # 3. Lagged effects - does sentiment predict future prices?
                    st.markdown("#### 3. Lagged Effects Analysis: Does Sentiment Predict Future Price Changes?")
                    
                    # Create lagged sentiment features
                    lag_data = clean_data.copy()
                    for lag in range(1, 6):
                        lag_data[f'Sentiment_Lag_{lag}'] = lag_data['Sentiment_Score'].shift(lag)
                    
                    # Calculate future price change 
                    lag_data['Future_Price_Change'] = lag_data['Price'].pct_change(-5) * 100  # 5-day future price change
                    
                    # Drop rows with NaN values
                    lag_data = lag_data.dropna()
                    
                    # Display correlation between lagged sentiment and future price changes
                    lag_corrs = [lag_data[f'Sentiment_Lag_{lag}'].corr(lag_data['Future_Price_Change']) for lag in range(1, 6)]
                    
                    # Create dataframe for display
                    lag_corr_df = pd.DataFrame({
                        'Lag (days)': range(1, 6),
                        'Correlation': lag_corrs
                    })
                    
                    st.dataframe(lag_corr_df)
                    
                    # Plot correlation by lag
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(lag_corr_df['Lag (days)'], lag_corr_df['Correlation'], color='skyblue')
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
                    plt.title('Correlation Between Lagged Sentiment and 5-day Future Price Change')
                    plt.xlabel('Sentiment Lag (days)')
                    plt.ylabel('Correlation Coefficient')
                    plt.xticks(range(1, 6))
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Find the maximum correlation
                    max_lag = lag_corr_df.iloc[lag_corr_df['Correlation'].abs().idxmax()]
                    
                    st.markdown(f"**Strongest correlation with future price change:** Lag {max_lag['Lag (days)']} days (r = {max_lag['Correlation']:.3f})")
                    
                    # 4. Granger causality test
                    st.markdown("#### 4. Granger Causality Test: Does Sentiment Help Predict Gold Prices?")
                    
                    # Prepare data for Granger causality test
                    granger_data = clean_data[['Date', 'Price', 'Sentiment_Score']].set_index('Date')
                    granger_data = granger_data.sort_index()
                    
                    # Calculate daily returns for stationarity
                    granger_data['Price_Change'] = granger_data['Price'].pct_change() * 100
                    granger_data = granger_data.dropna()
                    
                    # Test different maximum lags
                    max_lag = min(8, len(granger_data) // 10)  # Don't use too many lags with limited data
                    
                    try:
                        # Run Granger test (Sentiment -> Price)
                        gc_results = grangercausalitytests(
                            granger_data[['Sentiment_Score', 'Price_Change']], 
                            max_lag, 
                            verbose=False
                        )
                        
                        # Extract p-values for each lag
                        p_values = [gc_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                        
                        # Create dataframe for display
                        granger_results = pd.DataFrame({
                            'Lag': range(1, max_lag+1),
                            'P-Value': p_values,
                            'Significant at 5%': [p < 0.05 for p in p_values]
                        })
                        
                        st.dataframe(granger_results)
                        
                        # Check if any lag is significant
                        any_significant = any(granger_results['Significant at 5%'])
                        
                        if any_significant:
                            st.success("**Granger causality test indicates that sentiment helps predict future gold price changes**")
                            significant_lags = granger_results[granger_results['Significant at 5%']]['Lag'].tolist()
                            st.markdown(f"Significant lags: {', '.join(map(str, significant_lags))} days")
                        else:
                            st.warning("**Granger causality test does not show significant predictive power of sentiment on gold prices**")
                    
                    except Exception as e:
                        st.error(f"Granger causality test failed: {str(e)}")
                        st.info("This may be due to insufficient data or stationarity issues in the time series.")
                    
                    # 5. Summary of findings
                    st.subheader("Summary of Hypothesis Testing")
                    
                    # Check all test results 
                    chi_square_supports = p < 0.05 if 'p' in locals() else False
                    anova_supports = p_value < 0.05 if 'p_value' in locals() else False
                    granger_supports = any_significant if 'any_significant' in locals() else False
                    corr_supports = abs(corr_exp_price) > 0.1 if 'corr_exp_price' in locals() else False
                    lag_supports = max(abs(np.array(lag_corrs))) > 0.1 if 'lag_corrs' in locals() else False
                    
                    # Create summary table
                    summary_df = pd.DataFrame({
                        'Test': [
                            'Correlation Analysis', 
                            'Chi-Square Test', 
                            'ANOVA Test', 
                            'Lagged Correlation', 
                            'Granger Causality'
                        ],
                        'Finding': [
                            f"Correlation = {corr_exp_price:.3f}" if 'corr_exp_price' in locals() else "Not computed",
                            f"p-value = {p:.4f}" if 'p' in locals() else "Not computed",
                            f"p-value = {p_value:.4f}" if 'p_value' in locals() else "Not computed",
                            f"Max lag corr = {max(abs(np.array(lag_corrs))):.3f}" if 'lag_corrs' in locals() else "Not computed",
                            "Significant at some lags" if granger_supports else "Not significant"
                        ],
                        'Supports Hypothesis': [
                            "✓ Yes" if corr_supports else "✗ No",
                            "✓ Yes" if chi_square_supports else "✗ No",
                            "✓ Yes" if anova_supports else "✗ No",
                            "✓ Yes" if lag_supports else "✗ No",
                            "✓ Yes" if granger_supports else "✗ No"
                        ]
                    })
                    
                    st.dataframe(summary_df)
                    
                    # Overall conclusion
                    supports_count = sum([
                        corr_supports, 
                        chi_square_supports, 
                        anova_supports, 
                        lag_supports, 
                        granger_supports
                    ])
                    
                    if supports_count >= 3:
                        st.success(f"**Overall Conclusion:** {supports_count} out of 5 tests support our hypothesis that sentiment indicators derived from financial news have a significant relationship with gold prices.")
                    elif supports_count >= 1:
                        st.warning(f"**Overall Conclusion:** Only {supports_count} out of 5 tests support our hypothesis. The evidence is mixed and more data may be needed.")
                    else:
                        st.error("**Overall Conclusion:** None of the tests provide strong support for our hypothesis. The relationship between sentiment indicators and gold prices may be weaker than expected.")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    st.markdown("""
                    Based on our analysis, we recommend:
                    
                    1. **Trading Strategy Refinement:** 
                    - Consider sentiment indicators as a complementary signal rather than a primary one
                    - Focus on the exponentially weighted sentiment score which showed stronger correlations
                    - Use sentiment signals with appropriate lag periods for optimal predictive power
                    
                    2. **Further Research:**
                    - Expand the dataset with longer time periods to increase statistical power
                    - Investigate sentiment from additional sources (social media, forums, etc.)
                    - Explore non-linear relationships between sentiment and price movements
                    - Combine sentiment with technical indicators for improved predictive models
                    
                    3. **Risk Management:**
                    - Recognize that sentiment's impact on gold prices may vary in different market regimes
                    - Implement proper position sizing when using sentiment-based trading signals
                    - Continuously monitor the sentiment-price relationship for shifts in effectiveness
                    """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Could not connect to BigQuery. Please check your credentials.")

                        


if __name__ == "__main__":
    app()
