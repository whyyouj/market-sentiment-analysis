import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account

# Files that store general useful functions for other code files 


# Use this to get authenicated bigquery function in other files
def get_bigquery_client():
    """Returns an authenticated BigQuery client using credentials from secrets.toml"""
    try:
        # Getting credentials from secret
        credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        
        
        project_id = st.secrets["gcp_service_account"]["project_id"]
        
        client = bigquery.Client(credentials=credentials, project=project_id)
        return client
    except Exception as e:
        st.error(f"Error authenticating with BigQuery: {str(e)}")
        
        st.info("Make sure your .streamlit/secrets.toml file contains the gcp_service_account section")
        return None

# Getting data from bigquery
def load_data(client):
    """Loads data from BigQuery"""
    try:
        query = """
            SELECT Date, DXY, DFII10, VIX, CPI, EMA30, EMA252, RSI, Band_Spread
            FROM `IS3107_Project.gold_market_data`
        """
        data = client.query(query).to_dataframe()
        
        # Converts Date to datetime if it's not converted
        if 'Date' in data.columns and data['Date'].dtype != 'datetime64[ns]':
            
            data['Date'] = pd.to_datetime(data['Date'])
            
        return data
    except Exception as e:
        st.error(f"Error querying BigQuery: {str(e)}")
        return None

# One of the plots that all features will have
def plot_distribution(data, column_name, forward_filled=False):
    """Function to plot distribution of a given column."""
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column_name], bins=30, kde=True)
    
    plt.title(f'Distribution of {column_name}' + (' (Forward-Filled)' if forward_filled else ''))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    
    # If forward-filled, add a note
    if forward_filled:
        plt.annotate('Note: This variable contains forward-filled values which may\ncreate artificial peaks in the distribution.', xy=(0.5, 0.01), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8), ha='center', fontsize=9)
    
    st.pyplot(plt)

# One of the plots that all features will have
def plot_time_series(data, column_name, date_column='Date', forward_filled=False):
    """Function to plot time series with selective highlighting of forward-filled segments."""
    if date_column in data.columns:
        # Copying data for processing
        plot_data = data[[date_column, column_name]].copy().sort_values(date_column)
        actual_ff_segments = []
        
        if forward_filled:
            # Finding FF values
            diffs = plot_data[column_name].diff()
            zero_diff = diffs == 0
            
            # Labelling start and end of FF segments, if start without end, add llast point
            segment_start = np.where((zero_diff) & (~zero_diff.shift(1).fillna(False)))[0]
            segment_end = np.where((zero_diff) & (~zero_diff.shift(-1).fillna(False)))[0]
            
            
            
            if len(segment_start) > len(segment_end):
                segment_end = np.append(segment_end, len(plot_data) - 1)
            
            # Filter out single point segments
            for start, end in zip(segment_start, segment_end):
                # Only keep segments with at least 2 points
                if end > start:  
                    actual_ff_segments.append((start, end))


        plt.figure(figsize=(10, 5))
        plt.plot(plot_data[date_column], plot_data[column_name], color='#1f77b4', linewidth=1.5, alpha=0.8, label='Data')
        
        # Highlight forward-filled segments, if any
        if forward_filled and len(actual_ff_segments) > 0:
            for start, end in actual_ff_segments:
                segment_x = plot_data[date_column].iloc[start:end+1]
                segment_y = plot_data[column_name].iloc[start:end+1]
                
                # Different color + thicker line for forward-filled segments
                plt.plot(segment_x, segment_y, color='orange', linewidth=2.5, alpha=0.9)
                
                # Vertical line added to start of segment to increase visbility 
                plt.axvline(x=segment_x.iloc[0], color='orange', linewidth=0.8, linestyle='--', alpha=0.6)
        
        plt.title(f'Time Series of {column_name}' + (' (with forward-filled segments highlighted)' if forward_filled and len(actual_ff_segments) > 0 else ''))
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.xticks(rotation=45)
        
        # Legend included if FF present
        if forward_filled and len(actual_ff_segments) > 0:
            plt.plot([], [], color='orange', linewidth=2.5, label='Forward-filled segments')
            plt.legend(loc='best')
            
            # Add annotation about the number of segments
            plt.annotate(f'{len(actual_ff_segments)} forward-filled segments detected',  xy=(0.02, 0.02), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
                         ha='left', fontsize=9)
        elif forward_filled:
            plt.annotate('No significant forward-filled segments detected',  xy=(0.02, 0.02), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5",fc="lightyellow", alpha=0.8),
                         ha='left', fontsize=9)
            
        plt.tight_layout()
        st.pyplot(plt)

# Finding FF
def detect_potential_forward_filling(data, column_name):
    """Attempts to detect if a column might contain forward-filled values"""
    # Finding duplicate values (excluding NaN)
    data_clean = data[column_name].dropna()
    if len(data_clean) < 2:
        return False, 0
    diffs = data_clean.diff() # -> calculate difference with consecutive values
    zero_diffs = (diffs == 0).sum()
    
    # Calculate percentage of zero differences
    zero_diff_pct = zero_diffs / (len(data_clean) - 1) * 100
    
# Indicate threshold here
    is_likely_ff = zero_diff_pct > 3.0
    
    return is_likely_ff, zero_diff_pct


# One of the plots that all features will have
def plot_boxplot(data, column_name, forward_filled=False):
    """Function to create a boxplot for a given column."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[column_name])
    plt.title(f'Boxplot of {column_name}' + (' (Forward-Filled)' if forward_filled else ''))
    
    if forward_filled:
        plt.annotate('Note: Forward-filled values may affect the appearance of outliers', xy=(0.5, 0.01), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),ha='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(plt)
    
    

# One of the functions that all features will run
def run_feature_analysis(data, column_name, known_forward_filled=False):
    """Common function to analyze a single feature/column"""
    st.header(f"Analysis for {column_name}")
    is_forward_filled = known_forward_filled
    
    # Warning for FF
    if is_forward_filled:
        st.warning(f"⚠️ {column_name} contains forward-filled values. Interpretation should account for this.")
    
    # Check for missing values
    missing_count = data[column_name].isna().sum()
    missing_percent = (missing_count / len(data)) * 100

    is_ff, consecutive_pct = detect_potential_forward_filling(data, column_name)
    
    # Use previously defined functions
    # Summary statistics
    st.subheader("Summary Statistics:")
    stats = data[column_name].describe()
    st.dataframe(stats)
    
    # Distribution plot
    st.subheader("Distribution Plot:")
    plot_distribution(data, column_name, is_forward_filled)
    
    # Time series plot
    st.subheader("Time Series Plot:")
    plot_time_series(data, column_name, forward_filled=is_forward_filled)
    
    
    # Additional analysis
    st.subheader("Additional Insights:")
    
    # Check for outliers using IQR method
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]

    
    # Add warning for forward-filled data and outliers
    if is_forward_filled and len(outliers) > 0:
        st.info("⚠️ Outlier detection may be affected by forward-filled values. Some 'outliers' may be artifacts of the filling method.")
    
    # Calculating Skewness and kurtosis
    skewness = data[column_name].skew()
    kurtosis = data[column_name].kurt()
    
    st.write(f"**Skewness:** {skewness:.4f}")
    st.write(f"**Kurtosis:** {kurtosis:.4f}")
    
    if abs(skewness) > 1:
        st.write("*The distribution is highly skewed.*")
    elif abs(skewness) > 0.5:
        st.write("*The distribution is moderately skewed.*")
    else:
        st.write("*The distribution is approximately symmetric.*")
        
    # Add warning for forward-filled data and distribution measures
    if is_forward_filled:
        st.info("⚠️ Skewness and kurtosis measures may be affected by forward-filled values.")