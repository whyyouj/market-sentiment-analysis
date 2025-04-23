import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os
import time

# Create BigQuery client
@st.cache_resource(show_spinner=False)
def get_client(_creds, project):
    """Create and cache a BigQuery client"""
    if _creds is None:
        # Use default credentials from environment
        return bigquery.Client(project=project if project else None)
    else:
        return bigquery.Client(credentials=_creds, project=project if project else None)

# Run query with error handling and caching
@st.cache_data(ttl=600, show_spinner=False)
def run_query(_client, query, timeout=30):
    """Run a BigQuery query and return results as dataframe"""
    try:
        return _client.query(query).to_dataframe()
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return None

def debug_show_directory_info():
    """Show debugging information about directories and secrets file"""
    st.warning("🔍 DEBUG: Secrets File Location Check")
    
    # Show current working directory
    cwd = os.getcwd()
    st.write(f"Current working directory: `{cwd}`")
    
    # Check if .streamlit directory exists in current dir
    streamlit_dir = os.path.join(cwd, ".streamlit")
    if os.path.exists(streamlit_dir):
        st.write(f"✅ `.streamlit` directory exists at: `{streamlit_dir}`")
        
        # List all files in .streamlit directory
        files_in_streamlit = os.listdir(streamlit_dir)
        st.write("Files in `.streamlit` directory:")
        for file in files_in_streamlit:
            st.write(f"- `{file}`")
        
        # Check for secrets.toml specifically
        secrets_path = os.path.join(streamlit_dir, "secrets.toml")
        if os.path.exists(secrets_path):
            st.success(f"✅ `secrets.toml` file found at: `{secrets_path}`")
        else:
            st.error(f"❌ No `secrets.toml` file in `.streamlit` directory")
    else:
        st.error(f"❌ No `.streamlit` directory found at: `{streamlit_dir}`")
    
    # Check if secrets.toml exists in root dir (wrong location but common mistake)
    root_secrets = os.path.join(cwd, "secrets.toml")
    if os.path.exists(root_secrets):
        st.warning(f"⚠️ `secrets.toml` found in root directory: `{root_secrets}`")
        st.info("This is not the correct location. Please move it to the `.streamlit` folder.")
    
    # Check parent directory as well (sometimes needed in deployment environments)
    parent_dir = os.path.dirname(cwd)
    parent_streamlit_dir = os.path.join(parent_dir, ".streamlit")
    if os.path.exists(parent_streamlit_dir):
        parent_secrets = os.path.join(parent_streamlit_dir, "secrets.toml")
        if os.path.exists(parent_secrets):
            st.info(f"ℹ️ `secrets.toml` also found in parent directory: `{parent_secrets}`")
    
    # Show what Streamlit sees in st.secrets
    st.write("Secrets accessible to Streamlit:")
    try:
        # Don't show actual secret values, just keys
        secret_keys = list(st.secrets.keys()) if hasattr(st, 'secrets') else []
        if secret_keys:
            st.write(f"Available secret keys: {secret_keys}")
            if 'gcp_service_account' in secret_keys:
                st.success("✅ `gcp_service_account` key found in st.secrets!")
                if isinstance(st.secrets.gcp_service_account, dict):
                    gcp_keys = list(st.secrets.gcp_service_account.keys())
                    st.write(f"Keys in gcp_service_account: {gcp_keys}")
        else:
            st.error("❌ No secrets found in st.secrets object")
    except Exception as e:
        st.error(f"Error accessing st.secrets: {str(e)}")

def check_secrets_available():
    """Check if secrets are available for BigQuery"""
    try:
        return 'gcp_service_account' in st.secrets
    except:
        return False

def get_service_account_from_secrets():
    """Get service account credentials from st.secrets"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return credentials
    except Exception as e:
        st.error(f"Error loading credentials from secrets: {e}")
        return None

def app():
    st.title("🔍 BigQuery Connection Tester")
    st.write("Use this tool to test your connection to Google BigQuery and run sample queries.")

    # Debug directory information
    with st.expander("Debug: Check Secrets File Location", expanded=True):
        debug_show_directory_info()

    # Check if secrets are available
    has_secrets = check_secrets_available()

    # Setup authentication tabs
    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["Streamlit Secrets", "Service Account JSON", "Environment Variable"])

    credentials = None
    project_id = ""

    with auth_tab1:
        st.subheader("Option 1: Authenticate with Streamlit Secrets")
        
        if has_secrets:
            st.success("✅ Found BigQuery credentials in secrets.toml!")
            
            # Display some info about the credentials without revealing sensitive information
            if 'gcp_service_account' in st.secrets and 'project_id' in st.secrets['gcp_service_account']:
                project_id = st.secrets['gcp_service_account']['project_id']
                st.write(f"Project ID from secrets: `{project_id}`")
                
                if 'client_email' in st.secrets['gcp_service_account']:
                    email = st.secrets['gcp_service_account']['client_email']
                    st.write(f"Service Account Email: `{email}`")
            
            if st.button("Use Credentials from Secrets"):
                credentials = get_service_account_from_secrets()
                if credentials:
                    st.success("Service account credentials loaded successfully from secrets!")
        else:
            st.info("No BigQuery credentials found in secrets.toml file. To use this method:")
            st.code("""
# Create a .streamlit/secrets.toml file with:
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "your-private-key"
client_email = "your-service-account-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
""")

    with auth_tab2:
        st.subheader("Option 2: Authenticate with Service Account JSON")
        
        # File uploader or JSON input
        auth_method = st.radio("Choose input method", ["Upload JSON file", "Paste JSON content"])
        
        if auth_method == "Upload JSON file":
            uploaded_file = st.file_uploader("Upload service account JSON file", type=["json"])
            if uploaded_file is not None:
                try:
                    # Read the JSON content
                    credentials_json = json.load(uploaded_file)
                    credentials = service_account.Credentials.from_service_account_info(credentials_json)
                    if hasattr(credentials, 'project_id'):
                        project_id = credentials.project_id
                    st.success("Service account credentials loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading credentials: {e}")
        else:
            json_content = st.text_area("Paste service account JSON content", height=150)
            if json_content:
                try:
                    credentials_json = json.loads(json_content)
                    credentials = service_account.Credentials.from_service_account_info(credentials_json)
                    if hasattr(credentials, 'project_id'):
                        project_id = credentials.project_id
                    st.success("Service account credentials loaded successfully!")
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")

    with auth_tab3:
        st.subheader("Option 3: Authenticate with Environment Variable")
        st.write("If you've set the GOOGLE_APPLICATION_CREDENTIALS environment variable, use this option.")
        
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        st.text_input("Current GOOGLE_APPLICATION_CREDENTIALS path", value=cred_path, disabled=True)
        
        if st.button("Use Environment Variable"):
            try:
                credentials = None  # Will use default credentials from environment
                st.success("Using credentials from environment variable.")
            except Exception as e:
                st.error(f"Error using environment credentials: {e}")

    # Project selector (only shown when project_id is empty and credentials are loaded)
    project_id_input = ""
    if credentials is not None or auth_tab3.checkbox("Use environment credentials"):
        # Only prompt for project_id if it's not already set
        if not project_id:
            # Get the project from credentials or let user input it
            if credentials and hasattr(credentials, 'project_id'):
                default_project = credentials.project_id
            else:
                default_project = ""
                
            project_id_input = st.text_input("Project ID", value=default_project)
            if project_id_input:
                project_id = project_id_input

        # Test connection
        if project_id and st.button("Test Connection"):
            with st.spinner("Testing connection to BigQuery..."):
                try:
                    # Create a client and test the connection
                    client = get_client(credentials, project_id)
                    
                    # Try listing datasets as a simple test
                    datasets = list(client.list_datasets())
                    st.success(f"✅ Successfully connected to project: {project_id}")
                    
                    if datasets:
                        st.write(f"Available datasets:")
                        for dataset in datasets:
                            st.write(f"- {dataset.dataset_id}")
                    else:
                        st.info("No datasets found in this project.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    st.error("Please check your credentials and project ID.")

        # Query form
        st.divider()
        st.subheader("Run Custom Query")
        
        # Sample queries dropdown
        sample_queries = {
    "Select a sample query": "",
    
    "List all tables": "SELECT table_name FROM `IS3107_Project.INFORMATION_SCHEMA.TABLES`",
    
    "Explore gold_market_data schema": """
        SELECT 
            column_name, 
            data_type, 
            is_nullable
        FROM 
            `IS3107_Project.INFORMATION_SCHEMA.COLUMNS`
        WHERE 
            table_name = 'gold_market_data'
        ORDER BY
            ordinal_position
    """,
    
    "Preview gold_market_data": """
        SELECT *
        FROM `IS3107_Project.gold_market_data`
        LIMIT 100
    """,
    
    "Explore test_data schema": """
        SELECT 
            column_name, 
            data_type, 
            is_nullable
        FROM 
            `IS3107_Project.INFORMATION_SCHEMA.COLUMNS`
        WHERE 
            table_name = 'test_data'
        ORDER BY
            ordinal_position
    """,
    
    "Preview test_data": """
        SELECT *
        FROM `IS3107_Project.test_data`
        LIMIT 100
    """,
    
    "Explore exp_weighted_sentiment schema": """
        SELECT 
            column_name, 
            data_type, 
            is_nullable
        FROM 
            `IS3107_Project.INFORMATION_SCHEMA.COLUMNS`
        WHERE 
            table_name = 'exp_weighted_sentiment'
        ORDER BY
            ordinal_position
    """,
    
    "Preview exp_weighted_sentiment": """
        SELECT *
        FROM `IS3107_Project.exp_weighted_sentiment`
        LIMIT 100
    """,
    
    "Explore model_predictions schema": """
        SELECT 
            column_name, 
            data_type, 
            is_nullable
        FROM 
            `IS3107_Project.INFORMATION_SCHEMA.COLUMNS`
        WHERE 
            table_name = 'model_predictions'
        ORDER BY
            ordinal_position
    """,
    
    "Preview model_predictions": """
        SELECT *
        FROM `IS3107_Project.model_predictions`
        LIMIT 100
    """,
    
    "Gold price statistics": """
        SELECT
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            STDDEV(price) as std_dev_price
        FROM
            `IS3107_Project.gold_market_data`
        WHERE
            price IS NOT NULL
    """,
    
    "Gold price over time": """
        SELECT
            date,
            price,
            volume
        FROM
            `IS3107_Project.gold_market_data`
        ORDER BY
            date DESC
        LIMIT 100
    """,
    
    "Sentiment analysis summary": """
        SELECT
            AVG(sentiment_score) as avg_sentiment,
            MIN(sentiment_score) as min_sentiment,
            MAX(sentiment_score) as max_sentiment,
            COUNT(*) as total_records
        FROM
            `IS3107_Project.exp_weighted_sentiment`
    """,
    
    "Model predictions evaluation": """
        SELECT
            predicted_value,
            actual_value,
            ABS(predicted_value - actual_value) as prediction_error,
            (predicted_value - actual_value) / actual_value * 100 as percent_error
        FROM
            `IS3107_Project.model_predictions`
        LIMIT 100
    """,
    
    "Join gold prices with sentiment": """
        SELECT
            g.date,
            g.price as gold_price,
            s.sentiment_score,
            s.weighted_sentiment
        FROM
            `IS3107_Project.gold_market_data` g
        LEFT JOIN
            `IS3107_Project.exp_weighted_sentiment` s
        ON
            g.date = s.date
        ORDER BY
            g.date DESC
        LIMIT 100
    """,
    
    "Data completeness check": """
        SELECT
            'gold_market_data' as table_name,
            COUNT(*) as record_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as unique_dates
        FROM
            `IS3107_Project.gold_market_data`
        UNION ALL
        SELECT
            'exp_weighted_sentiment' as table_name,
            COUNT(*) as record_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as unique_dates
        FROM
            `IS3107_Project.exp_weighted_sentiment`
        UNION ALL
        SELECT
            'model_predictions' as table_name,
            COUNT(*) as record_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as unique_dates
        FROM
            `IS3107_Project.model_predictions`
    """
}

        
        selected_sample = st.selectbox("Sample queries", options=list(sample_queries.keys()))
        
        default_query = sample_queries[selected_sample] if selected_sample else ""
        query = st.text_area("SQL Query", value=default_query, height=150)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            limit_rows = st.number_input("Display rows limit", min_value=1, max_value=1000, value=100)
        
        if query and st.button("Run Query"):
            if not project_id and not project_id_input:
                st.error("Please provide a Project ID")
            else:
                with st.spinner("Running query..."):
                    try:
                        client = get_client(credentials, project_id or project_id_input)
                        start_time = time.time()
                        result_df = run_query(client, query)
                        query_time = time.time() - start_time
                        
                        if result_df is not None:
                            row_count = len(result_df)
                            st.success(f"Query completed in {query_time:.2f} seconds. Retrieved {row_count} rows.")
                            
                            # Show the results
                            st.dataframe(result_df.head(limit_rows))
                            
                            # Download link
                            if row_count > 0:
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="Download results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv",
                                )
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                        st.error("Please check your credentials and project ID.")
    else:
        st.info("Please provide authentication credentials and a project ID to test your BigQuery connection.")

    # Footer with helpful tips
    st.divider()
    st.caption("Helpful tips:")
    st.caption("1. Make sure your service account has the necessary permissions (BigQuery Data Viewer, BigQuery User, etc.)")
    st.caption("2. Check that your project ID is correct")
    st.caption("3. For large queries, consider using LIMIT to restrict the number of rows returned")
    st.caption("4. When using public datasets, make sure to use the full path: `project.dataset.table`")

# Create a function to get a BigQuery client using secrets for external import
def get_bigquery_client():
    """Returns a BigQuery client using credentials from secrets.toml"""
    if check_secrets_available():
        credentials = get_service_account_from_secrets()
        if credentials:
            project_id = st.secrets['gcp_service_account'].get('project_id', None)
            return get_client(credentials, project_id)
    return None

# Helper function to execute a BigQuery query and return results
def execute_query(query, client=None):
    """Execute a BigQuery query and return results as a dataframe"""
    if client is None:
        client = get_bigquery_client()
        
    if client is None:
        st.error("No BigQuery client available. Please check your credentials.")
        return None
        
    return run_query(client, query)

# This allows the file to be imported in app.py and also run directly
if __name__ == "__main__":
    st.set_page_config(
        page_title="BigQuery Connection Tester",
        page_icon="🔍",
        layout="wide"
    )
    app()
