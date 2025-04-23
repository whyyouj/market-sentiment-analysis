from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import json
from pathlib import Path

# Load config
CONFIG_PATH = Path("/opt/airflow/config/gold_dag_config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

DAG_CONFIG = load_config()

# API Datasets to Test (from your config)
API_DATASETS = [d for d in DAG_CONFIG['datasets'] if d['name'] in ['real_yields', 'vix', 'usd_rates']]

def extract_from_api_test(dataset_config, execution_date, **kwargs):
    """Test-only API extraction with no Kafka/Mysql dependencies"""
    dataset_name = dataset_config['name']
    print(f"\n=== Testing API Extraction for {dataset_name} ===")
    
    try:
        # Simulate execution date (or use actual from kwargs)
        test_date = execution_date
        
        if dataset_name == "real_yields":
            print("Fetching 10-Year TIPS yield from FRED...")
            df = pdr.DataReader('DFII10', 'fred', start=test_date, end='2015-11-17')
            print(df)
            result = {'DFII10': float(df.iloc[0]['DFII10'])} if not df.empty else None
            
        elif dataset_name == "vix":
            print("Fetching VIX from Yahoo Finance...")
            df = yf.download('^VIX', start=test_date, end='2015-11-17')
            print(df)
            result = {'VIX': float(df.iloc[0]['Close'])} if not df.empty else None
            
        elif dataset_name == "usd_rates":
            print("Fetching DXY from Yahoo Finance...")
            df = yf.download('DX-Y.NYB', start=test_date, end='2015-11-17')
            print(df)
            result = {'DXY': float(df.iloc[0]['Close'])} if not df.empty else None
        
        if result:
            print(f"Success - {dataset_name} data:", result)
            return result
        else:
            print(f"No data found for {dataset_name} on {test_date}")
            return {dataset_config['columns_to_extract'][0]: None}
            
    except Exception as e:
        print(f"API Extraction Failed for {dataset_name}: {str(e)}")
        raise

# Test DAG Configuration
with DAG(
    'test_api_extraction_only',
    start_date=datetime(2023, 3, 2),
    schedule_interval=None,
    catchup=False,
    default_args={
        'retries': 0,
        'provide_context': True
    }
) as dag:
    
    start_test = PythonOperator(
        task_id='start_test',
        python_callable=lambda: print("Starting API Extraction Tests")
    )
    
    test_tasks = []
    for dataset in API_DATASETS:
        task = PythonOperator(
            task_id=f"test_extract_{dataset['name']}",
            python_callable=extract_from_api_test,
            op_kwargs={
                'dataset_config': dataset,
                'execution_date': '2015-11-16'  # Fixed test date
            }
        )
        test_tasks.append(task)
    
    start_test >> test_tasks