import os
import time
from datetime import datetime
from google.cloud import bigquery
from airflow.models import Variable
import json
import subprocess
import sys

def clear_bigquery_tables(client, project_id, dataset_id):
    """Clear specific tables in BigQuery as requested"""
    # Clear ALL data in exp_weighted_sentiment table
    exp_weighted_query = f"""
    DELETE FROM `{project_id}.{dataset_id}.exp_weighted_sentiment` 
    WHERE true
    """
    client.query(exp_weighted_query).result()
    print("Cleared exp_weighted_sentiment table")
    
    # Clear ALL data in model_predictions table
    predictions_query = f"""
    DELETE FROM `{project_id}.{dataset_id}.model_predictions` 
    WHERE true
    """
    client.query(predictions_query).result()
    print("Cleared model_predictions table")
    
    # Clear ALL data in feature_importances table
    importances_query = f"""
    DELETE FROM `{project_id}.{dataset_id}.feature_importances` 
    WHERE true
    """
    client.query(importances_query).result()
    print("Cleared feature_importances table")
    
    # Clear data with Date > '2015-11-13' in gold_market_data table
    market_data_query = f"""
    DELETE FROM `{project_id}.{dataset_id}.gold_market_data` 
    WHERE Date > '2015-11-13'
    """
    client.query(market_data_query).result()
    print("Cleared gold_market_data table for dates after 2015-11-13")

def reset_training_status(client, project_id, dataset_id):
    """Reset the model training status table row"""
    current_time = datetime.now().isoformat()
    
    reset_query = f"""
    UPDATE `{project_id}.{dataset_id}.model_training_status`
    SET training_status = 'idle',
        start_time = NULL,
        end_time = NULL,
        updated_at = '{current_time}',
        triggered_by = 'system_init'
    WHERE status_id = 'model_training_status'
    """
    client.query(reset_query).result()
    print("Reset model_training_status table")

def setup_airflow_variable(config_file_path):
    """Set up the Airflow variable with the DAG configuration"""
    # Load the configuration file
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    # Set the Airflow variable
    Variable.set("gold_prediction_dag_config", config, serialize_json=True)
    print("Set gold_prediction_dag_config Airflow variable")

def run_airflow_dag():
    """Run the Airflow DAG from start_date to end_date"""
    # Call airflow CLI to run the DAG
    subprocess.run([
        "airflow", "dags", "backfill",
        "gold_price_prediction_pipeline",
        "--start-date", "2015-11-16",
        "--end-date", "2019-02-01",
        "--reset-dagruns"
    ])

def main():
    # Path to the DAG configuration file
    config_file_path = "../../config/gold_dag_config.json"
    
    # Load the configuration file to get project and dataset IDs
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    project_id = config["bigquery"]["project_id"]
    dataset_id = config["bigquery"]["dataset_id"]

    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Clear and reset BigQuery tables
    print("Clearing and resetting BigQuery tables...")
    clear_bigquery_tables(client, project_id, dataset_id)
    reset_training_status(client, project_id, dataset_id)
    
    # Set up the Airflow variable
    print("Setting up Airflow variable...")
    setup_airflow_variable(config_file_path)
    
    # Trigger the DAG
    print("Triggering DAG execution...")
    run_airflow_dag()
    
    print("DAG execution complete!")

if __name__ == "__main__":
    main()