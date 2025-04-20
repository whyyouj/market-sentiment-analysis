import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from confluent_kafka import Producer, Consumer, KafkaException
import pandas as pd
import numpy as np
from airflow.scripts.feature_transformations import calculate_sentiment_score, calculate_rsi, add_bollinger_bands

# Configuration file path - stored as an Airflow Variable
DAG_CONFIG = Variable.get("gold_prediction_dag_config", deserialize_json=True)

# Define date range for historical data
START_DATE = datetime(2015, 11, 16)
END_DATE = datetime(2019, 2, 1)

def get_current_date():
    """Get the current date in the format YYYY-MM-DD"""
    return datetime.now().strftime('%Y-%m-%d')

def check_execution_date(**kwargs):
    """Check if the execution date is within our desired range"""
    execution_date = kwargs['execution_date']
    if START_DATE <= execution_date <= END_DATE:
        return True
    else:
        print(f"Execution date {execution_date} is outside our desired range. Skipping this run.")
        return False

def create_kafka_producer():
    """Create and return a Kafka producer instance"""
    kafka_config = {
        'bootstrap.servers': Variable.get('kafka_bootstrap_servers'),
        'client.id': 'gold-prediction-producer'
    }
    return Producer(kafka_config)

def create_kafka_consumer(group_id, topic):
    """Create and return a Kafka consumer instance"""
    kafka_config = {
        'bootstrap.servers': Variable.get('kafka_bootstrap_servers'),
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(kafka_config)
    consumer.subscribe([topic])
    return consumer

def get_mysql_extraction_query(cols_to_extract, date_format_func, mysql_table, execution_date):
    cols = ', '.join(cols_to_extract)
    match date_format_func:
        case "standard_date_format":
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Date = '{execution_date}'
            """
        case "mdy_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            formatted_date = f"{dt.month}/{dt.day}/{dt.year}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Date = '{formatted_date}'
            """
        case "dmy_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            formatted_date = f"{dt.day}/{dt.month}/{dt.year}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Dates = '{formatted_date}'
            """
        case "year_month_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            period = f"M{'0' + str(dt.month) if dt.month < 10 else str(dt.month)}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Year = {dt.year} and Period = '{period}'
            """
        case _:
            query = ""
    
    return query

def extract_from_mysql(dataset_config, execution_date, **kwargs):
    """Extract data from MySQL for a specific dataset and the given date"""
    dataset_name = dataset_config['name']
    mysql_table = dataset_config['mysql_table']
    date_format_func = dataset_config['date_format_function']
    cols_to_extract = dataset_config['columns_to_extract']
    
    # apply custom date formatting function for data source to get the correct mysql query
    sql_query = get_mysql_extraction_query(cols_to_extract, date_format_func, mysql_table, execution_date)
    
    # connect to MySQL
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    try:
        # execute query
        results = mysql_hook.get_pandas_df(sql_query)
        
        # if no data found for the date, create null record
        if results.empty:
            print(f"[MySQL] No data found for {dataset_name} on {execution_date}: Creating null record")
            # create a record with null values
            cols_string = ', '.join(cols_to_extract)
            columns = mysql_hook.get_pandas_df(f"SELECT {cols_string} FROM {mysql_table} LIMIT 0").columns
            null_record = {col: None for col in columns}
            result_dict = null_record
        else:
            print(f"[MySQL] Successfully extracted {dataset_name} data for {execution_date}")
            result_dict = results.iloc[0].to_dict()
        
        data_json = json.dumps(result_dict).encode('utf-8')
        
        # produce to Kafka
        producer = create_kafka_producer()
        producer.produce(
            dataset_config['kafka_topic'],
            key=dataset_name,
            value=data_json
        )
        producer.flush()
        
        print(f"[Kafka] Successfully produced {dataset_name} data to Kafka for {execution_date}")
        return True
    
    except Exception as e:
        print(f"Error extracting {dataset_name} data: {str(e)}")
        raise

def transform_data(dataset_config, **kwargs):
    """Consume data from Kafka and apply transformations"""
    dataset_name = dataset_config['name']
    kafka_topic = dataset_config['kafka_topic']
    transformation_function = dataset_config['transformation_function']
    
    # create consumer
    consumer = create_kafka_consumer(f"{dataset_name}-consumer-group", kafka_topic)
    
    try:
        # poll for message
        msg = consumer.poll(timeout=30.0)
        
        if msg is None:
            print(f"[Kafka] No message received for {dataset_name}")
            return None
        
        if msg.error():
            raise KafkaException(msg.error())
        
        # parse the message
        data_json = msg.value().decode('utf-8')
        data = json.loads(data_json)
        
        # apply specific transformation based on the data source
        if transformation_function == 'price_transform':
            data['Price'] = data['Price'].replace(',', '').astype(float)

        elif transformation_function == 'calculate_sentiment_score':
            # use hugging face transformer to compute sentiment score of news
            data['Sentiment_Score'] = calculate_sentiment_score(data['News'])
            del data['News']
        elif transformation_function == 'rename_key':
            # rename key to fit data source
            data['CPI'] = data['Value']
            del data['Value']
        
        # store the transformed data in XCom
        kwargs['ti'].xcom_push(key=dataset_name, value=json.dumps(data).encode('utf-8'))
        print(f"[Python] Successfully transformed {dataset_name} data")
        
        return True
    
    except Exception as e:
        print(f"Error transforming {dataset_name} data: {str(e)}")
        raise
    
    finally:
        # close the consumer
        consumer.close()

def combine_datasets(**kwargs):
    """Combine all the transformed datasets into a single row for BigQuery"""
    ti = kwargs['ti']
    datasets_config = DAG_CONFIG['datasets']
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d')
    
    # dictionary to hold the combined data
    combined_data = {'Date': execution_date}
    
    # retrieve each dataset from XCom
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        dataset_json = ti.xcom_pull(task_ids=f"transform_{dataset_name}", key=dataset_name)
        
        if dataset_json:
            data = json.loads(dataset_json)
            
            # add all values from this data source to the combined data
            for k, v in data.items():
                combined_data[k] = v     
    
    # store the combined data in XCom
    ti.xcom_push(key='combined_data', value=json.dumps(combined_data).encode('utf-8'))
    print(f"[Python] Successfully combined all datasets for {execution_date}")
    
    return True

def get_historical_data(date_str, max_window_size, bq_hook, project_id, dataset_id, table_id):
    """Get historical data from BigQuery for feature engineering"""
    # calculate the start date for historical data
    start_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=max_window_size)).strftime('%Y-%m-%d')
    
    # query historical data
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE Date >= '{start_date}' AND Date < '{date_str}'
    ORDER BY Date ASC
    """
    
    # execute the query
    result = bq_hook.get_pandas_df(query)
    
    return result if not result.empty else pd.DataFrame()

def apply_feature_engineering(hist_df, current_data):
    """Apply feature engineering to the combined dataset"""
    # create a DataFrame for the current day
    current_df = pd.DataFrame([current_data])
    
    # convert date to datetime
    if 'Date' in current_df.columns:
        current_df['Date'] = pd.to_datetime(current_df['Date'])
    
    if not hist_df.empty:
        # ensure date is in datetime format
        if 'Date' in hist_df.columns:
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        
        # combine historical data with current data
        combined_df = pd.concat([hist_df, current_df]).reset_index(drop=True)
    else:
        combined_df = current_df
    
    # forward fill missing values
    combined_df = combined_df.ffill()
    
    # apply feature engineering functions
    # exponential moving average for price
    combined_df["EMA30"] = combined_df[["Price"]].ewm(alpha=0.9, min_periods=30, adjust=False).mean()
    combined_df["EMA252"] = combined_df[["Price"]].ewm(alpha=0.9, min_periods=252, adjust=False).mean()

    # RSI
    combined_df['RSI'] = calculate_rsi(combined_df['Price'])

    # Bollinger Bands and Spread
    combined_df['Band_Spread'] = add_bollinger_bands(combined_df['Price'])
    # combined_df["Upper_Band"], combined_df["Lower_Band"] = add_bollinger_bands(combined_df['Price'])
    # combined_df['Band_Spread'] = combined_df["Upper_Band"] - combined_df["Lower_Band"]
    # combined_df = combined_df.drop(["Upper_Band", "Lower_Band"], axis = 1)
    
    # TODO: exponential weighted score for news sentiment score

    # return only the current day's data with all features
    return combined_df.iloc[-1:].reset_index(drop=True)

def load_to_bigquery(**kwargs):
    """Load the processed data to BigQuery"""
    ti = kwargs['ti']
    bigquery_config = DAG_CONFIG['bigquery']
    
    project_id = bigquery_config['project_id']
    dataset_id = bigquery_config['dataset_id']
    gold_market_data_table = bigquery_config['gold_market_data_table']
    
    # get combined data
    combined_data = ti.xcom_pull(task_ids='combine_datasets', key='combined_data')
    
    if not combined_data:
        print("[BigQuery] No data to load to BigQuery")
        return
    
    # get the date string and max_window_size
    max_window_size = max(DAG_CONFIG['window_sizes'].values())
    
    # connect to BigQuery
    bq_hook = BigQueryHook(bigquery_conn_id='bigquery_default')
    
    # get historical data for feature engineering
    hist_df = get_historical_data(combined_data['Date'], max_window_size, bq_hook, project_id, dataset_id, gold_market_data_table)
    
    # apply feature engineering
    processed_df = apply_feature_engineering(hist_df, combined_data)
    
    # load the processed data to BigQuery
    processed_df.to_gbq(
        destination_table=f"{dataset_id}.{gold_market_data_table}",
        project_id=project_id,
        if_exists='append',
        location='US'
    )
    
    print(f"[BigQuery] Loaded data for {combined_data['Date']} to BigQuery table {project_id}.{dataset_id}.{gold_market_data_table}")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# create the DAG
with DAG(
    'gold_price_prediction_pipeline',
    default_args=default_args,
    description='A DAG to gather and process gold-related financial data for ML predictions',
    schedule_interval='@daily',
    catchup=True,  # enable catchup to process historical data
) as dag:
    
    # start task
    start_task = DummyOperator(task_id='start')
    
    # check execution date task
    check_date_task = PythonOperator(
        task_id='check_execution_date',
        python_callable=check_execution_date,
        provide_context=True
    )
    
    # create dynamic tasks for each data source based on datasets config
    extract_tasks = []
    transform_tasks = []
    
    for dataset_config in DAG_CONFIG['datasets']:
        dataset_name = dataset_config['name']
        
        # extract task
        extract_task = PythonOperator(
            task_id=f"extract_{dataset_name}",
            python_callable=extract_from_mysql,
            op_kwargs={'dataset_config': dataset_config, 'execution_date': '{{ ds }}'},
            provide_context=True
        )
        extract_tasks.append(extract_task)
        
        # transform task
        transform_task = PythonOperator(
            task_id=f"transform_{dataset_name}",
            python_callable=transform_data,
            op_kwargs={'dataset_config': dataset_config},
            provide_context=True
        )
        transform_tasks.append(transform_task)
        
        # set extract + transform dependencies for each data source
        check_date_task >> extract_task >> transform_task
    
    # combine datasets task
    combine_task = PythonOperator(
        task_id='combine_datasets',
        python_callable=combine_datasets,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS # Only run if all upstream tasks succeed
    )
    
    # load to BigQuery task: includes fetching historical data, forward filling, feature engineering
    load_bq_task = PythonOperator(
        task_id='load_to_bigquery',
        python_callable=load_to_bigquery,
        provide_context=True
    )
    
    # end task
    end_task = DummyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED  # Run even if some tasks are skipped
    )
    
    # set the final dependencies
    start_task >> check_date_task
    transform_tasks >> combine_task >> load_bq_task >> end_task
