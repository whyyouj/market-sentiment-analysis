from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka import KafkaProducer, KafkaConsumer
import json
import pandas as pd
from google.cloud import bigquery
import os
from google.oauth2 import service_account
from airflow.providers.mysql.hooks.mysql import MySqlHook


# Kafka Configurations
KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "gold-prices"

# BigQuery Configurations
BQ_PROJECT_ID = "market-sentiment-analysis101"
BQ_DATASET = "gold_market_data"
BQ_TABLE = "gold_prices"

# MongoDB and MySQL Configurations
DATABASE_NAME = "IS3107_Project"
MONGO_COLLECTION_NAME = "news_sentiment"
SQL_TABLE_NAME = 'gold_prices'

def read_data_from_mysql(latest_gold_price):
    print("attempting to read data from mysql....")
    mysql_hook = MySqlHook(mysql_conn_id="mysql_default")
    
    select_query = f'''
        SELECT * FROM {SQL_TABLE_NAME} ORDER BY date DESC
    '''
    records = mysql_hook.get_records(select_query)
    print("gold prices:")
    for record in records:
        print(record)
    print("Data read successfully!")

def produce_to_kafka():
    """Send sample market sentiment data to Kafka"""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "sentiment_score": 0.85,  # Example sentiment score
        "news_source": "Straits Times",
        "title": "Gold prices surge due to inflation concerns"
    }
    
    producer.send(KAFKA_TOPIC, sample_data)
    producer.flush()

def consume_from_kafka():
    """Consume messages from Kafka and process them"""
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=10000
    )

    processed_data = []
    
    for message in consumer:
        record = message.value
        record["processed_at"] = datetime.now().isoformat()
        record["new_sentiment_score"] = record["sentiment_score"] + 100
        processed_data.append(record)
        
        # # Stop after processing 10 messages
        # if len(processed_data) >= 10:
        #     break
    
    # Convert to DataFrame for further processing
    df = pd.DataFrame(processed_data)
    df.to_csv("/tmp/news_sentiment_kafka.csv", index=False)

def transform_func(path):
    df = pd.read_csv(path)
    # transformations using pandas etc

    df.to_csv("/tmp/news_sentiment_kafka.csv", index=False)

# Use this instead of directly creating the client
def get_bigquery_client():
    """Create BigQuery client with explicit credentials"""
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    else:
        # Fallback to default credentials (will likely fail if no ADC is set up)
        return bigquery.Client()

def load_to_bigquery():
    bq_client = get_bigquery_client()

    """Load processed data into Google BigQuery"""
    df = pd.read_csv("/tmp/news_sentiment_kafka.csv")

    table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    
    job = bq_client.load_table_from_dataframe(df, table_id)
    job.result()  # Wait for the job to complete

    print(f"Loaded {len(df)} rows into {table_id}")

# Define DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1
}

with DAG(
    "gold_prices_dag",
    default_args=default_args,
    catchup=False
) as dag:

    task_produce = PythonOperator(
        task_id="produce_to_kafka",
        python_callable=produce_to_kafka
    )

    task_consume = PythonOperator(
        task_id="consume_from_kafka",
        python_callable=consume_from_kafka
    )

    task_load_bq = PythonOperator(
        task_id="load_to_bigquery",
        python_callable=load_to_bigquery
    )

    # Define task dependencies
    task_produce >> task_consume >> task_load_bq
