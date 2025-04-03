from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka import KafkaProducer, KafkaConsumer
import json
import pandas as pd
from google.cloud import bigquery
import os
from google.oauth2 import service_account
from airflow.providers.mongo.hooks.mongo import MongoHook


# Kafka Configurations
KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "gold-news-sentiment"

# BigQuery Configurations
BQ_PROJECT_ID = "skillful-mason-454117-m5"
BQ_DATASET = "IS3107_Project"
BQ_TABLE = "test_data"

# MongoDB and MySQL Configurations
MONGO_DB_NAME = "IS3107_Project"
MONGO_COLLECTION_NAME = "news_sentiment"
SQL_TABLE_NAME = 'gold_prices'

def read_data_from_mongodb():
    print("attempting to read data from MongoDB....")
    mongo_hook = MongoHook(conn_id="mongo_default")
    client = mongo_hook.get_conn()
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    data = collection.find()
    for document in data:
        print(document)
    print("Data read successfully!")

def produce_to_kafka():
    """Send sample market sentiment data to Kafka"""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )

    # sample_data = {
    #     "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #     "sentiment_score": 0.85,  # Example sentiment score
    #     "news_source": "Straits Times",
    #     "title": "Gold prices surge due to inflation concerns"
    # }
    
    # producer.send(KAFKA_TOPIC, sample_data)
    # producer.flush()

    # read data from mongodb and send to kafka
    print("attempting to read data from MongoDB....")
    mongo_hook = MongoHook(conn_id="mongo_default")
    client = mongo_hook.get_conn()
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    data = collection.find()
    # for document in data:
    #     producer.send(KAFKA_TOPIC, document)
    #     producer.flush()

    for document in data:
        # Convert MongoDB document to dict and handle non-serializable types
        document_dict = dict(document)
        # Remove _id field or convert it to string
        if '_id' in document_dict:
            document_dict['_id'] = str(document_dict['_id'])
        
        try:
            producer.send(KAFKA_TOPIC, document_dict)
            producer.flush()
        except Exception as e:
            print(f"Error sending document to Kafka: {e}")

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
    
    for i, message in enumerate(consumer):
        record = message.value
        record["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record["news_source"] = "Straits Times"
        record["title"] = "Gold prices surge due to inflation concerns"
        record["new_sentiment_score"] = record["sentiment_score"] * (i + 1)
        record["processed_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        processed_data.append(record)
        
        # # Stop after processing 10 messages
        # if len(processed_data) >= 10:
        #     break
    
    # Convert to DataFrame for further processing
    df = pd.DataFrame(processed_data)
    df.to_csv("/tmp/news_sentiment_extracted.csv", index=False)

def transform_func(path):
    df = pd.read_csv(path)
    # transformations using pandas etc
    
    df.to_csv("/tmp/news_sentiment_transformed.csv", index=False)

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
    
def setup_bigquery():
    """Create BigQuery dataset and table if they don't exist"""
    bq_client = get_bigquery_client()

    try:
        # Check if dataset exists
        dataset_ref = bq_client.dataset(BQ_DATASET)
        try:
            bq_client.get_dataset(dataset_ref)
            print(f"Dataset {BQ_DATASET} already exists")
        except Exception:
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # Set the dataset location
            dataset = bq_client.create_dataset(dataset)
            print(f"Dataset {BQ_DATASET} created")
        
        # Define table schema
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("sentiment_score", "FLOAT"),
            bigquery.SchemaField("news_source", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("processed_at", "TIMESTAMP"),
            bigquery.SchemaField("new_sentiment_score", "FLOAT")
        ]
        
        # Check if table exists
        table_ref = bq_client.dataset(BQ_DATASET).table(BQ_TABLE)
        
        try:
            bq_client.get_table(table_ref)
            print(f"Table {BQ_TABLE} already exists")
        except Exception:
            # Create table
            table = bigquery.Table(table_ref, schema=schema)
            table = bq_client.create_table(table)
            print(f"Table {BQ_TABLE} created")
            
    except Exception as e:
        print(f"Error setting up BigQuery: {e}")
        raise

def load_to_bigquery():
    bq_client = get_bigquery_client()

    """Load processed data into Google BigQuery"""
    df = pd.read_csv("/tmp/news_sentiment_extracted.csv")

    for dt in ['timestamp', 'processed_at', 'date']:
        df[dt] = pd.to_datetime(df[dt])

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

    # test_mongo = PythonOperator(
    #     task_id="test_mongo_connection",
    #     python_callable=lambda: MongoHook(conn_id="mongo_default").get_conn()
    # )
    
    # test_kafka = PythonOperator(
    #     task_id="test_kafka_connection",
    #     python_callable=lambda: KafkaProducer(bootstrap_servers=KAFKA_BROKER)
    # )

with DAG(
    "gold_news_dag",
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
