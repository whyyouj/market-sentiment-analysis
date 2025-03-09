from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import requests
from airflow.providers.mongo.hooks.mongo import MongoHook
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# hide mongo credentials later
MONGO_ATLAS = "mongodb+srv://sinler:tGlS60LvOPMl8GuB@data-eng-project.7rwxv.mongodb.net/"
DATABASE_NAME = "IS3107_Project"
COLLECTION_NAME = "reddit_posts"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'depends_on_past': False,
    'email': ['example@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
}

@dag(
    dag_id="DAG-1",
    default_args=default_args, 
    schedule_interval='@daily', 
    catchup=False
)
def sentiment_pipeline():
    @task
    def insert_data_to_mongodb():
        mongohook = MongoHook(conn_id=MONGO_ATLAS)
        client = mongohook.get_conn()
        # client = pymongo.MongoClient("mongodb://mongodb:27017/")
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        sample_data = {
            "text": "Bitcoin is pumping!", 
            "sentiment_score": 0.8, 
            "sentiment": "positive"
            }
        
        collection.insert_one(sample_data)
        print("Document inserted successfully!")

    @task
    def read_data_from_mongodb():
        mongohook = MongoHook(conn_id=MONGO_ATLAS)
        client = mongohook.get_conn()
        # client = pymongo.MongoClient("mongodb://mongodb:27017/")
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        data = collection.find()
        for document in data:
            print(document)

    insert_data_to_mongodb()
    read_data_from_mongodb()  

pipeline_dag = sentiment_pipeline()


''' to be uncommented afterwards
def collect_reddit_data():
    print("scraping Reddit discussions...")

def collect_news_data():
    print("Fetching financial news...")

def fetch_stock_data():
    print("Fetching stock data...")

def preprocess_data():
    print("Preprocessing data...")

def analyse_sentiment(text):
    analyser = SentimentIntensityAnalyzer()
    sentiment_score = analyser.polarity_scores(text)['compound']
    print(f"Sentiment score: {sentiment_score}")
'''
'''
# store data to MongoDB




# Creating first task
start = DummyOperator(task_id='start', dag=dag)

# Creating second task
end = DummyOperator(task_id='end', dag=dag)

insert_task = PythonOperator(
    task_id='insert_data_to_mongodb',
    python_callable=insert_data_to_mongodb,
    dag=dag
)

read_data = PythonOperator(
    task_id='read_data_from_mongodb',
    python_callable=read_data_from_mongodb,
    dag=dag
)

# Setting the task dependencies
start >> insert_task >> read_data >> end
'''

'''
# Define the tasks
scrape_reddit = PythonOperator(task_id='collect_reddit_data', python_callable=collect_reddit_data, dag=dag)
fetch_news = PythonOperator(task_id='collect_news_data', python_callable=collect_news_data, dag=dag)
store_mongo = PythonOperator(task_id='store_to_mongo', python_callable=store_to_mongo, dag=dag)
preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)

# Sentiment Analysis task needs text input, modify to get dynamic input
# For example, passing a sample Reddit post or news text to be analyzed
analyse = PythonOperator(task_id='analyse_sentiment', python_callable=analyse_sentiment, op_args=["Bitcoin is pumping!"], dag=dag)

# Set the task dependencies
# [scrape_reddit, fetch_news] >> store_mongo >> preprocess >> analyse_sentiment'
'''