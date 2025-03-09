from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.providers.mongo.hooks.mongo import MongoHook
from bson import ObjectId

DATABASE_NAME = "IS3107_Project"
COLLECTION_NAME = "news_sentiment"

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
    dag_id='a1',
    default_args=default_args,
    schedule='@daily',
    catchup=False
)

def sentiment_pipeline():
    @task
    def insert_data_to_mongodb():
        print("attampting to insert data to MongoDB....")
        sample_data = {
            "date": "2025-01-01",
            "sentiment_score": 0.8,
            "sentiment": "positive"
        }
        mongo_hook = MongoHook(conn_id="mongo_default")
        client = mongo_hook.get_conn()
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        result = collection.insert_one(sample_data)
        print("Document inserted successfully!")
        sample_data["_id"] = str(result.inserted_id)

        return sample_data

    @task
    def read_data_from_mongodb(sample_data):
        print("attempting to read data from MongoDB....")
        mongo_hook = MongoHook(conn_id="mongo_default")
        client = mongo_hook.get_conn()
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        data = collection.find()
        for document in data:
            print(document)
        print("Data read successfully!")

    sample_data = insert_data_to_mongodb()
    read_data_from_mongodb(sample_data)

sentiment_pipeline()