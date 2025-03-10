from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from bson import ObjectId

DATABASE_NAME = "IS3107_Project"
MONGO_COLLECTION_NAME = "news_sentiment"
SQL_TABLE_NAME = 'gold_prices'

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
        collection = db[MONGO_COLLECTION_NAME]

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
        collection = db[MONGO_COLLECTION_NAME]

        data = collection.find()
        for document in data:
            print(document)
        print("Data read successfully!")

    @task
    def insert_data_to_mysql():
        print("attampting to insert data to mysql....")
        sample_gold_data = {
            "date": "2025-03-10",
            "price": 2911.5,
            "currency": "USD"
        }
        mysql_hook = MySqlHook(mysql_conn_id="mysql_default")
        
        insert_query = f'''
            INSERT INTO {SQL_TABLE_NAME} (date, price, currency)
            VALUES (
                '{sample_gold_data["date"]}', 
                {sample_gold_data["price"]}, 
                '{sample_gold_data["currency"]}'
            );
        '''
        mysql_hook.run(insert_query)
        print(f"Gold price for {sample_gold_data['date']} inserted successfully!")
        return sample_gold_data

    @task
    def read_data_from_mysql(sample_gold_data):
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

    sample_data = insert_data_to_mongodb()
    read_data_from_mongodb(sample_data)

    sample_gold_data = insert_data_to_mysql()
    read_data_from_mysql(sample_gold_data)

sentiment_pipeline()