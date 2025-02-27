import os
import time
from google.cloud import pubsub_v1, bigquery, storage, iam_credentials
from google.api_core.exceptions import Conflict
from google.cloud.resourcemanager import ProjectsClient

# set GCP project details
PROJECT_ID = "market-sentiment-analysis"
TOPIC_ID = "sentiment-news"
DATASET_ID = "market_data"
TABLE_ID = "sentiment_news"
BUCKET_NAME = f"{PROJECT_ID}-dataflow-temp"
SERVICE_ACCOUNT = f"{PROJECT_ID}@appspot.gserviceaccount.com"

# set project as environment variable to authenticate with google cloud
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# initialise clients
pubsub_client = pubsub_v1.PublisherClient()
bigquery_client = bigquery.Client()
storage_client = storage.Client()
resource_client =  ProjectsClient() #resource_manager.Client()

# function to create pub/sub topic in gcp pub/sub
def create_pubsub_topic():
    topic_path = pubsub_client.topic_path(PROJECT_ID, TOPIC_ID)
    try:
        pubsub_client.create_topic(request={"name": topic_path})
        print(f"Pub/Sub topic '{TOPIC_ID}' created.")
    except Conflict:
        print(f"Pub/Sub topic '{TOPIC_ID}' already exists.")

# function to create BigQuery dataset & table
def create_bigquery_table():
    dataset_ref = bigquery_client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)

    # create dataset if it doesn't exist
    try:
        bigquery_client.create_dataset(bigquery.Dataset(dataset_ref), timeout=30)
        print(f"BigQuery Dataset '{DATASET_ID}' created.")
    except Conflict:
        print(f"Dataset '{DATASET_ID}' already exists.")

    # define schema for table
    schema = [
        bigquery.SchemaField("time_published", "TIMESTAMP"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("summary", "STRING"),
        bigquery.SchemaField("overall_sentiment_score", "FLOAT")
    ]

    # create table with defined schema if it doesn't exist
    try:
        table = bigquery.Table(table_ref, schema=schema)
        bigquery_client.create_table(table)
        print(f"BigQuery Table '{TABLE_ID}' created.")
    except Conflict:
        print(f"Table '{TABLE_ID}' already exists.")

# function to create google cloud storage bucket
def create_storage_bucket():
    try:
        bucket = storage_client.create_bucket(BUCKET_NAME, location="US")
        print(f"Storage Bucket '{BUCKET_NAME}' created.")
    except Conflict:
        print(f"Bucket '{BUCKET_NAME}' already exists.")

# function to assign IAM roles
def assign_iam_roles():
    iam_roles = [
        "roles/pubsub.editor",
        "roles/bigquery.admin",
        "roles/dataflow.admin",
        "roles/storage.admin"
    ]

    for role in iam_roles:
        os.system(f"gcloud projects add-iam-policy-binding {PROJECT_ID} "
                  f"--member=serviceAccount:{SERVICE_ACCOUNT} "
                  f"--role={role}")
        # sleep to avoid hitting API rate limits
        time.sleep(2)

    print("IAM Roles assigned.")


if __name__ == "__main__":
    # run all setup functions
    print("[STAGE] Setting up Google Cloud resources")
    create_pubsub_topic()
    create_bigquery_table()
    create_storage_bucket()
    assign_iam_roles()
    print("[STAGE] GCP setup complete!")
