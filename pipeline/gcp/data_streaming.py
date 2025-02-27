from kafka import KafkaProducer
import json
import time
import requests
from google.cloud import pubsub_v1

# Set up Google Cloud Pub/Sub
PROJECT_ID = "market-sentiment-analysis"
TOPIC_ID = "sentiment-news"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

# Set up Kafka Producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# define variables for alpha vantage api
av_api_key = "URVXYBR8SYPKRCQ0" # Need to do some sort of security for this

ticker = "GOLD"
time_from = "20190101T0000"
time_to = "20241231T2359"

# function to fetch news sentiment data
def get_news_sentiment(ticker, time_from, time_to, limit = 1000):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={av_api_key}&limit={limit}&time_from={time_from}&time_to={time_to}"
    response = requests.get(url)
    data = response.json()

    data_to_publish = [{"time_published": article["time_published"], 
                        "title": article["title"], 
                        "summary": article["summary"],
                        "overall_sentiment_score": article["overall_sentiment_score"]} for article in data["feed"]]
    return data

# stream data
while True:
    news_sentiments = get_news_sentiment(ticker=ticker, time_from=time_from, time_to=time_to, limit = 1000)
    for news in news_sentiments:
        producer.send("news_topic", news)
        publisher.publish(topic_path, json.dumps(news).encode("utf-8"))  # Send to Pub/Sub
    time.sleep(10)  # Stream every 10 sec
