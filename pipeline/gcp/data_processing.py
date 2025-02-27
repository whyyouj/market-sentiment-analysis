import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Define Google Cloud project and BigQuery dataset
PROJECT_ID = "market-sentiment-analysis"
TOPIC_ID = "sentiment-news"
BQ_TABLE = "market_data.sentiment_news"
BUCKET_NAME = f"{PROJECT_ID}-dataflow-temp"

class ProcessData(beam.DoFn):
    def process(self, element):
        import json
        data = json.loads(element.decode("utf-8"))
        
        # Transform data (compute final sentiment score)
        data["overall_sentiment_score"] = 100 + data.get("overall_sentiment_score", 0)
        
        return [data]

def run():
    options = PipelineOptions(
        streaming=True,
        project=PROJECT_ID,
        region="us-central1",
        runner="DataflowRunner",
        temp_location=f"gs://{BUCKET_NAME}/temp"
    )

    with beam.Pipeline(options=options) as p:
        (
            p
            | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(topic=f"projects/{PROJECT_ID}/topics/{TOPIC_ID}")
            | "Decode JSON" >> beam.ParDo(ProcessData())
            | "Write to BigQuery" >> beam.io.WriteToBigQuery(
                BQ_TABLE,
                schema="time_published:TIMESTAMP, title:STRING, summary:STRING, overall_sentiment_score:FLOAT",
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == "__main__":
    run()