import requests
import json

av_api_key = "URVXYBR8SYPKRCQ0" # Need to do some sort of security for this

time_from = "20190101T0000"
time_to = "20241231T2359"

def get_news_sentiment(ticker, time_from, time_to, limit = 1000):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={av_api_key}&limit={limit}&time_from={time_from}&time_to={time_to}"
    response = requests.get(url)
    data = response.json()

    # Create filename with ticker
    filename = f"news_sentiment_{ticker}.json"

    # Save the JSON data to a file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Data saved to {filename}")
    return data

get_news_sentiment("EQX", time_from, time_to)

