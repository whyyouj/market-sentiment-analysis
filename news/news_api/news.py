import requests

url = ('https://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=f1bca23f41684f5dbaa541b52314c977')

response = requests.get(url)
response.json()