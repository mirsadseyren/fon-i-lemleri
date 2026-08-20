import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

print("Downloading normally...")
df = yf.download("THYAO.IS", period="5d", interval="1h", progress=False, session=session)
print(df.tail(2))

last_date = df.index[-1]
start_date = last_date.strftime('%Y-%m-%d')
print(f"Downloading with start={start_date}...")
df2 = yf.download("THYAO.IS", start=start_date, interval="1h", progress=False, session=session)
print(df2.head(2))
