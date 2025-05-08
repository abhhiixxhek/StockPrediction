top_50_stocks = [
    'RELIANCE.BSE', 'TCS.BSE', 'HDFCBANK.BSE', 'INFY.BSE', 'ICICIBANK.BSE',
    'HINDUNILVR.BSE', 'ITC.BSE', 'SBIN.BSE', 'BHARTIARTL.BSE', 'ASIANPAINT.BSE',
    'KOTAKBANK.BSE', 'AXISBANK.BSE', 'LT.BSE', 'BAJFINANCE.BSE', 'BAJAJFINSV.BSE',
    'MARUTI.BSE', 'NTPC.BSE', 'SUNPHARMA.BSE', 'TITAN.BSE', 'ULTRACEMCO.BSE',
    'POWERGRID.BSE', 'HCLTECH.BSE', 'TECHM.BSE', 'WIPRO.BSE', 'NESTLEIND.BSE',
    'HDFCLIFE.BSE', 'INDUSINDBK.BSE', 'GRASIM.BSE', 'TATASTEEL.BSE', 'JSWSTEEL.BSE',
    'CIPLA.BSE', 'DRREDDY.BSE', 'BPCL.BSE', 'ONGC.BSE', 'ADANIENT.BSE',
    'ADANIPORTS.BSE', 'BAJAJ-AUTO.BSE', 'EICHERMOT.BSE', 'HINDALCO.BSE', 'DIVISLAB.BSE',
    'BRITANNIA.BSE', 'HEROMOTOCO.BSE', 'COALINDIA.BSE', 'SBILIFE.BSE', 'ICICIPRULI.BSE',
    'APOLLOHOSP.BSE', 'SHREECEM.BSE', 'M&M.BSE', 'TATAMOTORS.BSE', 'AMBUJACEM.BSE'
]
import os
import time
import pandas as pd
import requests

API_KEY = 'YWDGQD7KDNH6I7GQ'
os.makedirs('stockdata', exist_ok=True)

for symbol in top_50_stocks:
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&datatype=csv&outputsize=full'
    try:
        df = pd.read_csv(url)
        df = df[df['timestamp'] >= '2020-05-01']  # Filter last 1 year
        df.to_csv(f'stockdata/{symbol.replace(".BSE", "")}.csv', index=False)
        print(f"Downloaded: {symbol}")
        time.sleep(12)  # To avoid hitting API limit
    except Exception as e:
        print(f"Failed for {symbol}: {e}")
