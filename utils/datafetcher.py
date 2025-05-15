import os
import time
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

# Compute the date 5 years ago from today
def get_five_years_ago_date() -> str:
    return (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

# Fetch top 50 symbols from NSE and convert to yfinance format (.NS)
def fetch_top50_nse() -> list[str]:
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    return [item['symbol'] + '.NS' for item in data]

# Fetch and store daily time series for a list of symbols
# Only keeps data since start_date (YYYY-MM-DD)
def fetch_and_store(symbols: list[str], start_date: str, output_dir: str = 'yahoostockdata') -> None:
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # fetch everything since start_date, including dividends and stock splits
            df = ticker.history(start=start_date, actions=True)
            if df.empty:
                print(f"No data for {symbol}, skipping.")
            else:
                df = df.reset_index()
                start_dt = pd.to_datetime(start_date).tz_localize('UTC')  # or match your system's timezone
                df = df[df['Date'] >= start_dt]

                out_path = os.path.join(output_dir, f"{symbol.replace('.NS','')}.csv")
                df.to_csv(out_path, index=False)
                print(f"Downloaded and saved: {symbol} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed for {symbol}: {e}")
        time.sleep(1)

# Job to refresh data
def refresh_stock_data():
    start_date = get_five_years_ago_date()
    symbols = fetch_top50_nse()
    print(f"Starting data fetch for {len(symbols)} symbols since {start_date}")
    fetch_and_store(symbols, start_date)
    print("Stock data refresh complete.")

if __name__ == "__main__":
    refresh_stock_data()
