from time import sleep
import sys
import threading
import requests
import pandas as pd
import datetime
import time
import json
import os
import indicators as indic
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
import pandas as pd

# Replace with your actual database credentials
USERNAME = "root"
PASSWORD = "your_password"
HOST = "localhost"  # Change if using a remote server
PORT = "3306"
DATABASE = "bybit"

# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{USERNAME}@{HOST}:{PORT}/{DATABASE}")

# Connect to MongoDB
#connect(db="bybit", host="localhost", port=27017)

# Bybit API URL
BASE_URL = "https://api.bybit.com"

# Parameters for 1-minute historical data (last 1000 candles)
params = {
    "category": "linear",  # "linear" for USDT Perpetual, "inverse" for Coin-margined
    "symbol": "BTCUSDT",
    "interval": "1",  # 1-minute candles
    "limit": 100000  # Max candles per request
}

# Make API request
response = requests.get(f"{BASE_URL}/v5/market/kline", params=params)

# Convert response to JSON
data = response.json()
#print(data)
# Convert to Pandas DataFrame
#df = pd.DataFrame(data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

# Convert 'timestamp' column to datetime
#df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

# Convert to timestamp in milliseconds
#df["startTime"] = df["timestamp"]
#df["stopTime"] = df["startTime"].astype(int) + 60000  # Add 1 minute for stopTime


def fetch_historical_klines(symbol, interval, start_time, end_time=None, max_candles=10000000):
    """Fetch historical kline (candlestick) data from Bybit with pagination."""

    all_candles = []
    limit = 1000  # Bybit's max limit per request

    while len(all_candles) < max_candles:
        params = {
            "category": "linear",  # USDT Perpetual
            "symbol": symbol,
            "interval": interval,
            "start": start_time,
            "limit": limit
        }
        
        # Convert to datetime
        showtime = datetime.utcfromtimestamp(start_time / 1000)
        print(showtime)

        if end_time:
            params["end"] = end_time  # Optional end time

        print(params)
        print(BASE_URL)

        req = requests.Request("GET", f"{BASE_URL}/v5/market/kline",  params=params)
        prepared = req.prepare()
        print(prepared.url)

        response = requests.get(f"{BASE_URL}/v5/market/kline", params=params)
        data = response.json()
        if "result" not in data or "list" not in data["result"]:
            print("Error fetching data:", data)
            break

        candles = data["result"]["list"]
    
        print(len(candles))
        if not candles:
            print("No more data available.")
            break

        if len(candles) < 5:
            break

        all_candles.extend(candles)
        print(f"Fetched {len(candles)} candles, Total: {len(all_candles)}")

        # Update `start_time` to continue pagination
        last_candle = candles[0]
        print(last_candle)
        start_time = int(last_candle[0]) + 1  # Move to next candle
        print(start_time)
        # Avoid API rate limits
        time.sleep(0.2)  

    return all_candles[:max_candles]  # Trim to max required candles


def initialkickoff():
    print("Initial Kickoff")
    os.system('clear')
    # Example Usage
    # Current time in milliseconds
    current_time_ms = int(time.time() * 1000)

    # Subtract 7 days
    one_year_ago = datetime.utcfromtimestamp(current_time_ms / 1000) - timedelta(days=1)

    # Convert back to milliseconds
    one_year_ago = int(one_year_ago.timestamp() * 1000)

    print(one_year_ago)

    # Example usage in fetch_historical_klines
    historical_data = fetch_historical_klines("BTCUSDT", "1", one_year_ago)

    #start_time = int(time.time() * 1000) - (1000 * 60 * 60)  # 1 hour ago (in ms)
    #historical_data = fetch_historical_klines("BTCUSDT", "1", 1737062655097)
    print(f"Fetched {len(historical_data)} total candles.")
    df = pd.DataFrame(historical_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    print(df.head)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    # Convert 'timestamp' column to datetime
    #df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df['RSI'] = indic.calculate_rsi(df)
    df = indic.calculate_bollinger_bands(df)
    df = indic.calculate_macd(df)
    df = indic.calculate_ichimoku(df)
    df.fillna(0, inplace=True)
    # Convert to timestamp in milliseconds
    df["startTime"] = df["timestamp"]
    df["stopTime"] = df["startTime"].astype(int) + 59999  # Add 1 minute for stopTime
    print(df.head)

    df.to_sql("1m-btcusd", con=engine, if_exists="replace", index=False)  # Use 'append' to add data without replacing
    print("finish kickoff")

initialkickoff()
print("Initial Kickoff")


