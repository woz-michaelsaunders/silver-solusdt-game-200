from mongoengine import Document, StringField, IntField, EmailField, connect, DecimalField, LongField, BooleanField
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
# Connect to MongoDB
connect(db="bybit", host="localhost", port=27017)



class Candlestick(Document):
    topic = StringField(required=True)
    startTime = LongField(required=True)
    stopTime = LongField(required=True)
    interval = IntField(required=True)
    openPrice = DecimalField(precision=4)
    closePrice = DecimalField(precision=4)
    highPrice = DecimalField(precision=4)
    lowPrice = DecimalField(precision=4)
    volume = DecimalField(precision=4)
    turnover = DecimalField(precision=4)
    confirm = BooleanField(required=True)
    currentTimestamp = LongField(required=True)
    rsi = IntField(required=False)
    middleBand = DecimalField(precision=4)
    standardDeviation = DecimalField(precision=4)
    upperBand = DecimalField(precision=6)
    lowerBand = DecimalField(precision=6)
    emaFast = DecimalField(precision=6)
    emaSlow = DecimalField(precision=6)
    macd = DecimalField(precision=6)
    signal = DecimalField(precision=6)
    histogram = DecimalField(precision=6)
    tenkanSen = DecimalField(precision=6)
    kijunSen = DecimalField(precision=6)
    senkouSpanA = DecimalField(precision=6)
    senkouSpanB = DecimalField(precision=6)
    chikouSpan = DecimalField(precision=6)


# Bybit API URL
BASE_URL = "https://api.bybit.com"

# Parameters for 1-minute historical data (last 1000 candles)
params = {
    "category": "linear",  # "linear" for USDT Perpetual, "inverse" for Coin-margined
    "symbol": "SOLUSDT",
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

def process_row(row):
    print(f"A: {row['startTime']}, B: {row['stopTime']}")
    startTime = row['startTime']
    stopTime = row['stopTime']
    print(startTime)
    print(stopTime)
    candle = Candlestick.objects(startTime=startTime,stopTime=stopTime).first()
    if candle:
        print("Found Candle")
        openPrice=row['open'],
        closePrice=row['close'],
        highPrice=row['high'],
        lowPrice=row['low'],
        volume=row['volume'],
        turnover=row['turnover'],
        confirm=True,
        currentTimestamp=startTime,
        rsi=row['RSI'],
        middleBand=row['Middle Band'],
        standardDeviation=row['Standard Deviation'],
        upperBand=row['Upper Band'],
        lowerBand=row['Lower Band'],
        emaFast=row['EMA_fast'],
        emaSlow=row['EMA_slow'],
        macd=row['MACD'],
        signal=row['Signal'],
        histogram=row['Histogram'],
        tenkanSen=row['Tenkan_sen'],
        kijunSen=row['Kijun_sen'],
        senkouSpanA=row['Senkou_Span_A'],
        senkouSpanB=row['Senkou_Span_B'],
        chikouSpan=row['Chikou_Span']
        candle.save()
        print(candle)
        #print("Saved the updated information for the current candle stick", candle.to_json())
        print("Candle Update")
    else:
        print("Not Found Candle")
        #print(float(df["open"]))
        candle = Candlestick(
            topic="kline.1.SOLUSDT",
            startTime=startTime,
            stopTime=stopTime,
            interval=1,  # Ensure it's an integer
            openPrice=row["open"],
            closePrice=row["close"],
            highPrice=row["high"],
            lowPrice=row["low"],
            volume=row["volume"],
            turnover=row["turnover"],
            confirm=True,
            rsi=row['RSI'],
            middleBand=row['Middle Band'],
            standardDeviation=row['Standard Deviation'],
            upperBand=row['Upper Band'],
            lowerBand=row['Lower Band'],
            emaFast=row['EMA_fast'],
            emaSlow=row['EMA_slow'],
            macd=row['MACD'],
            signal=row['Signal'],
            histogram=row['Histogram'],
            tenkanSen=row['Tenkan_sen'],
            kijunSen=row['Kijun_sen'],
            senkouSpanA=row['Senkou_Span_A'],
            senkouSpanB=row['Senkou_Span_B'],
            chikouSpan=row['Chikou_Span'],
            currentTimestamp=startTime
        )
        candle.save()
        print(candle)
        print("Candle created")

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
    one_days_ago = datetime.utcfromtimestamp(current_time_ms / 1000) - timedelta(hours=2)

    # Convert back to milliseconds
    one_days_ago_ms = int(one_days_ago.timestamp() * 1000)

    print(one_days_ago_ms)

    # Example usage in fetch_historical_klines
    historical_data = fetch_historical_klines("SOLUSDT", "1", one_days_ago_ms)

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


    df.apply(process_row, axis=1)
    print("finish kickoff")

initialkickoff()

while True:
    os.system('clear')
    # Example Usage
    
    print("Start Running every minute")
    # Current time in milliseconds
    current_time_ms = int(time.time() * 1000)

    # Subtract 7 days
    one_days_ago = datetime.utcfromtimestamp(current_time_ms / 1000) - timedelta(hours=1)

    # Convert back to milliseconds
    one_days_ago_ms = int(one_days_ago.timestamp() * 1000)

    print(one_days_ago_ms)

    # Example usage in fetch_historical_klines
    historical_data = fetch_historical_klines("SOLUSDT", "1", one_days_ago_ms)

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
    last_record = df.iloc[1]
    process_row(last_record)
    #df.apply(process_row, axis=1)

    time.sleep(60)
    print("Waiting 60 seconds")


