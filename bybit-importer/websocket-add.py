from pybit.unified_trading import WebSocket
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
#from forex_trading_env import ForexTradingEnv
#from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback
from time import sleep
import sys
import threading
import matplotlib.pyplot as plt
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd
from enum import Enum, auto
import os
import shutil
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import Float, BigInteger
import pymysql
import sys
import indicators as indic

# Replace with your actual database credentials
USERNAME = "root"
PASSWORD = "michael"
HOST = "localhost"  # Change if using a remote server
PORT = "3306"
DATABASE = "bybit"
TABLE = "1mbtcusd"

# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}", echo=True)

# SQL query to get the last 100 timestamps
#query = f"SELECT * FROM {TABLE} ORDER BY timestamp DESC LIMIT 50"
# Fetch data into Pandas DataFrame
#df = pd.read_sql(query, engine)
#print(df.head)

#sys.exit()

global ws


ws = WebSocket(
    testnet=False,
    channel_type="linear",
)



def handle_message(message):
    print(message)
    try:
       #Ingest last bar
       data = message["data"][0]  # Extract first item from "data" array
       #print(f"Fetched {len(historical_data)} total candles.")
       print(data) 
       if (data["confirm"] == True):     
         last_ingest = pd.DataFrame([data], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
         last_ingest = last_ingest.rename(columns={
    	   "timestamp": "Timestamp",
    	   "open": "Open",
    	   "high": "High",
   	   "close": "closePrice",
    	   "volume": "Volume",
    	   "turnover": "Turnover",
	   "low": "Low"
         })
         last_ingest["Timestamp"] = last_ingest["Timestamp"].astype("float64")
         last_ingest["Open"] = last_ingest["Open"].astype("float64")
         last_ingest["High"] = last_ingest["High"].astype("float64")
         last_ingest["closePrice"] = last_ingest["closePrice"].astype("float64")
         last_ingest["Volume"] = last_ingest["Volume"].astype("float64")
         last_ingest["Turnover"] = last_ingest["Turnover"].astype("float64")
         last_ingest["Low"] = last_ingest["Low"].astype("float64")
         print(last_ingest.duplicated().sum())
         print(last_ingest.head())	
         print(last_ingest.dtypes)
         last_ingest.to_sql(name=TABLE,con=engine, if_exists="append",index=False,
           dtype={
           "Timestamp": BigInteger(),
           "Open": Float(),
           "High": Float(),
           "Low": Float(),
           "closePrice": Float(),
           "Volume": Float(),
           "Turnover": Float(),
           }
         )

	 #select the latest 50 signals so can update
         # SQL query to get the last 100 timestamps
         query = f"SELECT * FROM {TABLE} ORDER BY timestamp DESC LIMIT 100"
	 # Fetch data into Pandas DataFrame
         df = pd.read_sql(query, engine)
         pd.set_option('display.max_columns', None)
         #print(df.head)   
         df['closePrice'] = pd.to_numeric(df['closePrice'], errors='coerce')      
         # Convert 'timestamp' column to datetime
         #df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
         df = df[::-1].reset_index(drop=True)
         df['rsi'] = indic.calculate_rsi(df)
         df = indic.calculate_bollinger_bands(df)
         df = indic.calculate_macd(df)
         df = indic.calculate_ichimoku(df)
         df.fillna(0, inplace=True)
         # Convert to timestamp in milliseconds
         df["startTime"] = df["Timestamp"]
         df["stopTime"] = df["startTime"].astype(int) + 59999  # Add 1 minute for stopTime
         print(df.head)

    except Exception as e:
        print("Error processing message:", e)




ws.kline_stream(
    interval=1,
    symbol="BTCUSDT",
    callback=handle_message
)



while True:
    try:
        sleep(1)
    except KeyboardInterrupt:
        print("Closing WebSocket...")
        if ws:
            ws.exit()
        break

