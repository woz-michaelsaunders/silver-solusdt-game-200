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

# Replace with your actual database credentials
USERNAME = "root"
PASSWORD = "michael"
HOST = "localhost"  # Change if using a remote server
PORT = "3306"
DATABASE = "bybit"

# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")



global ws


ws = WebSocket(
    testnet=True,
    channel_type="linear",
)



def handle_message(message):
    print(message)
    try:
       data = message["data"][0]  # Extract first item from "data" array
       print(data)

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

