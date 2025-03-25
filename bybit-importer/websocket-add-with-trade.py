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
from urllib3.packages.six.moves import http_client

# Replace with your actual database credentials
USERNAME = "root"
PASSWORD = "michael"
HOST = "localhost"  # Change if using a remote server
PORT = "3306"
DATABASE = "bybit"
TABLE = "1msolusd"

# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

# SQL query to get the last 100 timestamps
#query = f"SELECT * FROM {TABLE} ORDER BY timestamp DESC LIMIT 50"
# Fetch data into Pandas DataFrame
#df = pd.read_sql(query, engine)
#print(df.head)

#Live information to track
position = 0
bot_entry_price = 0
last_action = 0
balance = 10000


#sys.exit()

global ws
end_session_balance = 9600

ws = WebSocket(
    testnet=False,
    channel_type="linear",
)

class ForexTradingEnv(Env):
    def __init__(self, data):
        super(ForexTradingEnv, self).__init__()

        # Store the data (e.g., historical FX rates)
        self.last_action = 0
        self.data = data
        self.current_step = 0
        self.initial_balance = 10000  # Starting balance
        self.balance = self.initial_balance
        self.position = 0  # 2 or short position and 1 if holding a long position, 0 otherwise
        self.entry_price = 0
        self.RSI = 0
        self.bot_entry_price = 0
        self.rewards_log = []
        self.balance_log = []
        self.bot_entry_price = 0
        self.current_step = 0

        # Define action space: 0 = Buy Long, 1 = Sell Short, 2 = Hold, 3 = Close
        self.action_space = Discrete(4)

        #print(self.data.shape[1])

        # Define observation space: Open, High, Low, Close, Volume
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 4,), dtype=np.float32
        )

    def _get_observation(self):
        # Append position and entry price to market data
        obs = np.append(self.data.iloc[self.current_step].values, [self.position, self.bot_entry_price,self.last_action, self.balance])
        # Debugging print to check shape
        print(f"Observation shape: {obs.shape}, Expected: {self.observation_space.shape}")
        return obs

    def reset(self, seed=None, options=None):
        # Get the first observation (Market data + Position + Entry Price)
        obs = np.append(self.data.iloc[self.current_step].values, [self.position, self.bot_entry_price, self.last_action, self.balance])
        return obs, {}

    def unrealised_gains(self,current_price):
        unrealised_gains_value = 0
        if self.position == 1: #Holding a long position
            print("        ------- Buy / Long  --------")
            print("        bought price:", self.bot_entry_price)
            print("        current price:", current_price)
            unrealised_gains_value = ((current_price - self.bot_entry_price) * 100) - broker_fee - spread_cost
            print("        Unreleased Gains:", unrealised_gains_value)
            unrealised_gains_value = round(unrealised_gains_value, 6)
            print("        Unrealised gains:", f"{unrealised_gains_value:.10f}")
        if self.position == -1: #holidng a short prsition
            print("        ------- Sell / Short  --------")
            print("        current price:", current_price)
            print("        bought price:", self.bot_entry_price)
            unrealised_gains_value = ((self.bot_entry_price - current_price) * 100) - broker_fee - spread_cost
            print("        Unreleased Gains:", unrealised_gains_value)
            print("        unrealised gains:", f"{unrealised_gains_value:.10f}")

        return unrealised_gains_value


    def step(self, action):
        self.last_action = action
        done = False
        reward = 0
        lot_size = 0.01
        actual_quantity = 100
        unrealisedgains = 0
        spread = 1.05
        position = {
		"lotsize": 0.01,
		"spread": 1.05,
		"action": action
	}
        # Get the current and next prices
        current_price = self.data.iloc[self.current_step]["closePrice"]
        position["current_price"] = current_price
        next_price = self.data.iloc[min(self.current_step + 1, len(self.data) - 1)]["closePrice"]
        RSI = self.data.iloc[self.current_step]["rsi"]
        if self.position == -1:
            print("----------- Short Position --------------")
        elif self.position == 1:
            print("----------- Long Position ---------------")
        else: 
            print("-------------------------------------")
        print("Bot entry price: ", self.bot_entry_price)
        print("current price: ",current_price)
        print("next price: ",next_price)
        print("action: " , action)
        print("balance: ", self.balance)
        print("rsi: ", RSI)

        # Action: Define action space: 0 = Buy Long, 1 = Sell Short, 2 = Hold, 3 = Close
        # Postion: 0 position, 1 short position and 2 if holding a long position
        # Handle actions
        if action ==0:
            print("holding")
            if self.position == -1 or self.position == 1:
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
                position['unrealised_gains'] = unrealisedgains
                position['action'] = 'Holding with position'
                position['reward'] = reward
            else:
                reward = -1
                position['action'] = 'Holding no position'
        if action == 1:  # Buy
            if self.position == 0:  # Open a long positiona
                print("opening long position")
                self.position = 1
                self.last_price = current_price
                self.bot_entry_price = current_price
                position['action'] = 'Buy Long Position' 
            #elif self.position == -1:  # Close a short position
            #    reward = self.last_price - current_price  # Profit from short
            #    self.balance += reward
            #    self.position = 0
            else:
                print("long position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
                position['action'] = 'Buy Long - position already open'
                position['reward'] = reward
        elif action == 2:  # Sell
            if self.position == 0:  # Open a short position
                print("opening short position")
                self.position = -1
                self.last_price = current_price
                self.bot_entry_price = current_price
                position['action'] = 'Sell Open Position'
            elif self.position == -1 or self.position == 1:
                print("short position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
                position['action'] = 'Sell - position open already'
                position['reward'] = reward
            #elif self.position == 1:  # Close a long position
                #reward = current_price - self.last_price  # Profit from long
                #self.balance += reward
                #self.position = 0
        

        elif action == 3:
            print("close position")
            if self.position == -1:
                self.position = 0
                reward = ((self.bot_entry_price - current_price) * 100) - broker_fee - spread_cost
                self.balance += reward
                position['action'] = 'Close sell Position'
                position['reward'] = reward
            elif self.position == 1:
                self.position = 0
                reward = ((current_price - self.bot_entry_price) * 100) - broker_fee - spread_cost
                self.balance += reward
                position['action'] = 'Close buy Position'
                position['reward'] = reward
            else:
                print("No position Open")
                reward = -1
            
        # Move to the next step
        self.current_step += 1
        #print("EPOC:",  self.epoch)

        if self.balance <= end_session_balance:
            self.done = True
            done = True
            reward -= 1000000
        if (self.balance - reward) < end_session_balance:
            self.done = True
            done = True
            reward -= 1000000


    
        print("next step:", self.current_step)
        if self.current_step >= len(self.data) - 1:
            truncated = True  # End the episode if we've reached the end of the data
        else:
            truncated = False

        print("reward:", f"{reward:.10f}")
        print("balance:", self.balance)

        print(f"Step {self.current_step}: Reward={reward}, Balance={self.balance}")

         # Define observation
        if not done:
            #observation = self.data.iloc[self.current_step, 1:].values
             observation = np.append(self.data.iloc[self.current_step].values, [self.position, self.bot_entry_price, self.last_action, self.balance])
        else:
            observation = np.zeros(self.observation_space.shape)
        
        info = {
            "balance": self.balance,
            "position": self.position,
            "bot_entry_price": self.bot_entry_price
        }

        # Log reward and balance
        self.rewards_log.append(reward)
        self.balance_log.append(self.balance) 

        pdresults = pd.DataFrame([position])
        pdresults.to_sql(name='results',con=engine, if_exists="append",index=False)

        # Update balance
        #self.balance += reward
        global global_balance
        global_balance = self.balance
        print(global_balance)
        print("-------------------------------------")
        #user_input = input("Press the space bar and hit Enter (type 'q' to quit): ")
        # Return Object Description
        #obs (Observation)   The new state/observation after taking action
        #reward (float)  The reward received for taking action
        #terminated (bool)   True if the episode ends naturally (goal reached, balance = 0, etc.)
        #truncated (bool)    True if the episode ends due to a time limit or constraints
        #info (dict) Extra diagnostic data (e.g., P&L, indicators, trade details)
        return observation, reward, done, truncated , info


    def render(self):
        print("RENDERING")
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")



def evaluate_trade(df):
    #print("in evaluate trade ",df)
    #write next action
    # Reload the environment

    env = ForexTradingEnv(df)
 
    # Load the trained model
    model = PPO.load(f"ppo_forex_trading.zip",env=env)

    #print(model.observation_space.shape)
    df["position"] = position
    df["bot_entry_price"] = bot_entry_price
    df["last_action"] = last_action
    df["balance"] = balance
    first_row = df.iloc[0].copy() 
  
    print("Ready to predict")
    # Test the agent
    action, _states = model.predict(first_row, deterministic=False)
    
    print("Current Position: ",  position)
    print("Balance: ", balance)
    match action:
      case 0:
        print("Holding position")
        if position == 1 or position == -1:
           unrealised_gains = env.unrealised_gains(first_row["closePrice"])
        else:
           unrealised_gains = env.unrealised_gains(first_row["closePrice"])
      case 1:
        print("Buy position")
      case 2:
        print("Sell position")
      case 3:
        print("Close position")

def handle_message(message):
    #print(message)
    try:
       #Ingest last bar
       data = message["data"][0]  # Extract first item from "data" array
       #print(f"Fetched {len(historical_data)} total candles.")
       #print(data) 
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
         #print(last_ingest.duplicated().sum())
         #print(last_ingest.head())	
         #print(last_ingest.dtypes)
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
         #print(df.head)
         #df["position"] = position
         #df["bot_entry_price"] = bot_entry_price
         #df["last_action"] = last_action
         #df["balance"] = balance

         first_row = df.iloc[0].copy()
         #print("first row:", first_row)
         evaluate_trade(df)
	
         


    except Exception as e:
        print("Error processing message:", e)




ws.kline_stream(
    interval=1,
    symbol="SOLUSDT",
    callback=handle_message
)
#df = []
#env = ForexTradingEnv(df)
#env = ForexTradingEnv(df)

# Load the trained model
#model = PPO.load(f"ppo_forex_trading.zip")

# Test the agent
#obs, _ = env.reset()
#done = False



while True:
    try:
        sleep(1)
    except KeyboardInterrupt:
        print("Closing WebSocket...")
        if ws:
            ws.exit()
        break

