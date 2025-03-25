from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
#from forex_trading_env import ForexTradingEnv
#from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback
from time import sleep
import talib
import sys
import threading
import matplotlib.pyplot as plt
import indicators as indic
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
import json 
app_systems_file_path = "app-settings.json"

global_balance = 10000
highest_balance = 0

appinfo = {}
#setup to measure the highest balance so only keep the best models
if os.path.exists(app_systems_file_path):
    print("File exists")
    with open("app-settings.json", "r") as file:
        appinfo = json.load(file)
        highest_balance = appinfo["highest_balance"]
else:
    appinfo["highest_balance"] = 10000
    highest_balance = appinfo["highest_balance"]


broker_fee_per_lot = 0.1
lot_size = 1
broker_fee = broker_fee_per_lot * lot_size
spread_cost = 0
end_session_balance = 9600

# Replace with your actual database credentials
USERNAME = "root"
PASSWORD = "michael"
HOST = "192.168.1.84"  # Change if using a remote server
PORT = "3306"
DATABASE = "bybit"

# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")


if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")


# Convert NumPy types to Python native types
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  # Convert arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.int64, np.int32)):  # Convert NumPy numbers to Python numbers
        return obj.item()
    return obj  # Return as is if not a NumPy type



class ForexTradingEnv(Env):
    def __init__(self, data):
        super(ForexTradingEnv, self).__init__()
        self.stop_loss_price = 0
        self.take_profit_price = 0
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
        self.game_level = 1
        self.wins = 0
        self.losses = 0
        # Define action space: 0 = Buy Long, 1 = Sell Short, 2 = Hold, 3 = Close
        self.action_space = Discrete(2)

        # Define observation space: Open, High, Low, Close, Volume
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 4,), dtype=np.float32
        )

    def _get_observation(self):
        # Append position and entry price to market data
        obs = np.append(self.data.iloc[self.current_step].values, [self.position, self.bot_entry_price])
        # Debugging print to check shape
        print(f"Observation shape: {obs.shape}, Expected: {self.observation_space.shape}")
        return obs
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.bot_entry_price = 0
        self.rewards_log = []
        self.balance_log = []
        self.game_level = 1
        self.wins = 0 
        self.losses = 0
        # Get the first observation (Market data + Position + Entry Price)
        obs = np.append(self.data.iloc[self.current_step].values, [self.position, self.bot_entry_price, self.last_action, self.balance])
        return obs, {}

    def unrealised_gains(self,current_price):
        unrealised_gains_value = 0
        if self.position == 1: #Holding a long position
            print("        ------- Buy / Long  --------")
            print("        bought price:", self.bot_entry_price)
            print("        current price:", current_price)
            unrealised_gains_value = ((current_price - self.bot_entry_price)) - broker_fee - spread_cost
            print("        Unreleased Gains:", unrealised_gains_value)
            unrealised_gains_value = round(unrealised_gains_value, 6)
            print("        Unrealised gains:", f"{unrealised_gains_value:.10f}")
        if self.position == -1: #holidng a short prsition
            print("        ------- Sell / Short  --------")
            print("        current price:", current_price)
            print("        bought price:", self.bot_entry_price)
            unrealised_gains_value = ((self.bot_entry_price - current_price)) - broker_fee - spread_cost
            print("        Unreleased Gains:", unrealised_gains_value)
            print("        unrealised gains:", f"{unrealised_gains_value:.10f}")

        return unrealised_gains_value

    def calculate_trade_parameters(self,balance, risk_percent, entry_price, stop_loss_distance, is_long=True, spread=0.30, commission_rate=0.0006):
        """
            Calculate position size, stop-loss, and take-profit for SOL/USDT on Bybit.
    
        Args:
        balance (float): Account balance in USDT (e.g., 1000).
        risk_percent (float): Risk as percentage of balance (e.g., 30).
        entry_price (float): Entry price in USDT (e.g., 135).
        stop_loss_distance (float): Price distance to stop-loss (e.g., 5).
        is_long (bool): True for long, False for short.
        spread (float): Spread in USDT (default 0.30, adjust to Bybit).
        commission_rate (float): Taker fee rate (default 0.0006 for Bybit).
    
    Returns:
        dict: Trade parameters.
        """
        # Risk in USDT
        risk_usdt = balance * (risk_percent / 100)
    
        # Stop-loss price
        stop_loss = entry_price - stop_loss_distance if is_long else entry_price + stop_loss_distance
    
        # Position size in SOL
        position_size_sol = risk_usdt / stop_loss_distance
    
        # Risk in USDT (verify)
        raw_risk = position_size_sol * stop_loss_distance
    
        # Desired net reward (2:1)
        net_reward_usdt = raw_risk * 2
    
        # Fees
        entry_value = position_size_sol * entry_price
        entry_fee = entry_value * commission_rate
        total_fees = entry_fee * 2  # Rough estimate, refine with take-profit
    
        # Raw reward needed
        raw_reward_usdt = net_reward_usdt + total_fees
        raw_reward_per_sol = raw_reward_usdt / position_size_sol
    
        # Take-profit price
        take_profit = entry_price + raw_reward_per_sol if is_long else entry_price - raw_reward_per_sol
    
        # Recalculate fees with take-profit
        exit_value = position_size_sol * take_profit
        exit_fee = exit_value * commission_rate
        total_fees = entry_fee + exit_fee
    
        # Adjusted reward
        adjusted_reward = (position_size_sol * abs(take_profit - entry_price)) - total_fees
        effective_ratio = adjusted_reward / raw_risk
    
        return {
            'position_size_sol': position_size_sol,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_usdt': raw_risk,
            'adjusted_reward_usdt': adjusted_reward,
            'total_fees': total_fees,
            'effective_ratio': effective_ratio
        }


    def should_it_close(self,low,high):
        global game_level
        if (self.position == -1): # Short Position
            print("In short position")
            if (self.stop_loss_price <= high):
                reward = self.bot_entry_price - self.stop_loss_price
                self.bot_entry_price = 0
                self.position = 0 
                self.balance = self.balance - reward
                self.losses += 1
                self.game_level -= 1
                print("short position needs to be closed. Stop Loss Triggered")
                return reward
            else:
                print("short position is safe")
                return(0)
            if(self.take_profit_price <= low):
                print("Take profit trigger and needs to closed")
                reward = self.bot_entry_price - self.take_profit_price
                self.bot_entry_price = 0
                self.position = 0 
                self.balance = self.balance + reward
                self.wins += 1
                self.game_level += 1
                return reward
            else:
                print("take profit not trigger")
                return(0)
        if (self.position == 1):
            print("In long position")
            if (self.stop_loss_price >= low):
                print("Stop Loss Trigged. Position needs to be closed")
                reward = self.stop_loss_price - self.bot_entry_price
                self.bot_entry_price = 0 
                self.position = 0 
                self.balance = self.balance - reward
                self.losses += 1
                self.game_level -= 1
                return reward

            else:
                print("Long position is safe")
            if (self.take_profit_price >= high):
                print("Take Profit is triggered. Position needs to be closed")
                reward = self.take_profit_price - self.bot_entry_price
                self.bot_entry_price = 0
                self.position = 0 
                self.balance = self.balance + reward
                self.wins += 1
                self.game_level += 1
                return reward
            else:
                print("take profit not trigged")
                return(0)
        return(0)
    def step(self, action):
        self.last_action = action
        done = False
        reward = 0
        lot_size = 0.01
        actual_quantity = 100
        unrealisedgains = 0
        spread = 1.05
        global highest_balance

        print(model.observation_space.shape)
        

        #Tesing to see if this is true
        # Example: Long SOL/USDT
        test_balance = 1000.00
        test_risk_percent = 30
        test_entry_price = 135.00
        test_stop_loss_distance = 5.00

        result = self.calculate_trade_parameters(test_balance, test_risk_percent, test_entry_price, test_stop_loss_distance)
        print(f"Balanace: {test_balance} USDT")
        print(f"Position Size: {result['position_size_sol']:.2f} SOL")
        print(f"Entry Price: ${test_entry_price:.2f}")
        print(f"Stop-Loss: ${result['stop_loss']:.2f}")
        print(f"Take-Profit: ${result['take_profit']:.2f}")
        print(f"Risk: ${result['risk_usdt']:.2f}")
        print(f"Adjusted Reward: ${result['adjusted_reward_usdt']:.2f}")
        print(f"Total Fees: ${result['total_fees']:.2f}")
        print(f"Effective Ratio: {result['effective_ratio']:.2f}:1")

        reward = self.should_it_close(self.data.iloc[self.current_step]["Low"],self.data.iloc[self.current_step]["High"])
        print("should it close reward",reward)

        print(f"GAME LEVEL: {self.game_level}")
        print(f"wins: {self.wins}")
        print(f"losses: {self.losses}")


        # Get the current and next prices
        current_price = self.data.iloc[self.current_step]["closePrice"]
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
        if action == 0:  # Buy
            if self.position == 0:  # Open a long positiona
                print("opening long position")
                self.position = 1
                self.last_price = current_price
                self.bot_entry_price = current_price
                self.take_profit_price = self.bot_entry_price + 20
                self.stop_loss_price = self.bot_entry_price - 10 
            else:
                print("long position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
        elif action == 1:  # Sell
            if self.position == 0:  # Open a short position
                print("opening short position")
                self.position = -1
                self.last_price = current_price
                self.bot_entry_price = current_price
                self.take_profit_price = self.bot_entry_price - 20
                self.stop_loss_price = self.bot_entry_price + 10

            elif self.position == -1 or self.position == 1:
                print("short position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains

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
        print("Best Bot Record:" , highest_balance)
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



# Define a custom callback to track epochs
class EpochLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpochLoggerCallback, self).__init__(verbose)
        self.epoch = 0

    def _on_step(self) -> bool:
        # Called at every step; you can log or track information here
        global global_balance
        print(global_balance)
        return True

    def _on_rollout_start(self) -> None:
        # Called at the start of a new rollout (epoch)
        print(f"Starting epoch {self.epoch + 1}...")
        global global_balance
        global_balance = 10000
        print("Starting balance:", global_balance)
        #user_input = input("Press the space bar and hit Enter (type 'q' to quit): ")

    def _on_rollout_end(self) -> None:
        # Called at the end of each epoch (rollout collection phase)
        global global_balance
        global highest_balance
        global app_systems_file_path
        global appinfo
        print("balance",global_balance)
        self.epoch += 1
        if highest_balance > global_balance or global_balance < 10000 :
            print("Model is worse than previouse model")
        else:
            print("Model is better keep")
            print(f"Epoch {self.epoch} completed.")
            model.save(f"models/solusdt_scalping_model_{global_balance}_{self.epoch}.zip")
            highest_balance = global_balance
            appinfo["highest_balance"] = highest_balance
            with open(app_systems_file_path,"w") as file:
                json.dump(appinfo, file, indent=4,default=convert_numpy)
            global_balance = 10000

# Define the epoch logger callback
epoch_logger_callback = EpochLoggerCallback()

# Load Forex data
#data = pd.read_csv("20230101_20250131_xau_usd_data.csv")
#candles = Candlestick.objects().order_by('-startTime')[:100]

# Convert to a list (if needed)
#candles_list = list(candles)


# Loop available to print out prices if need. Leave here in case needed for future
#for candle in candles:
#    print(candle.topic, candle.openPrice, candle.closePrice)
#    sys.exit()

# Query and load data into a Pandas DataFrame
df = pd.read_sql("SELECT * FROM `fifteenminsolusdt` ORDER BY timestamp DESC ", con=engine)

# Display DataFrame
print(df.head())
data = pd.DataFrame(df)

data = data.rename(columns={
           "timestamp": "Timestamp",
           "open": "Open",
           "high": "High",
       "close": "closePrice",
           "volume": "Volume",
       "low": "Low",
         })
data = data[::-1].reset_index(drop=True)
data['rsi'] = indic.calculate_rsi(data)
data = indic.calculate_bollinger_bands(data)
data = indic.calculate_macd(data)
data = indic.calculate_ichimoku(data)
data['EMA_9'] = EMAIndicator(close=data['closePrice'], window=9).ema_indicator()
data['EMA_12'] = EMAIndicator(close=data['closePrice'], window=12).ema_indicator()
data['EMA_20'] = EMAIndicator(close=data['closePrice'], window=20).ema_indicator()
data['OBV'] = talib.OBV(data['closePrice'], data['Volume'])
data['ADX'] = talib.ADX(data['High'], data['Low'], data['closePrice'], timeperiod=14)
data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
data['CCI'] = talib.CCI(data['High'], data['Low'], data['closePrice'], timeperiod=14)
data['ATR'] = talib.ATR(data['High'], data['Low'], data['closePrice'], timeperiod=14)
data['MFI'] = talib.MFI(data['High'], data['Low'], data['closePrice'], data['Volume'], timeperiod=14)
# Calculate VWAP using `ta`
vwap = VolumeWeightedAveragePrice(
    high=data['High'],
    low=data['Low'],
    close=data['closePrice'],
    volume=data['Volume'],
    window=14
)
data['VWAP'] = vwap.volume_weighted_average_price()
data.drop('Timestamp',axis=1,inplace=True)
# Convert to timestamp in milliseconds
#data["startTime"] = data["Timestamp"]
#data["stopTime"] = data["startTime"].astype(int) + 59999

#sys.exit()
print(data.head(5))
#As new learning data is being added. Drop learning NA data
data = data.dropna()
#data = data.drop(columns=["_id","topic","confirm"])
pd.set_option('display.max_columns', None)

print(data.head())

# Create the custom Forex trading environment
env = ForexTradingEnv(data)


# Wrap the environment to handle vectorized operations (required by Stable-Baselines3)
env = make_vec_env(lambda: ForexTradingEnv(data), n_envs=1)
eval_env = make_vec_env(lambda: ForexTradingEnv(data), n_envs=1)


# Define the evaluation callback
eval_callback = EvalCallback(
    eval_env=eval_env,  # Separate evaluation environment
    best_model_save_path="./models/",  # Saves the best model here
    log_path="./logs/",  # Logs evaluation results
    eval_freq=50000,  # Evaluates every 50,000 steps
    deterministic=True,  # Uses deterministic actions during evaluation
    render=False  # No graphical rendering
)

# Combine both callbacks into a list
callback_list = CallbackList([eval_callback, epoch_logger_callback])


# Create the PPO model, specifying GPU usage with 'cuda'
#model = DQN(
#    "MlpPolicy",  # Use a Multi-Layer Perceptron policy
#    env,          # Pass the environment
#    verbose=1,    # Show training progress
#    exploration_fraction=0.3,
#    exploration_final_eps=0.1,
#    device="cuda" # Use GPU (ensure PyTorch detects your GPU)
#)

#model = PPO(
#    "MlpPolicy",  # Use a Multi-Layer Perceptron policy
#    env,  # Replace 'data' with your historical data
#    verbose=1,  # Set verbosity for logging
#    learning_rate=3e-4,  # You can tune this hyperparameter
#    n_steps=2048,  # Number of steps to run for each environment update
#    batch_size=64,  # Mini-batch size
#    gae_lambda=0.95,  # Lambda for GAE (Generalized Advantage Estimation)
#    gamma=0.99,  # Discount factor
#    ent_coef=0.05,  # Entropy coefficient
#)


# Load the existing model
if os.path.exists("models/solusd_best_model.zip"):
    print("File exists!")
    model = PPO.load("models/solusd_best_model.zip", env=env)
else:
    print("File does not exist.")
    model = PPO(
        "MlpPolicy",  # Use a Multi-Layer Perceptron policy
        env,  # Replace 'data' with your historical data
        verbose=1,  # Set verbosity for logging
        learning_rate=3e-4,  # You can tune this hyperparameter
        n_steps=2048,  # Number of steps to run for each environment update
        batch_size=64,  # Mini-batch size
        gae_lambda=0.95,  # Lambda for GAE (Generalized Advantage Estimation)
        gamma=0.99,  # Discount factor
        ent_coef=0.01,  # Entropy coefficient
        clip_range=0.2,
    )



# Train the model
model.learn(total_timesteps=500000,callback=callback_list)
# Save the model for later use
model.save("models/solusd_best_model")
print("Model saved.")

def plot_training_results(rewards_log, balance_log, save_path=None):
    fig, ax1 = plt.subplots()

    # Format filename with date and time
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        save_path = f"training_plot_{timestamp}.png"

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.plot(rewards_log, color="tab:blue", marker='o', linestyle='-', label="Reward")  # Add markers
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Balance", color="tab:green")
    ax2.plot(balance_log, color="tab:green", marker='x', linestyle='-', label="Balance")  # Add markers
    ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.tight_layout()
    plt.title("Training Performance: Balance Over Time")
    plt.savefig(save_path)  # Save plot to file
    print(f"Plot saved as {save_path}")  # Confirm file was saved
    
    # Convert balance_log to a DataFrame
    df1 = pd.DataFrame({'balance': balance_log,'batch': timestamp})

    # Save to MySQL
    df1.to_sql('balance_table_sol', con=engine, if_exists='replace', index=False)

    print("Balance log saved successfully!")

real_env = env.envs[0].env  # Unwrap the environment
plot_training_results(real_env.rewards_log, real_env.balance_log)


# Output the best epoch
#print(f"Best Epoch: {reward_callback.best_epoch}, Reward: {reward_callback.best_epoch_reward}")

# Plot rewards
#plt.plot(reward_callback.episode_rewards, label="Episode Rewards")
#plt.axvline(reward_callback.best_epoch, color="red", linestyle="--", label="Best Epoch")
#plt.title("Episode Rewards Over Time")
#plt.xlabel("Episode")
#plt.ylabel("Cumulative Reward")
#plt.legend()
#plt.show()

