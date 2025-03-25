from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
#from forex_trading_env import ForexTradingEnv
#from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback
from mongoengine import Document, StringField, IntField, EmailField, connect, DecimalField
from time import sleep
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


global_balance = 10000
highest_balance = 10000
broker_fee_per_lot = 7
lot_size = 0.01
broker_fee = broker_fee_per_lot * lot_size
spread_cost = 0
end_session_balance = 9600

if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# Connect to MongoDB
connect(db="bybit", host="localhost", port=27017)

class Candlestick(Document):
    topic = StringField(required=True)
    startTime = IntField(required=True)
    stopTime = IntField(required=True)
    interval = IntField(required=True)
    openPrice = DecimalField(precision=4)
    highPrice = DecimalField(precision=4)
    lowPrice = DecimalField(precision=4)
    closePrice = DecimalField(precision=4)
    volume = DecimalField(precision=4)
    turnover = DecimalField(precision=4)
    confirm = StringField(required=True)
    currentTimestamp = IntField(required=True)




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

        # Define action space: 0 = Buy Long, 1 = Sell Short, 2 = Hold, 3 = Close
        self.action_space = Discrete(4)

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


        # Get the current and next prices
        current_price = self.data.iloc[self.current_step]["Close"]
        next_price = self.data.iloc[min(self.current_step + 1, len(self.data) - 1)]["Close"]
        RSI = self.data.iloc[self.current_step]["RSI"]
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
            else:
                reward = -1
        if action == 1:  # Buy
            if self.position == 0:  # Open a long positiona
                print("opening long position")
                self.position = 1
                self.last_price = current_price
                self.bot_entry_price = current_price
            #elif self.position == -1:  # Close a short position
            #    reward = self.last_price - current_price  # Profit from short
            #    self.balance += reward
            #    self.position = 0
            else:
                print("long position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
        elif action == 2:  # Sell
            if self.position == 0:  # Open a short position
                print("opening short position")
                self.position = -1
                self.last_price = current_price
                self.bot_entry_price = current_price
            elif self.position == -1 or self.position == 1:
                print("short position already open")
                unrealisedgains = self.unrealised_gains(current_price)
                reward = unrealisedgains
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
            elif self.position == 1:
                self.position = 0
                reward = ((current_price - self.bot_entry_price) * 100) - broker_fee - spread_cost
                self.balance += reward
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
        #global global_balance
        #global highest_balance
        #print("balance",global_balance)
        self.epoch += 1
        #if highest_balance > global_balance or global_balance < 10000 :
        #    print("Model is worse than previouse model")
        #else:
        #    print("Model is better keep")
        #    print(f"Epoch {self.epoch} completed.")
        #    model.save(f"ppo_scalping_model_{global_balance}_{self.epoch}.zip")
        #    shutil.move(f"ppo_scalping_model_{global_balance}_{self.epoch}.zip","models_to_test")
        #    highest_balance = global_balance


# Define the epoch logger callback
epoch_logger_callback = EpochLoggerCallback()

# Load Forex data
#data = pd.read_csv("20230101_20250131_xau_usd_data.csv")
candles = Candlestick.objects()
print(candles)
data = [candle.to_mongo().to_dict() for candle in candles]
data = pd.DataFrame(data)
#sys.exit()

data['RSI'] = indic.calculate_rsi(data)
data = indic.calculate_bollinger_bands(data)
data = indic.calculate_macd(data)
data = indic.calculate_ichimoku(data)
print(data)
#data = data.dropna()
#sys.exit()
#data = data.drop(columns=["Datetime"])
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

# Load the existing model
model = PPO.load("models/best_model.zip", env=env)
# Train the model
model.learn(total_timesteps=200000,callback=callback_list)

def plot_training_results(rewards_log, balance_log, save_path="training_plot.png"):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.plot(rewards_log, color="tab:blue", marker='o', linestyle='-', label="Reward")  # Add markers
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Balance", color="tab:green")
    ax2.plot(balance_log, color="tab:green", marker='x', linestyle='-', label="Balance")  # Add markers
    ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.tight_layout()
    plt.title("Training Performance: Reward & Balance Over Time")
    plt.savefig(save_path)  # Save plot to file
    print(f"Plot saved as {save_path}")  # Confirm file was saved

real_env = env.envs[0].env  # Unwrap the environment
plot_training_results(real_env.rewards_log, real_env.balance_log)

# Save the model for later use
#model.save("ppo_forex_trading")
#print("Model saved.")

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

