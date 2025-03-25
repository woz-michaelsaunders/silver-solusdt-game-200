import os
import re

# Define the directory containing the files
directory = '/home/michaelsaunders/solusd-rl-learn-2/models/'

# Pattern to match files and extract profit
pattern = re.compile(r'solusdt_scalping_model_(\d+\.\d+)_\d+\.zip')

# Dictionary to store filename and profit
profits = {}

# List files in the directory
for file in os.listdir(directory):
    match = pattern.match(file)
    if match:
        profit = float(match.group(1))
        profits[file] = profit

# Find the file with the highest profit
if profits:
    best_file = max(profits, key=profits.get)
    best_profit = profits[best_file]
    print(f"Most profitable model: {best_file}")
    print(f"Profit: {best_profit}")
else:
    print("No matching models found.")


