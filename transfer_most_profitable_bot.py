import os
import re
import paramiko
from scp import SCPClient

# Define source directory and remote details
source_directory = '/home/michaelsaunders/solusd-rl-learn-2/models'
remote_host = '192.168.1.92'
remote_port = 22
remote_user = 'michaelsaunders'
remote_path = '/home/michaelsaunders/prod/best_model.zip'
ssh_key_path = '/home/michaelsaunders/.ssh/id_rsa'  # Path to private key

# Pattern to match filenames and extract profit
pattern = re.compile(r'solusdt_scalping_model_(\d+\.\d+)_\d+\.zip')

# Dictionary to store filename and profit
profits = {}

# Find the file with the highest profit
for file in os.listdir(source_directory):
    match = pattern.match(file)
    if match:
        profit = float(match.group(1))
        profits[file] = profit

if profits:
    # Find the best file
    best_file = max(profits, key=profits.get)
    best_profit = profits[best_file]
    print(f"Most profitable model: {best_file}")
    print(f"Profit: {best_profit}")

    # Create SSH connection using key-based authentication
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connecting to {remote_host} using key {ssh_key_path}...")
        pkey = paramiko.RSAKey.from_private_key_file(ssh_key_path)
        ssh.connect(
            hostname=remote_host,
            port=remote_port,
            username=remote_user,
            pkey=pkey
        )

        # Create SCP client and transfer file
        with SCPClient(ssh.get_transport()) as scp:
            file_path = os.path.join(source_directory, best_file)
            print(f"Transferring {file_path} to {remote_host}:{remote_path}")
            scp.put(file_path, remote_path)
            print("Transfer complete!")

    except Exception as e:
        print(f"Failed to transfer file: {e}")
    finally:
        ssh.close()

else:
    print("No matching models found.")
