import redis

# Connect to Redis
redis_client = redis.Redis(host='192.168.1.87', port=6379, db=0)

# Define the stream name
stream_name = "oneminbtcusd"

# Add a message to the stream
message_id = redis_client.xadd(stream_name, {"user": "Alice", "message": "Hello, Redis Stream!"})

print(f"Message sent with ID: {message_id}")
