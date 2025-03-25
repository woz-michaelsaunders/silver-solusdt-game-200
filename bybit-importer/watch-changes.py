from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["bybit"]
collection = db["candlestick"]

with collection.watch() as stream:
    for change in stream:
        if change["operationType"] == "insert":
            print("New document inserted:", change["fullDocument"])
            # Run your Python logic here
