from mongoengine import Document, StringField, IntField, EmailField, connect, DecimalField
import asyncio
import websockets
import json

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

async def bybit_websocket():
    async with websockets.connect(BYBIT_WS_URL) as websocket:
        # Subscribe to 1-minute candlesticks for BTC/USDT
        subscribe_message = {
            "op": "subscribe",
            "args": ["kline.1.BTCUSDT"]
        }
        await websocket.send(json.dumps(subscribe_message))

        # Listen for messages
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)

                # Process and print the latest candle data
                if "data" in data:
                    candle = data["data"][0]
                    print(f"Time: {candle['start']}, Open: {candle['open']}, High: {candle['high']}, "
                          f"Low: {candle['low']}, Close: {candle['close']}, Volume: {candle['volume']}")

            except Exception as e:
                print(f"Error: {e}")
                break

# Run WebSocket client
asyncio.run(bybit_websocket())

